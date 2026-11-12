"""Unattended pipeline: micro-batch probe, LR search and the main seed series for several models.

Stages run for all models before the next stage starts, so a model that cannot train at all
(missing kernels, out of memory) is found in the first minutes, not after a day of other runs.
A failed model is skipped in later stages; the remaining models continue.

The pipeline is restartable: finished work is skipped, interrupted training is resumed with --resume.
Progress is written to runs/run_all/status.json, child output to runs/run_all/logs/.

Usage:
  python scripts/run_all.py
  python scripts/run_all.py --configs configs/train/mamba-1.4b.json --stages probe lr
  python scripts/run_all.py --seeds 0 1 --set train_fraction=0.1     # overrides go to every stage
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
DEFAULT_CONFIGS = ["configs/train/pythia-1.4b.json", "configs/train/mamba-1.4b.json",
                   "configs/train/rwkv-4-1.5b.json", "configs/train/zamba2-1.2b.json"]
DEFAULT_LRS = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.runs = Path(args.runs_dir)
        self.home = self.runs / "run_all"
        (self.home / "logs").mkdir(parents=True, exist_ok=True)
        self.status_path = self.home / "status.json"
        self.status = {}  # describes the current invocation; finished work is re-marked as done when skipped
        self.failed = set()
        self.mbs = {}

    def mark(self, model, stage, state, detail=None):
        entry = {"status": state, "updated": now()}
        if detail is not None:
            entry["detail"] = detail
        self.status.setdefault(model, {})[stage] = entry
        self.status_path.write_text(json.dumps(self.status, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[{now()}] {model} {stage}: {state}" + (f" ({detail})" if detail else ""), flush=True)

    def fail(self, model, stage, detail):
        self.failed.add(model)
        self.mark(model, stage, "failed", detail)

    def call(self, cmd, log_name):
        """Run a child process with output appended to its log; retry once on failure."""
        log = self.home / "logs" / f"{log_name}.log"
        for attempt in range(1 + self.args.retries):
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"\n=== {now()} attempt {attempt + 1}: {' '.join(cmd)}\n")
                f.flush()
                code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
            if code == 0:
                return 0
            print(f"[{now()}] {log_name}: exit code {code}, see {log}", flush=True)
        return code

    def tail(self, log_name, lines=3):
        log = self.home / "logs" / f"{log_name}.log"
        text = log.read_text(encoding="utf-8", errors="replace").strip().splitlines() if log.exists() else []
        return " | ".join(text[-lines:])

    def overrides(self, model):
        items = list(self.args.set)
        if model in self.mbs:
            items.append(f"micro_batch_size={self.mbs[model]}")
        out = []
        for item in items:
            out += ["--set", item]
        return out

    # --- stages ------------------------------------------------------------------------------
    def probe(self, model, config):
        out = self.runs / "probe" / model
        result = out / "probe.json"
        if not result.exists():
            if out.exists():
                shutil.rmtree(out)  # an interrupted probe is cheap to repeat
            self.mark(model, "probe", "running")
            cmd = [sys.executable, str(SCRIPTS / "probe_microbatch.py"), "--config", config, "--output", str(out),
                   "--sizes", *map(str, self.args.probe_sizes)]
            for item in self.args.set:
                cmd += ["--set", item]
            if self.call(cmd, f"{model}-probe") != 0 or not result.exists():
                return self.fail(model, "probe", self.tail(f"{model}-probe"))
        probe = json.loads(result.read_text(encoding="utf-8"))
        size = probe.get("recommended_micro_batch_size")
        if size is None and any(r["status"] == "ok" and r.get("peak_mem_gb") is None for r in probe["results"]):
            size = json.loads(Path(config).read_text(encoding="utf-8"))["micro_batch_size"]  # CPU run, no memory data
        if size is None:
            errors = [r.get("stderr", "")[-300:] for r in probe["results"] if r["status"] != "ok"]
            return self.fail(model, "probe", f"no micro-batch fits; last error: {errors[-1] if errors else 'none'}")
        self.mbs[model] = size
        self.mark(model, "probe", "done", f"micro_batch_size={size}")

    def lr_search(self, model, config):
        out = self.runs / "lr_search" / model
        grid = sorted(set(self.args.lrs))
        for _ in range(1 + self.args.lr_extensions):
            self.mark(model, "lr", "running", f"grid {' '.join(f'{x:g}' for x in grid)}")
            cmd = [sys.executable, str(SCRIPTS / "lr_search.py"), "--config", config, "--output", str(out),
                   "--lrs", *map(str, grid), *self.overrides(model)]
            if self.call(cmd, f"{model}-lr") != 0:
                return self.fail(model, "lr", self.tail(f"{model}-lr"))
            summary = json.loads((out / "lr_search.json").read_text(encoding="utf-8"))
            best = summary["best_lr"]
            if best is None:
                return self.fail(model, "lr", "every LR diverged")
            if not summary["best_on_grid_edge"]:
                return self.mark(model, "lr", "done", f"best lr {best:g}")
            grid = sorted(set(grid + ([best * 2, best * 5] if best == max(grid) else [best / 2, best / 5])))
        self.fail(model, "lr", f"best lr {best:g} still on the grid edge after {self.args.lr_extensions} extensions")

    def main_series(self, model, config):
        lr_json = self.runs / "lr_search" / model / "lr_search.json"
        best = json.loads(lr_json.read_text(encoding="utf-8"))["best_lr"]
        for seed in self.args.seeds:
            stage = f"main_s{seed}"
            out = self.runs / "main" / model / f"s{seed}"
            if (out / "summary.json").exists():
                used = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))["config"]["lr"]
                if abs(used - best) > 1e-12:
                    return self.fail(model, stage, f"{out} was trained with lr {used:g}, best is {best:g}; "
                                                   "rename the old series folder and restart")
                self.mark(model, stage, "done")
                continue
            cmd = [sys.executable, str(SCRIPTS / "train_lora.py"), "--config", config, "--lr-from", str(lr_json),
                   "--seed", str(seed), "--output", str(out), *self.overrides(model)]
            if (out / "trainer_state.pt").exists():
                cmd.append("--resume")
            elif out.exists() and any(out.iterdir()):
                # crashed before the first checkpoint: keep the remains for inspection, start again
                out.rename(out.with_name(f"{out.name}.incomplete-{time.strftime('%Y%m%d-%H%M%S')}"))
            self.mark(model, stage, "running")
            code = self.call(cmd, f"{model}-s{seed}")
            if code != 0 and (out / "trainer_state.pt").exists() and "--resume" not in cmd:
                code = self.call(cmd + ["--resume"], f"{model}-s{seed}")
            if code != 0 or not (out / "summary.json").exists():
                return self.fail(model, stage, self.tail(f"{model}-s{seed}"))
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.mark(model, stage, "done", f"{summary['gpu_seconds'] / 3600:.2f} GPU-h")

    def run(self):
        models = [(Path(c).stem, c) for c in self.args.configs]
        stages = {"probe": self.probe, "lr": self.lr_search, "main": self.main_series}
        print(f"[{now()}] models: {', '.join(m for m, _ in models)}; stages: {' '.join(self.args.stages)}", flush=True)
        for stage in self.args.stages:
            for model, config in models:
                if model in self.failed:
                    self.mark(model, stage, "skipped", "an earlier stage failed")
                    continue
                if stage != "probe" and model not in self.mbs:
                    self.probe(model, config)  # restores the micro-batch size when earlier stages are not requested
                    if model in self.failed:
                        continue
                try:
                    stages[stage](model, config)
                except Exception as exc:  # a broken result file must not stop the other models
                    self.fail(model, stage, f"{type(exc).__name__}: {exc}")
        done = [m for m, _ in models if m not in self.failed]
        print(f"[{now()}] finished; ok: {', '.join(done) or 'none'}; failed: {', '.join(sorted(self.failed)) or 'none'}")
        return 1 if self.failed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--stages", nargs="+", default=["probe", "lr", "main"], choices=["probe", "lr", "main"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--lrs", type=float, nargs="+", default=DEFAULT_LRS)
    ap.add_argument("--lr-extensions", type=int, default=2, help="automatic grid extensions when the best LR is on the edge")
    ap.add_argument("--probe-sizes", type=int, nargs="+", default=[8, 4, 2, 1])
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="passed to every stage")
    sys.exit(Pipeline(ap.parse_args()).run())


if __name__ == "__main__":
    main()
