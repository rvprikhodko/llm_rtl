"""Find the largest micro_batch_size that fits into GPU memory in the worst case.

Each candidate trains a few steps on the longest training examples only, so every micro-batch
has near-maximal length and the measured peak memory is an upper bound for the real run.
The global batch size is untouched: a smaller micro-batch only adds gradient accumulation.

Usage:
  python scripts/probe_microbatch.py --config configs/train/pythia-1.4b.json --output runs/probe/pythia-1.4b
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

TRAIN = Path(__file__).resolve().parent / "train_lora.py"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 8, 4, 2, 1])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--memory-limit-gb", type=float, default=13.5,
                    help="accepted peak allocated memory; leaves room for the desktop and other users of the GPU")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="passed through to train_lora.py")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for item in args.set:
        key, _, value = item.partition("=")
        if key == "train_file":
            cfg["train_file"] = value
    records = [json.loads(line) for line in open(cfg["train_file"], encoding="utf-8") if line.strip()]
    records.sort(key=lambda r: len(r["Instruction"]) + len(r["Response"]), reverse=True)
    longest = out / "longest.jsonl"
    with open(longest, "w", encoding="utf-8", newline="\n") as f:
        for r in records[:cfg["global_batch_size"] * args.steps]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    results = []
    for size in args.sizes:
        if size > cfg["global_batch_size"]:
            continue
        run = out / f"mbs_{size}"
        if run.exists():
            sys.exit(f"{run} exists; use a new --output")
        cmd = [sys.executable, str(TRAIN), "--config", args.config, "--output", str(run)]
        for item in args.set + [f"train_file={longest}", f"micro_batch_size={size}", f"max_steps={args.steps}",
                                "train_fraction=1.0", "log_every_steps=1", "save_every_epoch_fraction=null", "save_every_tokens=null",
                                "save_every_gpu_seconds=1000000000"]:
            cmd += ["--set", item]
        print(f"micro_batch_size={size}: probing")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            oom = "out of memory" in proc.stderr.lower()
            results.append({"micro_batch_size": size, "status": "oom" if oom else "error", "stderr": proc.stderr[-1500:]})
            print(f"  {'OOM' if oom else 'error'}")
            continue
        rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
        peak = json.loads((run / "summary.json").read_text(encoding="utf-8")).get("peak_mem_gb")
        tps = max(r["tokens_per_second"] for r in rows[1:]) if len(rows) > 1 else rows[-1]["tokens_per_second"]
        results.append({"micro_batch_size": size, "status": "ok", "peak_mem_gb": peak, "tokens_per_second_long": tps})
        print(f"  peak {peak} GB, {tps} tokens/s on longest examples")

    fitting = [r for r in results if r["status"] == "ok" and r["peak_mem_gb"] is not None and r["peak_mem_gb"] <= args.memory_limit_gb]
    best = max(fitting, key=lambda r: r["micro_batch_size"]) if fitting else None
    summary = {"config": args.config, "memory_limit_gb": args.memory_limit_gb, "results": results,
               "recommended_micro_batch_size": best["micro_batch_size"] if best else None}
    (out / "probe.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if best:
        print(f"\nrecommended micro_batch_size = {best['micro_batch_size']}")
    elif any(r["status"] == "ok" and r["peak_mem_gb"] is None for r in results):
        print("\npeak memory is unavailable: the probe ran without CUDA")
    else:
        print("\nno micro_batch_size fits; check errors in probe.json")


if __name__ == "__main__":
    main()
