"""Learning-rate search with the same budget for every architecture.

Every candidate LR is trained by train_lora.py on the same fixed subset (train_fraction), for the same
number of epochs and with the same seed; the LR with the lowest final validation loss wins.
Using the final checkpoint rather than the best one keeps the criterion identical for all models.

Finished candidates are skipped and interrupted ones are resumed, so the search can be restarted.

Usage:
  python scripts/lr_search.py --config configs/train/pythia-1.4b.json --output runs/lr_search/pythia-1.4b
  python scripts/lr_search.py --config ... --output ... --lrs 1e-4 2e-4 5e-4 --set micro_batch_size=2
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

TRAIN = Path(__file__).resolve().parent / "train_lora.py"
DEFAULT_LRS = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3]


def read_final(run):
    """Return (status, final val_loss, row) for one candidate directory."""
    if (run / "diverged.json").exists():
        return "diverged", math.inf, json.loads((run / "diverged.json").read_text(encoding="utf-8"))
    if not (run / "summary.json").exists():
        return "incomplete", math.inf, None
    rows = [json.loads(line) for line in (run / "checkpoints/index.jsonl").read_text(encoding="utf-8").splitlines() if line]
    final = rows[-1]
    return "done", final["val_loss"], final


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lrs", type=float, nargs="+", default=DEFAULT_LRS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-fraction", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="passed through to train_lora.py")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    common = [f"train_fraction={args.train_fraction}", f"epochs={args.epochs}"] + args.set
    for lr in args.lrs:
        run = out / f"lr_{lr:g}"
        status, _, _ = read_final(run)
        if status in ("done", "diverged"):
            print(f"lr={lr:g}: {status}, skipped")
            continue
        cmd = [sys.executable, str(TRAIN), "--config", args.config, "--output", str(run), "--seed", str(args.seed)]
        if (run / "trainer_state.pt").exists():
            cmd.append("--resume")
        elif run.exists() and any(run.iterdir()):
            sys.exit(f"{run} has files but no trainer_state.pt; remove it and restart the search")
        for item in common + [f"lr={lr}"]:
            cmd += ["--set", item]
        print(f"lr={lr:g}: training")
        code = subprocess.run(cmd).returncode
        if code != 0 and not (run / "diverged.json").exists():
            sys.exit(f"lr={lr:g}: train_lora.py failed with exit code {code}")

    results = []
    for lr in args.lrs:
        status, val_loss, row = read_final(out / f"lr_{lr:g}")
        results.append({"lr": lr, "status": status, "final_val_loss": None if math.isinf(val_loss) else val_loss,
                        "gpu_seconds": row.get("gpu_seconds") if status == "done" else None})
    done = [r for r in results if r["status"] == "done"]
    best = min(done, key=lambda r: r["final_val_loss"]) if done else None
    on_edge = best is not None and best["lr"] in (min(args.lrs), max(args.lrs))
    summary = {
        "config": args.config, "seed": args.seed, "train_fraction": args.train_fraction, "epochs": args.epochs,
        "overrides": args.set, "criterion": "final validation loss", "results": results,
        "best_lr": best["lr"] if best else None, "best_on_grid_edge": on_edge,
        "total_gpu_seconds": sum(r["gpu_seconds"] or 0 for r in results),
    }
    (out / "lr_search.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    print(f"\n{'lr':>10} {'status':>10} {'val_loss':>10} {'gpu_s':>10}")
    for r in results:
        loss = f"{r['final_val_loss']:.4f}" if r["final_val_loss"] is not None else "-"
        gpu = f"{r['gpu_seconds']:.0f}" if r["gpu_seconds"] is not None else "-"
        print(f"{r['lr']:>10g} {r['status']:>10} {loss:>10} {gpu:>10}")
    if best:
        print(f"\nbest lr = {best['lr']:g}")
    if on_edge:
        print("WARNING: best LR is on the edge of the grid; extend the grid in that direction and rerun")


if __name__ == "__main__":
    main()
