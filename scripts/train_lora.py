"""LoRA fine-tuning of a causal LM on instruction -> Verilog pairs, identical for every architecture.

Comparability rules built in:
  * batches are formed from character lengths with a seed, so the example order and the
    optimizer steps are the same for every model and every micro-batch size;
  * right padding and loss on response tokens only, which is neutral for Transformer, SSM and RNN models;
  * every checkpoint records tokens seen and pure training GPU-seconds, so results can be compared
    both at equal tokens and at equal GPU time;
  * the LoRA rank can be derived from a target number of trainable parameters.

Usage:
  python scripts/train_lora.py --config configs/train/pythia-1.4b.json --seed 0 --output runs/pythia-1.4b/s0
  python scripts/train_lora.py --config ... --output runs/... --resume          # continue an interrupted run
  python scripts/train_lora.py --config ... --set lr=1e-4 --set train_fraction=0.2   # override config keys

A non-finite loss stops the run and writes diverged.json in the output directory.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Linear layers adapted by default; RWKV names match both attention and feed-forward blocks
DEFAULT_TARGETS = {
    "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    # PEFT rejects out_proj and conv1d for Mamba models
    "mamba": ["in_proj", "x_proj", "dt_proj"],
    "mamba2": ["in_proj"],
    "rwkv": ["key", "value", "receptance", "output"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}
SUBSET_SEED = 1234  # fixed so that hyperparameter search subsets are the same for every architecture


def parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def load_config(path, overrides):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for item in overrides:
        key, _, value = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = parse_value(value)
    return cfg


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(record, tokenizer, template, max_len):
    prompt = template.format(instruction=record["Instruction"].strip())
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    r_ids = tokenizer(record["Response"].strip() + "\n", add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    ids = p_ids + r_ids
    labels = [-100] * len(p_ids) + r_ids
    return {"input_ids": ids[:max_len], "labels": labels[:max_len], "truncated": len(ids) > max_len}


def make_batches(char_lengths, batch_size, seed, epoch, group=50):
    """Shuffle, sort by length inside mega-batches of `group` batches, then shuffle the batches."""
    rng = random.Random(seed * 100003 + epoch)
    order = list(range(len(char_lengths)))
    rng.shuffle(order)
    batches = []
    mega = batch_size * group
    for start in range(0, len(order), mega):
        chunk = sorted(order[start:start + mega], key=lambda i: char_lengths[i], reverse=True)
        batches += [chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size)]
    rng.shuffle(batches)
    return batches


def collate(examples, pad_id, device):
    width = max(len(e["input_ids"]) for e in examples)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), width), -100, dtype=torch.long)
    mask = torch.zeros((len(examples), width), dtype=torch.long)
    for row, e in enumerate(examples):
        n = len(e["input_ids"])
        ids[row, :n] = torch.tensor(e["input_ids"])
        labels[row, :n] = torch.tensor(e["labels"])
        mask[row, :n] = 1
    return ids.to(device), labels.to(device), mask.to(device)


def token_loss_sum(model, ids, labels, mask, pass_mask, amp_dtype):
    kwargs = {"attention_mask": mask} if pass_mask else {}
    with torch.autocast(ids.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        logits = model(input_ids=ids, **kwargs).logits
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    return F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
                           ignore_index=-100, reduction="sum")


def lora_params_per_rank(model, targets):
    """Trainable LoRA parameters per unit of rank: sum of (in + out) over adapted linear layers."""
    total = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in targets:
            total += module.in_features + module.out_features
    return total


class GpuMonitor(threading.Thread):
    """Samples NVML counters to detect foreign load on a shared GPU; silently disabled without NVML."""

    def __init__(self, path, interval=2.0):
        super().__init__(daemon=True)
        self.path, self.interval, self.stop_event = path, interval, threading.Event()
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml, self.handle = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.nvml = None

    def info(self):
        if not self.nvml:
            return {}
        n = self.nvml
        return {"gpu_name": n.nvmlDeviceGetName(self.handle), "driver": n.nvmlSystemGetDriverVersion()}

    def run(self):
        if not self.nvml:
            return
        n = self.nvml
        with open(self.path, "a", encoding="utf-8") as f:
            if f.tell() == 0:
                f.write("time,util_gpu,util_mem,mem_used_mb,mem_total_mb,sm_clock_mhz,power_w,temp_c\n")
            while not self.stop_event.is_set():
                try:
                    u = n.nvmlDeviceGetUtilizationRates(self.handle)
                    m = n.nvmlDeviceGetMemoryInfo(self.handle)
                    f.write(f"{time.time():.1f},{u.gpu},{u.memory},{m.used >> 20},{m.total >> 20},"
                            f"{n.nvmlDeviceGetClockInfo(self.handle, n.NVML_CLOCK_SM)},"
                            f"{n.nvmlDeviceGetPowerUsage(self.handle) / 1000:.1f},"
                            f"{n.nvmlDeviceGetTemperature(self.handle, n.NVML_TEMPERATURE_GPU)}\n")
                    f.flush()
                except Exception:
                    pass
                self.stop_event.wait(self.interval)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip() or None
    except OSError:
        return None


@torch.no_grad()
def evaluate(model, examples, cfg, pad_id, device, amp_dtype):
    model.eval()
    loss_sum, tokens = 0.0, 0
    order = sorted(range(len(examples)), key=lambda i: len(examples[i]["input_ids"]))
    for start in range(0, len(order), cfg["micro_batch_size"]):
        batch = [examples[i] for i in order[start:start + cfg["micro_batch_size"]]]
        ids, labels, mask = collate(batch, pad_id, device)
        loss_sum += token_loss_sum(model, ids, labels, mask, cfg["pass_attention_mask"], amp_dtype).item()
        tokens += int((labels[:, 1:] != -100).sum())
    model.train()
    return loss_sum / max(1, tokens)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-at-step", type=int, help="save a checkpoint and exit after this step; continue later with --resume")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config key, e.g. lora.r=8")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    if args.seed is not None:
        cfg["seed"] = args.seed
    out = Path(args.output)
    state_path = out / "trainer_state.pt"
    if out.exists() and any(out.iterdir()) and not args.resume:
        sys.exit(f"{out} is not empty; use --resume or another --output")
    if args.resume and not state_path.exists():
        sys.exit(f"--resume: {state_path} not found")
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)

    seed = cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and cfg.get("bf16", True) else None

    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg.get("revision"))
    if tokenizer.eos_token_id is None:
        sys.exit("tokenizer has no EOS token")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # --- data -------------------------------------------------------------------------------
    train_records = load_jsonl(cfg["train_file"])
    if cfg.get("train_fraction", 1.0) < 1.0:
        keep = random.Random(SUBSET_SEED).sample(range(len(train_records)), int(len(train_records) * cfg["train_fraction"]))
        train_records = [train_records[i] for i in sorted(keep)]
    val_records = load_jsonl(cfg["val_file"])
    template, max_len = cfg["prompt_template"], cfg["max_len"]
    train = [encode(r, tokenizer, template, max_len) for r in train_records]
    val = [encode(r, tokenizer, template, max_len) for r in val_records]
    char_lengths = [len(r["Instruction"]) + len(r["Response"]) for r in train_records]
    gbs, mbs = cfg["global_batch_size"], cfg["micro_batch_size"]
    steps_per_epoch = math.ceil(len(train) / gbs)
    total_steps = cfg.get("max_steps") or steps_per_epoch * cfg["epochs"]

    # --- model ------------------------------------------------------------------------------
    load_dtype = torch.bfloat16 if amp_dtype is not None else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg.get("revision"), dtype=load_dtype,
                                                 **cfg.get("model_kwargs", {}))
    model.config.use_cache = False
    model_type = model.config.model_type
    lora_cfg = cfg["lora"]
    targets = lora_cfg.get("target_modules") or DEFAULT_TARGETS.get(model_type)
    if not targets:
        sys.exit(f"no default LoRA targets for model_type={model_type}; set lora.target_modules")
    per_rank = lora_params_per_rank(model, set(targets))
    if per_rank == 0:
        sys.exit(f"LoRA targets {targets} match no nn.Linear layers in {model_type}")
    rank = lora_cfg.get("r")
    if lora_cfg.get("target_trainable_params"):
        rank = max(1, round(lora_cfg["target_trainable_params"] / per_rank))
    total_params = sum(p.numel() for p in model.parameters())
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=rank * lora_cfg.get("alpha_ratio", 2), lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_params = sum(p.numel() for p in trainable)

    optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0),
                                  betas=tuple(cfg.get("betas", (0.9, 0.999))))
    warmup = max(1, int(total_steps * cfg.get("warmup_ratio", 0.03)))
    min_ratio = cfg.get("min_lr_ratio", 0.1)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- state --------------------------------------------------------------------------------
    monitor = GpuMonitor(out / "gpu_monitor.csv")
    # Epoch slices are defined in optimizer steps, so every architecture is checkpointed after the same examples
    slice_steps, fraction = [], cfg.get("save_every_epoch_fraction")
    if fraction:
        k = 1
        while round(k * fraction * steps_per_epoch) <= total_steps:
            slice_steps.append(round(k * fraction * steps_per_epoch))
            k += 1
    slice_steps = sorted(set(s for s in slice_steps if s > 0))
    st = {"step": 0, "tokens": 0, "supervised_tokens": 0, "gpu_seconds": 0.0, "wall_seconds": 0.0,
          "next_token_ckpt": cfg.get("save_every_tokens"), "next_time_ckpt": cfg.get("save_every_gpu_seconds")}
    if args.resume:
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        st = saved["state"]
        random.setstate(saved["rng_python"])
        torch.set_rng_state(saved["rng_torch"])
        if device.type == "cuda" and saved.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(saved["rng_cuda"])
        print(f"resumed at step {st['step']}")
    else:
        manifest = {
            "config": cfg, "config_path": str(args.config), "command": sys.argv,
            "model_type": model_type, "lora_targets": targets, "lora_rank": rank, "lora_params_per_rank": per_rank,
            "total_params": total_params, "trainable_params": trainable_params,
            "train_examples": len(train), "val_examples": len(val),
            "train_truncated": sum(e["truncated"] for e in train), "val_truncated": sum(e["truncated"] for e in val),
            "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "warmup_steps": warmup,
            "epoch_slice_steps": slice_steps,
            "train_file_sha256": file_sha256(cfg["train_file"]), "val_file_sha256": file_sha256(cfg["val_file"]),
            "git_commit": git_commit(), "host": platform.node(), "python": sys.version.split()[0],
            "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
            "nvml": monitor.info(),
            "versions": {m: __import__(m).__version__ for m in ("transformers", "peft")},
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{model_type}: params {total_params / 1e6:.1f}M, LoRA r={rank} trainable {trainable_params / 1e6:.2f}M, "
          f"{len(train)} train examples, {total_steps} steps")

    if device.type == "cuda":
        monitor.start()
        torch.cuda.reset_peak_memory_stats()
    metrics = open(out / "metrics.jsonl", "a", encoding="utf-8")
    index = open(out / "checkpoints" / "index.jsonl", "a", encoding="utf-8")

    def save_checkpoint(reason):
        name = f"step{st['step']:06d}"
        model.save_pretrained(out / "checkpoints" / name)
        val_loss = evaluate(model, val, cfg, pad_id, device, amp_dtype)
        row = {"checkpoint": name, "reason": reason, "epoch": round(st["step"] / steps_per_epoch, 4),
               "val_loss": round(val_loss, 5),
               **{k: st[k] for k in ("step", "tokens", "supervised_tokens", "gpu_seconds", "wall_seconds")}}
        index.write(json.dumps(row) + "\n")
        index.flush()
        torch.save({"adapter": get_peft_model_state_dict(model), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "state": st, "rng_python": random.getstate(),
                    "rng_torch": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None}, state_path)
        print(f"checkpoint {name} ({reason}): val_loss {val_loss:.4f}")

    model.train()
    wall_start = time.perf_counter() - st["wall_seconds"]
    log_every = cfg.get("log_every_steps", 10)
    epoch_batches = {}
    while st["step"] < total_steps:
        epoch, pos = divmod(st["step"], steps_per_epoch)
        if epoch not in epoch_batches:
            epoch_batches = {epoch: make_batches(char_lengths, gbs, seed, epoch)}
        batch = [train[i] for i in epoch_batches[epoch][pos]]
        batch_tokens = sum(len(e["input_ids"]) for e in batch)
        supervised = sum(sum(1 for t in e["labels"][1:] if t != -100) for e in batch)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss_total = 0.0
        for start in range(0, len(batch), mbs):
            ids, labels, mask = collate(batch[start:start + mbs], pad_id, device)
            loss = token_loss_sum(model, ids, labels, mask, cfg["pass_attention_mask"], amp_dtype) / supervised
            loss.backward()
            loss_total += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.get("grad_clip", 1.0)).item()
        if not math.isfinite(loss_total):
            (out / "diverged.json").write_text(json.dumps({"step": st["step"] + 1, "loss": str(loss_total)}), encoding="utf-8")
            sys.exit(f"non-finite loss at step {st['step'] + 1}; run marked as diverged")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_seconds = time.perf_counter() - t0

        st["step"] += 1
        st["tokens"] += batch_tokens
        st["supervised_tokens"] += supervised
        st["gpu_seconds"] += step_seconds
        st["wall_seconds"] = time.perf_counter() - wall_start
        if st["step"] % log_every == 0 or st["step"] == total_steps:
            row = {"step": st["step"], "epoch": round(st["step"] / steps_per_epoch, 4), "loss": round(loss_total, 5),
                   "lr": scheduler.get_last_lr()[0], "grad_norm": round(grad_norm, 4),
                   "step_seconds": round(step_seconds, 3), "tokens_per_second": round(batch_tokens / step_seconds, 1),
                   **{k: st[k] for k in ("tokens", "supervised_tokens", "gpu_seconds", "wall_seconds")}}
            if device.type == "cuda":
                row["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            print(json.dumps(row))

        reasons = []
        if st["step"] in slice_steps:
            reasons.append("epoch_slice")
        if st["next_token_ckpt"] and st["tokens"] >= st["next_token_ckpt"]:
            reasons.append("tokens")
            while st["next_token_ckpt"] <= st["tokens"]:
                st["next_token_ckpt"] += cfg["save_every_tokens"]
        if st["next_time_ckpt"] and st["gpu_seconds"] >= st["next_time_ckpt"]:
            reasons.append("gpu_time")
            while st["next_time_ckpt"] <= st["gpu_seconds"]:
                st["next_time_ckpt"] += cfg["save_every_gpu_seconds"]
        if st["step"] == total_steps:
            reasons.append("final")
        elif st["step"] == args.stop_at_step:
            reasons.append("stop")
        if reasons:
            save_checkpoint("+".join(reasons))
        if st["step"] == args.stop_at_step and st["step"] < total_steps:
            print(f"stopped at step {st['step']}; continue with --resume")
            break

    monitor.stop_event.set()
    if st["step"] < total_steps:
        metrics.close()
        index.close()
        return
    summary = {"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **st}
    if device.type == "cuda":
        summary["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    metrics.close()
    index.close()


if __name__ == "__main__":
    main()
