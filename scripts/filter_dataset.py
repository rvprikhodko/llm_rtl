"""Filtering of an instruction -> Verilog dataset (RTLCoder Resyn27k format) and a fixed train/val split.

Quality filters run first, then near-duplicate removal among the survivors, so a cluster
is always represented by a record that passed every other check.

Usage:
  python scripts/filter_dataset.py dataset/Resyn27k.json --out dataset/filtered \
      --rtllm vai-experiments-rtllm/datasets/trusted/rtllm-882030d9e61b7787/cases [--iverilog]

Outputs in --out:
  train.jsonl, val.jsonl   kept records: {"id", "Instruction", "Response"}; id is the source line index
  rejected.jsonl           {"id", "reasons", ...} for every dropped record
  filter_report.json       parameters, counts per reason, split sizes, input checksum
"""
import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_dataset import (MODULE_RE, NAMED_MODULE_RE, STDCELL_RE, SV_RE, duplicate_clusters, iverilog_check, load,
                             normalize_code, response_text, shingles)

# Plain Verilog identifier; excludes escaped names such as \$mul taken from Yosys cell libraries
PLAIN_MODULE_RE = re.compile(r"\bmodule\s+[A-Za-z_][A-Za-z0-9_$]*\s*(#|\(|;)")


def rtllm_matches(records, cases_dir, desc_threshold, code_threshold):
    """Map record index -> list of RTLLM cases its instruction or code resembles."""
    instr_sh = [shingles(r["Instruction"], 3) for r in records]
    code_sh = [shingles(normalize_code(response_text(r)), 5) for r in records]
    hits = collections.defaultdict(list)
    for case in sorted(Path(cases_dir).iterdir()):
        meta = json.loads((case / "case.json").read_text(encoding="utf-8"))
        desc = shingles((case / "source/design_description.txt").read_text(encoding="utf-8", errors="replace"), 3)
        oracle = shingles(normalize_code("\n".join(
            p.read_text(encoding="utf-8", errors="replace") for p in (case / "oracle/project/rtl").glob("*.v"))), 5)
        for i in range(len(records)):
            d = len(desc & instr_sh[i]) / max(1, len(desc | instr_sh[i]))
            c = len(oracle & code_sh[i]) / max(1, len(oracle | code_sh[i]))
            if d >= desc_threshold or c >= code_threshold:
                hits[i].append({"case": meta["id"], "desc_jaccard": round(d, 3), "code_jaccard": round(c, 3)})
    return hits


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("--out", default="dataset/filtered")
    p.add_argument("--rtllm", help="trusted RTLLM cases directory for decontamination")
    p.add_argument("--tokenizer", default="EleutherAI/pythia-1.4b")
    p.add_argument("--max-tokens", type=int, default=2048, help="instruction + response, without prompt template")
    p.add_argument("--dedup-threshold", type=float, default=0.85, help="estimated Jaccard of response 5-grams")
    p.add_argument("--rtllm-desc-threshold", type=float, default=0.5)
    p.add_argument("--rtllm-code-threshold", type=float, default=0.6)
    p.add_argument("--drop-systemverilog", action="store_true")
    p.add_argument("--iverilog", action="store_true", help="drop references that fail iverilog -g2012")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--val-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records = load(args.dataset)
    codes = [response_text(r) for r in records]
    instrs = [r["Instruction"] for r in records]
    reasons = collections.defaultdict(list)
    details = collections.defaultdict(dict)

    for i, (instr, code) in enumerate(zip(instrs, codes)):
        if not isinstance(records[i]["Response"], (list, str)) or not code.strip():
            reasons[i].append("empty_response")
            continue
        if not PLAIN_MODULE_RE.search(code) or "endmodule" not in code:
            reasons[i].append("no_plain_module")
        if STDCELL_RE.search(instr) or STDCELL_RE.search(code):
            reasons[i].append("stdcell")
        named = set(NAMED_MODULE_RE.findall(instr))
        if named and not named & set(MODULE_RE.findall(code)):
            reasons[i].append("named_module_missing")
            details[i]["named"] = sorted(named)
        if args.drop_systemverilog and SV_RE.search(code):
            reasons[i].append("systemverilog")

    from tokenizers import Tokenizer
    tok = Tokenizer.from_pretrained(args.tokenizer)
    lengths = [len(a.ids) + len(b.ids) for a, b in zip(tok.encode_batch(instrs), tok.encode_batch(codes))]
    for i, n in enumerate(lengths):
        if n > args.max_tokens:
            reasons[i].append("too_long")
            details[i]["tokens"] = n

    iverilog_status = "not_run"
    if args.iverilog:
        checked = iverilog_check(codes, args.workers or 8)
        if checked is None:
            sys.exit("iverilog not found: install it or run without --iverilog")
        for i, ok, err in checked:
            if not ok:
                reasons[i].append("iverilog_fail")
                details[i]["stderr"] = err
        iverilog_status = "checked"

    if args.rtllm:
        for i, hits in rtllm_matches(records, args.rtllm, args.rtllm_desc_threshold, args.rtllm_code_threshold).items():
            reasons[i].append("rtllm_overlap")
            details[i]["rtllm"] = hits

    # exact instruction duplicates, then near-duplicate responses, both among quality survivors
    seen_instr = {}
    for i in range(len(records)):
        if reasons[i]:
            continue
        key = instrs[i].strip()
        if key in seen_instr:
            reasons[i].append("duplicate_instruction")
            details[i]["kept"] = seen_instr[key]
        else:
            seen_instr[key] = i
    survivors = [i for i in range(len(records)) if not reasons[i]]
    roots, _ = duplicate_clusters([normalize_code(codes[i]) for i in survivors], threshold=args.dedup_threshold)
    for pos, root in enumerate(roots):
        if root != pos:
            reasons[survivors[pos]].append("near_duplicate")
            details[survivors[pos]]["kept"] = survivors[root]

    kept = [i for i in range(len(records)) if not reasons[i]]
    rng = random.Random(args.seed)
    val = set(rng.sample(kept, min(args.val_size, len(kept))))

    def dump(path, ids):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for i in ids:
                f.write(json.dumps({"id": i, "Instruction": instrs[i], "Response": codes[i]}, ensure_ascii=False) + "\n")

    dump(out / "train.jsonl", [i for i in kept if i not in val])
    dump(out / "val.jsonl", sorted(val))
    with open(out / "rejected.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i in sorted(reasons):
            if reasons[i]:
                f.write(json.dumps({"id": i, "reasons": reasons[i], **details[i]}, ensure_ascii=False) + "\n")

    any_reason = collections.Counter(r for rs in reasons.values() for r in rs)
    only_reason = collections.Counter(rs[0] for rs in reasons.values() if len(rs) == 1)
    kept_tokens = sum(lengths[i] for i in kept)
    report = {
        "input": str(args.dataset),
        "input_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        "parameters": {k: v for k, v in vars(args).items() if k not in ("dataset", "out", "workers")},
        "iverilog": iverilog_status,
        "records_in": len(records),
        "records_kept": len(kept),
        "records_dropped": len(records) - len(kept),
        "dropped_by_reason_any": dict(any_reason.most_common()),
        "dropped_by_single_reason": dict(only_reason.most_common()),
        "split": {"train": len(kept) - len(val), "val": len(val), "seed": args.seed},
        "kept_tokens_total": kept_tokens,
    }
    (out / "filter_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
