"""Analysis of an instruction -> Verilog dataset (RTLCoder Resyn27k format).

Reports structure, lengths in Pythia tokens, duplicates, interface consistency,
optional iverilog syntax check and overlap with the corrected RTLLM corpus.

Usage:
  python scripts/analyze_dataset.py dataset/Resyn27k.json --out dataset/analysis \
      --rtllm vai-experiments-rtllm/datasets/trusted/rtllm-882030d9e61b7787/cases [--iverilog]
"""
import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

MODULE_RE = re.compile(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)")
NAMED_MODULE_RE = re.compile(r"\bmodule\s+(?:named|called)?\s*`([A-Za-z_][A-Za-z0-9_$]*)`", re.I)
COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
TRUTH_TABLE_RE = re.compile(r"^\s*\|(\s*[01xX]\s*\|){3,}\s*$", re.M)
SV_RE = re.compile(r"\b(logic|always_ff|always_comb|always_latch|typedef|interface|enum|unique case|priority case)\b")
STDCELL_RE = re.compile(r"sky130_|gf180|_fd_sc_|\bsg13g2_", re.I)
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")


def load(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    return data


def response_text(record):
    resp = record["Response"]
    return resp[0] if isinstance(resp, list) else resp


def normalize_code(code):
    return re.sub(r"\s+", " ", COMMENT_RE.sub("", code)).strip()


def percentiles(values):
    a = np.asarray(values)
    return {p: int(np.percentile(a, p)) for p in (50, 75, 90, 95, 99)} | {"max": int(a.max()), "mean": round(float(a.mean()), 1)}


def shingles(text, n):
    words = WORD_RE.findall(text.lower())
    return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


PRIME = (1 << 31) - 1


def minhash(sh, a, b):
    # 31-bit shingle hashes keep (a * x + b) inside uint64 for every permutation
    x = np.fromiter((int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "little") % PRIME for s in sh),
                    dtype=np.uint64, count=len(sh))
    return ((a[:, None] * x[None, :] + b[:, None]) % PRIME).min(axis=1)


def duplicate_clusters(codes, num_perm=64, bands=16, threshold=0.85):
    """Return (root index per record, confirmed pair count); records sharing a root are near-duplicates."""
    rng = np.random.default_rng(0)
    a = rng.integers(1, PRIME, num_perm).astype(np.uint64)
    b = rng.integers(0, PRIME, num_perm).astype(np.uint64)
    sigs = [minhash(shingles(c, 5), a, b) for c in codes]
    rows = num_perm // bands
    buckets = collections.defaultdict(list)
    for idx, sig in enumerate(sigs):
        for band in range(bands):
            buckets[(band, tuple(sig[band * rows:(band + 1) * rows]))].append(idx)
    pairs = set()
    for members in buckets.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add((members[i], members[j]))
    # union-find over pairs whose estimated Jaccard passes the threshold
    parent = list(range(len(codes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    confirmed = 0
    for i, j in sorted(pairs):
        if np.mean(sigs[i] == sigs[j]) >= threshold:
            confirmed += 1
            ri, rj = find(i), find(j)
            # the smallest index becomes the root, so the earliest record is the one kept
            parent[max(ri, rj)] = min(ri, rj)
    return [find(i) for i in range(len(codes))], confirmed


def near_duplicates(codes, **kwargs):
    roots, confirmed = duplicate_clusters(codes, **kwargs)
    clusters = collections.Counter(roots)
    redundant = sum(size - 1 for size in clusters.values() if size > 1)
    return {"pairs": confirmed, "clusters": sum(1 for s in clusters.values() if s > 1), "redundant_records": redundant}


def iverilog_check(codes, workers):
    exe = shutil.which("iverilog")
    if not exe:
        return None

    def one(item):
        idx, code = item
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "dut.sv"
            src.write_text(code, encoding="utf-8")
            try:
                proc = subprocess.run([exe, "-g2012", "-o", os.devnull, str(src)], capture_output=True, text=True, timeout=30)
                return idx, proc.returncode == 0, proc.stderr[-300:]
            except subprocess.TimeoutExpired:
                return idx, False, "timeout"

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, enumerate(codes)))


def rtllm_overlap(records, cases_dir, top=3):
    instr_sh = [shingles(r["Instruction"], 3) for r in records]
    code_sh = [shingles(normalize_code(response_text(r)), 5) for r in records]
    code_modules = [set(MODULE_RE.findall(response_text(r))) for r in records]
    result = []
    for case in sorted(Path(cases_dir).iterdir()):
        meta = json.loads((case / "case.json").read_text(encoding="utf-8"))
        desc = (case / "source/design_description.txt").read_text(encoding="utf-8", errors="replace")
        oracle = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in (case / "oracle/project/rtl").glob("*.v"))
        d_sh, o_sh = shingles(desc, 3), shingles(normalize_code(oracle), 5)

        def jac(x, y):
            return len(x & y) / max(1, len(x | y))

        by_desc = sorted(((jac(d_sh, s), i) for i, s in enumerate(instr_sh)), reverse=True)[:top]
        by_code = sorted(((jac(o_sh, s), i) for i, s in enumerate(code_sh)), reverse=True)[:top]
        same_name = [i for i, mods in enumerate(code_modules) if meta["target_module"] in mods]
        result.append({
            "case": meta["id"], "module": meta["target_module"],
            "max_desc_jaccard": round(by_desc[0][0], 3), "max_code_jaccard": round(by_code[0][0], 3),
            "top_desc": [(round(s, 3), i) for s, i in by_desc], "top_code": [(round(s, 3), i) for s, i in by_code],
            "same_module_name_records": len(same_name),
        })
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("--out", default="dataset/analysis")
    p.add_argument("--rtllm", help="path to trusted RTLLM cases directory")
    p.add_argument("--tokenizer", default="EleutherAI/pythia-1.4b")
    p.add_argument("--iverilog", action="store_true", help="run iverilog -g2012 on every reference")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records = load(args.dataset)
    codes = [response_text(r) for r in records]
    instrs = [r["Instruction"] for r in records]
    report = {"records": len(records), "fields": sorted({k for r in records for k in r})}
    report["response_list_lengths"] = dict(collections.Counter(len(r["Response"]) if isinstance(r["Response"], list) else 1 for r in records))

    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_pretrained(args.tokenizer)
        n_instr = [len(e.ids) for e in tok.encode_batch(instrs)]
        n_code = [len(e.ids) for e in tok.encode_batch(codes)]
        total = [a + b for a, b in zip(n_instr, n_code)]
        report["tokens"] = {
            "tokenizer": args.tokenizer, "instruction": percentiles(n_instr), "response": percentiles(n_code), "total": percentiles(total),
            "sum_total": int(sum(total)),
            "fits": {str(L): round(float(np.mean(np.asarray(total) <= L)), 4) for L in (512, 1024, 1536, 2048, 4096)},
        }
    except Exception as exc:  # tokenizer download can fail offline
        report["tokens"] = {"error": str(exc)}
    report["chars"] = {"instruction": percentiles([len(s) for s in instrs]), "response": percentiles([len(s) for s in codes])}

    modules = [MODULE_RE.findall(c) for c in codes]
    named = [NAMED_MODULE_RE.findall(s) for s in instrs]
    report["structure"] = {
        "modules_per_response": dict(sorted(collections.Counter(min(len(m), 5) for m in modules).items())),
        "no_endmodule": sum("endmodule" not in c for c in codes),
        "systemverilog_constructs": sum(bool(SV_RE.search(c)) for c in codes),
        "stdcell_related": sum(bool(STDCELL_RE.search(i) or STDCELL_RE.search(c)) for i, c in zip(instrs, codes)),
        "truth_table_in_instruction": sum(bool(TRUTH_TABLE_RE.search(s)) for s in instrs),
        "instruction_names_module": sum(bool(n) for n in named),
        "named_module_missing_in_response": sum(bool(n) and not set(n) & set(m) for n, m in zip(named, modules)),
    }

    norm = [normalize_code(c) for c in codes]
    report["duplicates"] = {
        "exact_instruction": len(instrs) - len(set(s.strip() for s in instrs)),
        "exact_response_normalized": len(norm) - len(set(norm)),
        "exact_pair": len(records) - len({(s.strip(), c) for s, c in zip(instrs, norm)}),
        "near_response_minhash_0.85": near_duplicates(norm),
    }

    if args.iverilog:
        checked = iverilog_check(codes, args.workers)
        if checked is None:
            report["iverilog"] = "iverilog not found"
        else:
            fails = [(i, err) for i, ok, err in checked if not ok]
            report["iverilog"] = {"checked": len(checked), "failed": len(fails), "fail_rate": round(len(fails) / len(checked), 4)}
            (out / "iverilog_failures.jsonl").write_text("\n".join(json.dumps({"idx": i, "stderr": e}) for i, e in fails), encoding="utf-8")

    if args.rtllm:
        overlap = rtllm_overlap(records, args.rtllm)
        report["rtllm_overlap_summary"] = {
            "cases": len(overlap),
            "desc_jaccard_ge_0.5": sum(o["max_desc_jaccard"] >= 0.5 for o in overlap),
            "code_jaccard_ge_0.5": sum(o["max_code_jaccard"] >= 0.5 for o in overlap),
            "cases_with_same_module_name": sum(o["same_module_name_records"] > 0 for o in overlap),
        }
        (out / "rtllm_overlap.json").write_text(json.dumps(overlap, indent=1, ensure_ascii=False), encoding="utf-8")

    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
