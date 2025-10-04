# -*- coding: utf-8 -*-
# eval_nmr.py
"""
Evaluate 13C (and optional 1H) NMR predictions in JSONL files.

Each JSONL record is expected to have:
- "input" (or "smiles")
- "gt_answer"
- "llm_answer"

It will:
1) Parse <13C_NMR>...</13C_NMR> tags and extract carbon chemical shifts.
2) (Optional) Parse <1H_NMR>...</1H_NMR> tags and extract proton peaks:
   shift (or range centroid), multiplicity, nH.
3) Compute metrics:
   - 13C: greedy nearest-neighbor matching within tolerance (ppm)
           -> precision, recall, F1, MAE(|Δppm|)
   - 1H: weighted Jaccard-style similarity considering nH and ppm distance
         + simple peak recall/precision (counts)
4) Save per-sample details and per-model summaries as CSV.

Usage:
    python eval_nmr.py --root "E:\\spectrumllm_bench\\spectrum\\smiles2spec\\code\\results" --out "E:\\nmr_eval_out" --tolC 0.5 --tolH 0.12 --sigmaH 0.06

Dependencies:
    pip install pandas
"""

import argparse
import json
import os
import re
import math
from typing import List, Tuple, Dict, Optional
import pandas as pd


# -----------------------------
# Utilities: file loading
# -----------------------------
def iter_jsonl_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".jsonl"):
                yield os.path.join(dirpath, fn)


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # be tolerant to format issues
                continue


# -----------------------------
# 13C parsing & scoring
# -----------------------------
_13C_TAG = re.compile(
    r"<(?:13C_NMR|cnmr)>(.*?)</(?:13C_NMR|cnmr)>", re.IGNORECASE | re.DOTALL)


def extract_13c_tag(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = _13C_TAG.search(text)
    return m.group(1).strip() if m else None


def parse_13c_shifts(tag_content: Optional[str]) -> List[float]:
    """Parse numbers after 'δ' inside the 13C tag."""
    if not tag_content:
        return []
    s = tag_content
    m = re.search(r"δ(.*)$", s, flags=re.DOTALL)
    s = m.group(1) if m else s
    vals = []
    for token in re.split(r"[,\s]+", s.strip()):
        try:
            vals.append(float(token))
        except Exception:
            pass
    return vals


def greedy_match_within_tol(gt: List[float], pred: List[float], tol: float = 0.5):
    gt_sorted = sorted(gt)
    pred_sorted = sorted(pred)
    used = [False]*len(gt_sorted)
    matches = []
    for x in pred_sorted:
        best_j, best_d = None, None
        for j, g in enumerate(gt_sorted):
            if used[j]:
                continue
            d = abs(x - g)
            if d <= tol and (best_d is None or d < best_d):
                best_j, best_d = j, d
        if best_j is not None:
            used[best_j] = True
            matches.append((x, gt_sorted[best_j], best_d))
    return matches


def score_13c(gt: List[float], pred: List[float], tol: float = 0.5):
    """Return dict with n_gt, n_pred, n_match, recall, precision, f1, mae."""
    if not gt and not pred:
        return dict(n_gt=0, n_pred=0, n_match=0, recall=1.0, precision=1.0, f1=1.0, mae=None)
    matches = greedy_match_within_tol(gt, pred, tol=tol)
    n_gt, n_pred, n_match = len(gt), len(pred), len(matches)
    recall = n_match / n_gt if n_gt else 0.0
    precision = n_match / n_pred if n_pred else 0.0
    f1 = 2*precision*recall / \
        (precision+recall) if (precision+recall) > 0 else 0.0
    mae = sum(d for _, _, d in matches)/n_match if n_match > 0 else None
    return dict(n_gt=n_gt, n_pred=n_pred, n_match=n_match, recall=recall, precision=precision, f1=f1, mae=mae)


# -----------------------------
# 1H parsing & scoring (optional)
# -----------------------------
_H_TAG = re.compile(r"<1H_NMR>(.*?)</1H_NMR>", re.IGNORECASE | re.DOTALL)


def extract_1h_tag(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = _H_TAG.search(text)
    return m.group(1).strip() if m else None


_PEAK_PATTERN = re.compile(
    r"\s*([0-9]*\.?[0-9]+)(?:\s*-\s*([0-9]*\.?[0-9]+))?\s*\(([^)]*?)\)"
)


def parse_1h_peaks(tag_content: Optional[str]) -> List[Dict]:
    """
    Parse peaks like:
      δ 4.11 (q, J= 6.59 Hz, 2H), 2.50-2.70 (m, 12H), ...
    Returns: list of dict with keys: shift, range_min, range_max, nH, mult, raw
    """
    if not tag_content:
        return []
    s = tag_content
    m = re.search(r"δ(.*)$", s, flags=re.DOTALL)
    s = m.group(1) if m else s
    peaks = []
    for m in _PEAK_PATTERN.finditer(s):
        a = float(m.group(1))
        b_str = m.group(2)
        b = float(b_str) if b_str else None
        info = m.group(3)
        # nH = last integer before 'H'
        nHs = re.findall(r"(\d+)\s*H", info, flags=re.IGNORECASE)
        nH = int(nHs[-1]) if nHs else 1
        mult = None
        m2 = re.match(r"\s*([a-zA-Z]+)", info)
        if m2:
            mult = m2.group(1).lower()
        shift = (a + b)/2.0 if b is not None else a
        peaks.append({
            "shift": shift,
            "range_min": a,
            "range_max": b if b is not None else a,
            "nH": nH,
            "mult": mult,
            "raw": m.group(0).strip()
        })
    return peaks


def match_and_score_1h(gt_peaks: List[Dict], pred_peaks: List[Dict], tol: float = 0.12, sigma: float = 0.06):
    """
    Weighted Jaccard-like similarity based on nH and ppm distance.
    - Match greedily within tol; weight = min(nH_pred, nH_gt) * exp(-0.5*(d/sigma)^2)
    - score = matched_w / (sum_pred_nH + sum_gt_nH - matched_w)
    Also returns a simple count-based precision/recall by peaks (ignore nH).
    """
    gt_used = set()
    matched_w = 0.0
    n_match = 0
    for p in pred_peaks:
        best = None
        best_score = -1.0
        for j, g in enumerate(gt_peaks):
            if j in gt_used:
                continue
            d = abs(p["shift"] - g["shift"])
            if d <= tol:
                w = min(p["nH"], g["nH"]) * math.exp(-0.5 * (d/sigma)**2)
                if w > best_score:
                    best_score = w
                    best = j
        if best is not None:
            gt_used.add(best)
            matched_w += best_score
            n_match += 1
    sum_pred = sum(p["nH"] for p in pred_peaks)
    sum_gt = sum(g["nH"] for g in gt_peaks)
    score = matched_w / (sum_pred + sum_gt - matched_w + 1e-12)

    # simple counts (by peaks)
    n_gt = len(gt_peaks)
    n_pred = len(pred_peaks)
    prec = n_match / n_pred if n_pred else 0.0
    rec = n_match / n_gt if n_gt else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0

    return dict(
        hnmr_score=score,
        hnmr_n_gt=n_gt,
        hnmr_n_pred=n_pred,
        hnmr_n_match=n_match,
        hnmr_precision=prec,
        hnmr_recall=rec,
        hnmr_f1=f1
    )


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=False,
                    help="Root directory containing JSONL result files")
    ap.add_argument("--file", required=False,
                    help="Evaluate only this JSONL file (overrides --root)")
    ap.add_argument("--out", required=True, help="Output directory for CSVs")
    ap.add_argument("--tolC", type=float, default=0.5,
                    help="13C matching tolerance in ppm (default 0.5)")
    ap.add_argument("--tolH", type=float, default=0.12,
                    help="1H matching tolerance in ppm (default 0.12)")
    ap.add_argument("--sigmaH", type=float, default=0.06,
                    help="1H Gaussian sigma in ppm for weighting (default 0.06)")
    ap.add_argument("--evalH", action="store_true",
                    help="Also evaluate 1H NMR if present")
    args = ap.parse_args()

    if not args.file and not args.root:
        raise SystemExit("Must provide either --file or --root")
    if args.file and not os.path.isfile(args.file):
        raise SystemExit(f"File not found: {args.file}")
    if args.root and not os.path.isdir(args.root):
        print(f"[WARN] Root directory not found: {args.root}")

    os.makedirs(args.out, exist_ok=True)

    # Collect files
    if args.file:
        files_to_eval = [args.file]
    else:
        files_to_eval = list(iter_jsonl_files(args.root))
    if not files_to_eval:
        raise SystemExit("No JSONL files to evaluate.")

    per_sample = []
    for path in files_to_eval:
        model_name = os.path.splitext(os.path.basename(path))[0]
        for rec in read_jsonl(path):
            smiles = rec.get("input") or rec.get("smiles") or ""
            gt = rec.get("label", "")
            pred = rec.get("predict", "")

            # 13C metrics
            gt_13c_list = parse_13c_shifts(extract_13c_tag(gt))
            pred_13c_list = parse_13c_shifts(extract_13c_tag(pred))
            c_metrics = score_13c(gt_13c_list, pred_13c_list, tol=args.tolC)

            row = {
                "file": path,
                "model": model_name,
                "smiles": smiles,
                "gt_13C_tag": extract_13c_tag(gt),
                "pred_13C_tag": extract_13c_tag(pred),
                "c_n_gt": c_metrics["n_gt"],
                "c_n_pred": c_metrics["n_pred"],
                "c_n_match": c_metrics["n_match"],
                "c_recall": c_metrics["recall"],
                "c_precision": c_metrics["precision"],
                "c_f1": c_metrics["f1"],
                "c_mae": c_metrics["mae"],
            }

            if args.evalH:
                gt_h_tag = extract_1h_tag(gt)
                pred_h_tag = extract_1h_tag(pred)
                gt_h_peaks = parse_1h_peaks(gt_h_tag)
                pred_h_peaks = parse_1h_peaks(pred_h_tag)
                h_metrics = match_and_score_1h(
                    gt_h_peaks, pred_h_peaks, tol=args.tolH, sigma=args.sigmaH)
                row.update({
                    "gt_1H_tag": gt_h_tag,
                    "pred_1H_tag": pred_h_tag,
                    **h_metrics
                })

            per_sample.append(row)

    if not per_sample:
        raise SystemExit("No records parsed from provided file(s).")

    df = pd.DataFrame(per_sample)

    summary_aggs = {
        "c_f1": "mean",
        "c_recall": "mean",
        "c_precision": "mean",
        "c_mae": "mean",
        "c_n_gt": "sum",
        "c_n_pred": "sum",
        "c_n_match": "sum"
    }
    if "hnmr_f1" in df.columns:
        summary_aggs.update({
            "hnmr_score": "mean",
            "hnmr_f1": "mean",
            "hnmr_recall": "mean",
            "hnmr_precision": "mean",
            "hnmr_n_gt": "sum",
            "hnmr_n_pred": "sum",
            "hnmr_n_match": "sum"
        })

    summary = df.groupby(["model"], dropna=False).agg(
        summary_aggs).reset_index().sort_values("c_f1", ascending=False)

    detail_path = os.path.join(args.out, "nmr_eval_details.csv")
    summary_path = os.path.join(args.out, "nmr_eval_summary.csv")
    df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Wrote:\n  {summary_path}\n  {detail_path}")
    if args.evalH:
        print("1H metrics included (weighted Jaccard score + peak-level P/R/F1).")
    else:
        print("Run with --evalH to evaluate 1H NMR as well.")


if __name__ == "__main__":
    main()
