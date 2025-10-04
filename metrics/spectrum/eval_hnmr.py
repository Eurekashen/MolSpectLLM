# -*- coding: utf-8 -*-
"""
Evaluate 1H NMR predictions in JSONL benchmark files.

Each JSONL record:
- "input" (SMILES)
- "gt_answer"  (with <1H_NMR>...</1H_NMR>)
- "llm_answer" (same format)

Metrics:
1) Weighted Jaccard score (by nH, δ tolerance & Gaussian weighting)
2) Peak-level precision / recall / F1 (count-based)
3) MAE (ppm) for matched peaks

Usage:
    python eval_hnmr.py --root /Users/shenshuaike/Documents/Research/TrimoLLM/eval_results/spectra_test_fix_final_report_2/bench_smiles2spec_hnmr/generated_predictions.jsonl --out /Users/shenshuaike/Documents/Research/TrimoLLM/eval_results/spectra_test_fix_final_report_2/bench_smiles2spec_hnmr/ --tol 0.12 --sigma 0.06
"""

import argparse
import os
import json
import re
import math
import pandas as pd

_H_TAG = re.compile(r"<1H_NMR>(.*?)</1H_NMR>", re.IGNORECASE | re.DOTALL)
_PEAK_PATTERN = re.compile(
    r"\s*([0-9]*\.?[0-9]+)(?:\s*-\s*([0-9]*\.?[0-9]+))?\s*\(([^)]*?)\)")


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
            except:
                continue


def extract_1h_tag(text: str):
    if not isinstance(text, str):
        return None
    m = _H_TAG.search(text)
    return m.group(1).strip() if m else None


def parse_1h_peaks(tag_content: str):
    if not tag_content:
        return []
    s = tag_content
    m = re.search(r"δ(.*)$", s, flags=re.DOTALL)
    s = m.group(1) if m else s
    peaks = []
    for m in _PEAK_PATTERN.finditer(s):
        a = float(m.group(1))
        b = float(m.group(2)) if m.group(2) else None
        info = m.group(3)
        nHs = re.findall(r"(\d+)\s*H", info, flags=re.IGNORECASE)
        nH = int(nHs[-1]) if nHs else 1
        shift = (a+b)/2.0 if b is not None else a
        peaks.append({"shift": shift, "nH": nH, "raw": m.group(0).strip()})
    return peaks


def match_and_score(gt_peaks, pred_peaks, tol=0.12, sigma=0.06):
    gt_used = set()
    matched_w, n_match = 0.0, 0
    deltas = []
    for p in pred_peaks:
        best, best_score = None, -1.0
        for j, g in enumerate(gt_peaks):
            if j in gt_used:
                continue
            d = abs(p["shift"]-g["shift"])
            if d <= tol:
                w = min(p["nH"], g["nH"])*math.exp(-0.5*(d/sigma)**2)
                if w > best_score:
                    best_score = w
                    best = (j, g, d)
        if best is not None:
            j, g, d = best
            gt_used.add(j)
            matched_w += best_score
            n_match += 1
            deltas.append(d)
    sum_pred = sum(p["nH"] for p in pred_peaks)
    sum_gt = sum(g["nH"] for g in gt_peaks)
    score = matched_w/(sum_pred+sum_gt-matched_w+1e-12)
    n_gt, n_pred = len(gt_peaks), len(pred_peaks)
    prec = n_match/n_pred if n_pred else 0.0
    rec = n_match/n_gt if n_gt else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    mae = sum(deltas)/len(deltas) if deltas else None
    return dict(h_score=score, h_prec=prec, h_rec=rec, h_f1=f1, h_mae=mae,
                h_n_gt=n_gt, h_n_pred=n_pred, h_n_match=n_match)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=False,
                    default="/Users/shenshuaike/Documents/Research/TrimoLLM/eval_results/bench_smiles2spec_hnmr/generated_predictions.jsonl", help="Root dir with JSONL files")
    ap.add_argument("--out", required=False,
                    default="/Users/shenshuaike/Documents/Research/TrimoLLM/eval_results/bench_smiles2spec_hnmr", help="Output dir for CSVs")
    ap.add_argument("--tol", type=float, default=0.12, help="ppm tolerance")
    ap.add_argument("--sigma", type=float, default=0.06, help="Gaussian sigma")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # per_sample=[]
    # for path in iter_jsonl_files(args.root):
    #     model=os.path.splitext(os.path.basename(path))[0]
    #     for rec in read_jsonl(path):
    #         gt=parse_1h_peaks(extract_1h_tag(rec.get("gt_answer","")))
    #         pred=parse_1h_peaks(extract_1h_tag(rec.get("llm_answer","")))
    #         metrics=match_and_score(gt,pred,tol=args.tol,sigma=args.sigma)
    #         row={"file":path,"model":model,"smiles":rec.get("input") or rec.get("smiles") or ""}
    #         row.update(metrics)
    #         per_sample.append(row)

    # df=pd.DataFrame(per_sample)
    # summary=df.groupby("model").agg({
    #     "h_score":"mean","h_prec":"mean","h_rec":"mean","h_f1":"mean","h_mae":"mean"
    # }).reset_index().sort_values("h_score",ascending=False)

    # df.to_csv(os.path.join(args.out,"hnmr_eval_details.csv"),index=False,encoding="utf-8-sig")
    # summary.to_csv(os.path.join(args.out,"hnmr_eval_summary.csv"),index=False,encoding="utf-8-sig")
    # print("[OK] wrote results to",args.out)

    # eval a single file
    path = args.root
    per_sample = []
    model = os.path.splitext(os.path.basename(path))[0]
    for rec in read_jsonl(path):
        gt = parse_1h_peaks(extract_1h_tag(rec.get("label", "")))
        pred = parse_1h_peaks(extract_1h_tag(rec.get("predict", "")))
        metrics = match_and_score(gt, pred, tol=args.tol, sigma=args.sigma)
        row = {"file": path, "model": model, "smiles": rec.get(
            "input") or rec.get("smiles") or ""}
        row.update(metrics)
        per_sample.append(row)
    df = pd.DataFrame(per_sample)
    summary = df.agg({
        "h_score": "mean", "h_prec": "mean", "h_rec": "mean", "h_f1": "mean", "h_mae": "mean"
    }).to_frame().T
    summary.insert(0, "model", model)
    df.to_csv(os.path.join(args.out, "hnmr_eval_details.csv"),
              index=False, encoding="utf-8-sig")
    summary.to_csv(os.path.join(args.out, "hnmr_eval_summary.csv"),
                   index=False, encoding="utf-8-sig")
    print("[OK] wrote results to", args.out)
    # print results
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
