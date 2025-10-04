# -*- coding: utf-8 -*-
"""
Evaluate IR spectra from a single JSONL result file using cosine similarity:
    1) Parse <IR>(range) peaks ... </IR>
    2) Convert stick peaks to broadened spectrum (Gaussian σ)
    3) Cosine similarity between ground truth and prediction spectra

Supports numeric intensities and textual labels: s/strong, m/medium, w/weak, br/broad.
Outputs two CSVs in --out:
    ir_cosine_details.csv  (per record)
    ir_cosine_summary.csv  (aggregate for the file/model)

Usage:
python eval_ir.py \
    --file /Users/shenshuaike/Documents/Research/TrimoLLM/eval_results/spectra_test_fix_final_report_2/bench_smiles2spec_ir/generated_predictions.jsonl \
    --out  /Users/shenshuaike/Documents/Research/TrimoLLM/eval_results/spectra_test_fix_final_report_2/bench_smiles2spec_ir/ \
    --min 400 --max 4000 --step 1 --sigma 8
"""

import os
import re
import json
import math
import argparse
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

IR_TAG = re.compile(r"<IR>\s*\((.*?)\)\s*(.*?)</IR>",
                    re.IGNORECASE | re.DOTALL)

# --- file iteration (single file mode) ---


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

# --- parsing ---


def parse_ir_tag(text: str) -> Tuple[Optional[Tuple[int, int]], List[Tuple[float, float]]]:
    """
    Parse IR tag: <IR>(500~4000) 1710(0.8), 1200(m), 2950(strong), 3300(br), ...</IR>
    Returns: (range_min, range_max) or None, and list of (frequency_cm1, intensity in [0,1]).
    """
    if not isinstance(text, str):
        return None, []
    m = IR_TAG.search(text)
    if not m:
        return None, []
    rng_str = m.group(1).strip()
    peaks_str = m.group(2).strip()

    # range
    r = None
    m2 = re.match(r"\s*(\d+)\s*~\s*(\d+)\s*$", rng_str)
    if m2:
        r = (int(m2.group(1)), int(m2.group(2)))

    peaks: List[Tuple[float, float]] = []

    # numeric intensity: e.g., 1710(0.8)
    for pm in re.finditer(r"(\d+(?:\.\d+)?)\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*\)", peaks_str):
        f = float(pm.group(1))
        I = float(pm.group(2))
        peaks.append((f, I))

    # textual intensity: s/strong, m/medium, w/weak, br/broad
    for pm in re.finditer(r"(\d+(?:\.\d+)?)\s*\(\s*([a-zA-Z]+)\s*\)", peaks_str):
        f = float(pm.group(1))
        label = pm.group(2).lower()
        if label in ["s", "strong", "str"]:
            I = 1.0
        elif label in ["m", "med", "medium"]:
            I = 0.5
        elif label in ["w", "weak"]:
            I = 0.2
        elif label in ["br", "broad"]:
            I = 0.3
        else:
            I = 0.5  # default medium
        peaks.append((f, I))

    # sort by frequency ascending for consistent ordering
    peaks.sort(key=lambda x: x[0])

    return r, peaks

# --- spectrum construction & similarity ---


def stick_to_grid(peaks: List[Tuple[float, float]], grid: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """
    Convert stick spectrum to continuous grid via Gaussian broadening (σ in cm^-1).
    If sigma <= 0, place intensity to nearest grid bin (no broadening).
    """
    y = np.zeros_like(grid, dtype=float)
    if not peaks:
        return y
    if sigma <= 0:
        for f, I in peaks:
            idx = int(np.argmin(np.abs(grid - f)))
            y[idx] += I
        return y

    width = int(max(1, math.ceil(4*sigma)))  # truncate kernel at ±4σ
    for f, I in peaks:
        c = int(np.searchsorted(grid, f))
        left = max(0, c - width)
        right = min(len(grid)-1, c + width)
        xs = grid[left:right+1]
        w = np.exp(-0.5 * ((xs - f)/sigma)**2)
        y[left:right+1] += I * w
    return y


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def evaluate_ir_single(path: str, grid_min=400, grid_max=4000, step=1, sigma=8.0):
    grid = np.arange(grid_min, grid_max + step, step, dtype=float)
    model = os.path.splitext(os.path.basename(path))[0]
    rows = []
    for rec in read_jsonl(path):
        smiles = rec.get("input") or rec.get("smiles") or ""
        gt_tag = rec.get("label", "")
        pred_tag = rec.get("predict", "")
        _, gt_peaks = parse_ir_tag(gt_tag)
        _, pr_peaks = parse_ir_tag(pred_tag)
        y_gt = stick_to_grid(gt_peaks, grid, sigma=sigma)
        y_pr = stick_to_grid(pr_peaks, grid, sigma=sigma)
        cs = cosine_sim(y_gt, y_pr)
        rows.append({
            "model": model,
            "file": path,
            "smiles": smiles,
            "cosine": cs,
            "gt_npeaks": len(gt_peaks),
            "pred_npeaks": len(pr_peaks),
            "pred_nonempty": int(len(pr_peaks) > 0)
        })
    df = pd.DataFrame(rows)
    # build summary row
    summary_df = pd.DataFrame({
        "model": [model],
        "n": [len(df)],
        "avg_cosine": [df["cosine"].mean()],
        "med_cosine": [df["cosine"].median()],
        "avg_gt_peaks": [df["gt_npeaks"].mean()],
        "avg_pred_peaks": [df["pred_npeaks"].mean()],
        "nonempty_ratio": [df["pred_nonempty"].mean()],
        "avg_cosine_nonzero": [df[df["pred_nonempty"] > 0]["cosine"].mean() if (df["pred_nonempty"] > 0).any() else float("nan")],
        "med_cosine_nonzero": [df[df["pred_nonempty"] > 0]["cosine"].median() if (df["pred_nonempty"] > 0).any() else float("nan")]
    })
    return df, summary_df

# --- main ---


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True,
                    help="IR JSONL result file to evaluate")
    ap.add_argument("--out", required=True, help="Output directory for CSVs")
    ap.add_argument("--min", type=int, default=400,
                    help="Grid min wavenumber (cm^-1)")
    ap.add_argument("--max", type=int, default=4000,
                    help="Grid max wavenumber (cm^-1)")
    ap.add_argument("--step", type=int, default=1, help="Grid step (cm^-1)")
    ap.add_argument("--sigma", type=float, default=8.0,
                    help="Gaussian broadening sigma (cm^-1)")
    args = ap.parse_args()

    if not os.path.isfile(args.file):
        raise SystemExit(f"File not found: {args.file}")

    os.makedirs(args.out, exist_ok=True)

    df_detail, df_summary = evaluate_ir_single(
        args.file, grid_min=args.min, grid_max=args.max, step=args.step, sigma=args.sigma
    )

    detail_path = os.path.join(args.out, "ir_cosine_details.csv")
    summary_path = os.path.join(args.out, "ir_cosine_summary.csv")
    df_detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("[OK] wrote:")
    print(" ", summary_path)
    print(" ", detail_path)
    print(
        f"Params: grid=({args.min}~{args.max}, step={args.step}), sigma={args.sigma}")


if __name__ == "__main__":
    main()
