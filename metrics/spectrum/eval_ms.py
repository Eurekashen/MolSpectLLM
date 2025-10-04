# -*- coding: utf-8 -*-
"""
Evaluate a single MS spectrum JSONL results file.

Each JSONL record expected keys (flexible):
    - input / smiles
    - gt_answer or label (ground truth containing <ms...> tag)
    - llm_answer or predict (prediction containing <ms...> tag)

Parses MS tags:
    <ms_{mode}_fragments>...</ms_{mode}_fragments>
    <ms_positive_{energy}>...</ms_positive_{energy}>
    <ms_negative_{energy}>...</ms_negative_{energy}>

Accepted fragment notations:
    mz(intensity)   |  mz: intensity  |  mz intensity  |  mz(label)
    Intensities may be raw numbers (0-1 or percentages) or labels s/m/w/br.
    Qwen-style descriptive formats auto-parsed assigning decaying intensities.

Metrics per record:
    - cosine: Gaussian-broadened spectrum cosine similarity
    - jaccard: peak-set overlap within m/z tolerance
    - mae: intensity MAE over matched peaks

Outputs in --out:
    ms_eval_details.csv   (per record rows)
    ms_eval_summary.csv   (single summary row for the file/model)

Usage:
python eval_ms.py \
    --file /Users/shenshuaike/Documents/Research/TrimoLLM/eval_results/spectra_test_fix_final_report_2/bench_smiles2spec_ms/generated_predictions.jsonl \
    --out  /Users/shenshuaike/Documents/Research/TrimoLLM/eval_results/spectra_test_fix_final_report_2/bench_smiles2spec_ms/ \
    --mzmin 0 --mzmax 1000 --step 1 --sigma 0.5 --mztol 0.5
"""

import os
import re
import json
import math
import argparse
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

# (Directory iteration removed for single-file mode)


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


# ---------------- MS tag parsing ----------------
# Match any <ms...>...</ms...> tag (positive/negative/_fragments etc.)
MS_TAG = re.compile(r"<\s*ms[^>]*>(.*?)</\s*ms[^>]*>",
                    re.IGNORECASE | re.DOTALL)


def parse_ms_tag(text: str) -> List[Tuple[float, float]]:
    """
    Parse fragments from MS tag body into list of (mz, intensity).

    Accepts:
      - "100(0.8), 120(0.3)"       # numeric in parentheses
      - "100: 80, 120: 30"         # colon style (will rescale later)
      - "100 80; 120 30"           # whitespace style
      - "100(s), 120(m), 90(w)"    # text strength -> mapped to numeric
      - percentages "80%" allowed; intensities will be normalized to [0,1].
      - Qwen format: "[M-H]- 213.09, 198.07, 170.06" (auto-assign intensities)

    Returns list sorted by m/z ascending.
    """
    if not isinstance(text, str):
        return []
    m = MS_TAG.search(text)
    if not m:
        return []
    body = m.group(1)

    frags: List[Tuple[float, float]] = []

    # Check if this is Qwen format (contains [M-H]- or similar patterns without explicit intensities)
    is_qwen_format = (
        # Contains [M-H]-, [M-Na]-, etc.
        re.search(r'\[M-[H|Na|K|NH4]\]', body) or
        # Has m/z values but no intensities
        (re.search(r'\d+\.\d+', body) and not re.search(r'[\(\:\s]\d+', body)) or
        re.search(r'fragment ions:', body) or  # Contains "fragment ions:" text
        re.search(r'→', body) or  # Contains arrow symbol
        re.search(r'at m/z \d+', body) or  # Contains "at m/z" pattern
        # Contains "loss of" text (common in Qwen descriptions)
        re.search(r'loss of', body)
    )

    # Debug: Print the detection logic (only in debug mode)
    if os.environ.get('DEBUG_MS_PARSER', '0') == '1':
        print(f"[DEBUG] Qwen detection details:")
        print(f"  Body: {body}")
        mh_detected = bool(re.search(r'\[M-[H|Na|K|NH4]\]', body))
        mz_no_intensity = bool(re.search(r'\d+\.\d+', body)
                               and not re.search(r'[\(\:\s]\d+', body))
        fragment_detected = bool(re.search(r'fragment ions:', body))
        arrow_detected = bool(re.search(r'→', body))
        atmz_detected = bool(re.search(r'at m/z \d+', body))
        loss_detected = bool(re.search(r'loss of', body))
        print(f"  [M-H] pattern: {mh_detected}")
        print(f"  Has m/z but no intensities: {mz_no_intensity}")
        print(f"  fragment ions: {fragment_detected}")
        print(f"  arrow: {arrow_detected}")
        print(f"  at m/z: {atmz_detected}")
        print(f"  loss of: {loss_detected}")
        print(f"  Final result: {is_qwen_format}")

    if is_qwen_format:
        # Enhanced parsing for Qwen format
        # Extract all m/z values from the text (including both integers and decimals)
        # Use word boundaries to avoid extracting numbers from chemical formulas
        mz_values = re.findall(r'\b(\d+(?:\.\d+)?)\b', body)
        if mz_values:
            if os.environ.get('DEBUG_MS_PARSER', '0') == '1':
                print(
                    f"[DEBUG] Detected Qwen format, extracted {len(mz_values)} m/z values: {mz_values}")
            # Assign decreasing intensities: first peak = 100%, second = 80%, third = 60%, etc.
            for i, mz_str in enumerate(mz_values):
                mz = float(mz_str)
                # Exponential decay: 100%, 80%, 64%, 51%, 41%, etc.
                intensity = 1.0 * (0.8 ** i)
                frags.append((mz, intensity))
                if os.environ.get('DEBUG_MS_PARSER', '0') == '1':
                    print(
                        f"[DEBUG] Assigned m/z {mz} -> intensity {intensity:.3f}")
            # Sort by m/z and return
            frags.sort(key=lambda x: x[0])
            if os.environ.get('DEBUG_MS_PARSER', '0') == '1':
                print(f"[DEBUG] Qwen parsing result: {frags}")
            return frags
        else:
            if os.environ.get('DEBUG_MS_PARSER', '0') == '1':
                print(
                    f"[DEBUG] No m/z values found in Qwen format text: {body}")

    # Standard parsing for other models
    # 1) m/z(parenthesized numeric) e.g., 120(0.8) or 120(80) or 120(80%)
    for pm in re.finditer(r"(\d+(?:\.\d+)?)\s*\(\s*([+-]?\d+(?:\.\d+)?)(%?)\s*\)", body):
        mz = float(pm.group(1))
        val = float(pm.group(2))
        if pm.group(3):  # percentage
            inten = val / 100.0
        else:
            inten = val
        frags.append((mz, float(inten)))

    # 2) m/z(parenthesized text) e.g., 120(s), 90(weak), 130(br)
    for pm in re.finditer(r"(\d+(?:\.\d+)?)\s*\(\s*([a-zA-Z]+)\s*\)", body):
        mz = float(pm.group(1))
        label = pm.group(2).lower()
        if label in ["s", "strong", "str"]:
            inten = 1.0
        elif label in ["m", "med", "medium"]:
            inten = 0.5
        elif label in ["w", "weak"]:
            inten = 0.2
        elif label in ["br", "broad"]:
            inten = 0.3
        else:
            inten = 0.5
        frags.append((mz, float(inten)))

    # 3) colon style "120: 80" or "120 : 0.8"
    for pm in re.finditer(r"(\d+(?:\.\d+)?)\s*:\s*([+-]?\d+(?:\.\d+)?)(%?)", body):
        mz = float(pm.group(1))
        val = float(pm.group(2))
        inten = val / 100.0 if pm.group(3) else val
        frags.append((mz, float(inten)))

    # 4) whitespace pairs "120 80" (avoid matching inside earlier styles)
    # Use a conservative pattern: number space number, bounded by separators
    for pm in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)(%?)(?![\w.])", body):
        mz = float(pm.group(1))
        val = float(pm.group(2))
        inten = val / 100.0 if pm.group(3) else val
        frags.append((mz, float(inten)))

    # Merge duplicates by summing intensities
    merged = {}
    for mz, inten in frags:
        merged[mz] = merged.get(mz, 0.0) + float(inten)
    out = [(mz, merged[mz]) for mz in merged]
    out.sort(key=lambda x: x[0])
    return out

# ---------------- Spectrum building & metrics ----------------


def to_unit(y: np.ndarray) -> np.ndarray:
    # Avoid degenerate scaling; L2-normalization is applied inside cosine
    return y


def sticks_to_grid(peaks: List[Tuple[float, float]], grid: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """
    Convert stick spectrum (m/z, intensity) to continuous grid via Gaussian broadening.
    sigma in Da. If sigma <= 0, accumulate at nearest bin.
    Intensities will be linearly rescaled to [0,1] per spectrum if max>0.
    """
    y = np.zeros_like(grid, dtype=float)
    if not peaks:
        return y

    # normalize intensities per spectrum to [0,1] (common in MS)
    maxI = max(abs(I) for _, I in peaks) if peaks else 0.0
    norm_peaks = [(mz, (I / maxI) if maxI > 0 else 0.0) for mz, I in peaks]

    if sigma <= 0:
        for mz, I in norm_peaks:
            idx = int(np.argmin(np.abs(grid - mz)))
            y[idx] += I
        return y

    width = int(max(1, math.ceil(4*sigma)))
    for mz, I in norm_peaks:
        c = int(np.searchsorted(grid, mz))
        left = max(0, c - width)
        right = min(len(grid)-1, c + width)
        xs = grid[left:right+1]
        w = np.exp(-0.5 * ((xs - mz)/sigma)**2)
        y[left:right+1] += I * w
    return y


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def jaccard_peaks(gt: List[Tuple[float, float]], pr: List[Tuple[float, float]], mztol: float = 0.5) -> float:
    """
    Peak-set Jaccard with m/z tolerance:
      - Greedy match peaks within ±mztol Da (each gt at most once).
      - J = matched / (len(gt)+len(pr)-matched)
    Only uses m/z positions (not intensities).
    """
    gt_mz = [mz for mz, _ in gt]
    pr_mz = [mz for mz, _ in pr]
    gt_used = set()
    matched = 0
    for mz in pr_mz:
        # nearest unused gt within tolerance
        best_j, best_d = None, None
        for j, g in enumerate(gt_mz):
            if j in gt_used:
                continue
            d = abs(mz - g)
            if d <= mztol and (best_d is None or d < best_d):
                best_j, best_d = j, d
        if best_j is not None:
            gt_used.add(best_j)
            matched += 1
    denom = len(gt_mz) + len(pr_mz) - matched
    return matched/denom if denom > 0 else 1.0


def mae_intensity(gt: List[Tuple[float, float]], pr: List[Tuple[float, float]], mztol: float = 0.5) -> Optional[float]:
    """
    Match peaks within ±mztol Da (greedy); compute MAE of intensities (after per-spectrum [0,1] normalization).
    """
    if not gt and not pr:
        return None
    # Normalize intensities

    def norm(peaks):
        if not peaks:
            return []
        m = max(abs(I) for _, I in peaks)
        return [(mz, (I/m if m > 0 else 0.0)) for mz, I in peaks]
    gt_n = norm(gt)
    pr_n = norm(pr)

    gt_n.sort(key=lambda x: x[0])
    pr_n.sort(key=lambda x: x[0])
    used = set()
    diffs = []
    for mz, I in pr_n:
        best = None
        best_d = None
        for j, (gmz, gI) in enumerate(gt_n):
            if j in used:
                continue
            d = abs(mz - gmz)
            if d <= mztol and (best_d is None or d < best_d):
                best = (j, gI)
                best_d = d
        if best is not None:
            j, gI = best
            used.add(j)
            diffs.append(abs(I - gI))
    if not diffs:
        return None
    return float(np.mean(diffs))

# ---------------- Evaluation ----------------


def evaluate_ms_single(path: str, mzmin=0, mzmax=1000, step=1, sigma=0.5, mztol=0.5):
    grid = np.arange(mzmin, mzmax + step, step, dtype=float)
    model = os.path.splitext(os.path.basename(path))[0]
    rows = []
    for rec in read_jsonl(path):
        smiles = rec.get("input") or rec.get("smiles") or ""
        gt_tag = rec.get("gt_answer") or rec.get("label") or ""
        pr_tag = rec.get("llm_answer") or rec.get("predict") or ""

        gt = parse_ms_tag(gt_tag)
        pr = parse_ms_tag(pr_tag)

        y_gt = sticks_to_grid(gt, grid, sigma=sigma)
        y_pr = sticks_to_grid(pr, grid, sigma=sigma)

        cs = cosine_sim(y_gt, y_pr)
        jac = jaccard_peaks(gt, pr, mztol=mztol)
        mae = mae_intensity(gt, pr, mztol=mztol)

        rows.append({
            "model": model,
            "file": path,
            "smiles": smiles,
            "gt_npeaks": len(gt),
            "pred_npeaks": len(pr),
            "pred_nonempty": int(len(pr) > 0),
            "cosine": cs,
            "jaccard": jac,
            "mae": mae
        })
    df = pd.DataFrame(rows)
    # Single summary row
    nonzero = df[df["pred_nonempty"] > 0]
    summary = pd.DataFrame({
        "model": [model],
        "n": [len(df)],
        "avg_cosine": [df["cosine"].mean()],
        "med_cosine": [df["cosine"].median()],
        "avg_jaccard": [df["jaccard"].mean()],
        "med_jaccard": [df["jaccard"].median()],
        "avg_mae": [df["mae"].mean()],
        "avg_gt_peaks": [df["gt_npeaks"].mean()],
        "avg_pred_peaks": [df["pred_npeaks"].mean()],
        "nonempty_ratio": [df["pred_nonempty"].mean()],
        "avg_cosine_nonzero": [nonzero["cosine"].mean() if len(nonzero) else float("nan")],
        "med_cosine_nonzero": [nonzero["cosine"].median() if len(nonzero) else float("nan")],
        "avg_jaccard_nonzero": [nonzero["jaccard"].mean() if len(nonzero) else float("nan")],
        "med_jaccard_nonzero": [nonzero["jaccard"].median() if len(nonzero) else float("nan")],
        "avg_mae_nonzero": [nonzero["mae"].mean() if len(nonzero) else float("nan")]
    })
    return df, summary

# ---------------- Main ----------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True,
                    help="MS JSONL results file to evaluate")
    ap.add_argument("--out", required=True, help="Output directory for CSVs")
    ap.add_argument("--mzmin", type=float, default=0.0, help="Grid min m/z")
    ap.add_argument("--mzmax", type=float, default=1000.0, help="Grid max m/z")
    ap.add_argument("--step", type=float, default=1.0, help="Grid step (Da)")
    ap.add_argument("--sigma", type=float, default=0.5,
                    help="Gaussian broadening (Da)")
    ap.add_argument("--mztol", type=float, default=0.5,
                    help="m/z tolerance for Jaccard/MAE (Da)")
    args = ap.parse_args()

    if not os.path.isfile(args.file):
        raise SystemExit(f"File not found: {args.file}")
    os.makedirs(args.out, exist_ok=True)

    df_detail, df_summary = evaluate_ms_single(
        args.file, mzmin=args.mzmin, mzmax=args.mzmax, step=args.step, sigma=args.sigma, mztol=args.mztol
    )

    detail_path = os.path.join(args.out, "ms_eval_details.csv")
    summary_path = os.path.join(args.out, "ms_eval_summary.csv")
    df_detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("[OK] wrote:")
    print(" ", summary_path)
    print(" ", detail_path)
    print(
        f"Params: grid=({args.mzmin}~{args.mzmax}, step={args.step}), sigma={args.sigma}, mztol={args.mztol}")


if __name__ == "__main__":
    main()
