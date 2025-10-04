"""Fingerprint-based similarity metrics for generated 3D molecular structures.

The module scans an SDF directory, pairs predicted geometries with their ground
truth counterparts, and reports the average Tanimoto similarity between RDKit
RDK fingerprints. Use the ``main`` entry point for quick command-line
summaries or import :func:`evaluate_directory` to obtain the raw scores.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

import numpy as np
from rdkit import Chem, DataStructs


@dataclass
class FingerprintResult:
    """Container for the fingerprint similarity of a single SDF pair."""

    prediction: Path
    reference: Path
    similarity: float

    def as_dict(self) -> dict[str, str | float]:
        return {
            "prediction": str(self.prediction),
            "reference": str(self.reference),
            "similarity": float(self.similarity),
        }


def iter_prediction_pairs(sdf_dir: Path, pred_keyword: str = "pred", gt_keyword: str = "gt") -> Iterator[tuple[Path, Path]]:
    """Yield matched ``(prediction, reference)`` SDF file pairs.

    Files ending with ``_gt.sdf`` are treated as references by default. For
    each predicted structure we attempt to construct the corresponding ground
    truth path by replacing ``pred_keyword`` with ``gt_keyword`` in the file
    name. Only existing pairs are returned.
    """

    for pred_path in sorted(sdf_dir.glob("*.sdf")):
        if pred_path.name.endswith("_{}".format(gt_keyword) + ".sdf"):
            continue
        if pred_keyword not in pred_path.name:
            # Allow implicit predictions such as ``molecule_0.sdf``
            candidate = pred_path.with_name(pred_path.stem + f"_{gt_keyword}.sdf")
        else:
            candidate = pred_path.with_name(pred_path.name.replace(pred_keyword, gt_keyword))
        if candidate.exists():
            yield pred_path, candidate


def load_first_molecule(sdf_path: Path):
    """Return the first molecule in an SDF file or ``None`` if parsing fails."""

    try:
        supplier = Chem.SDMolSupplier(str(sdf_path))
        mol = next(iter(supplier), None)
        if mol is not None:
            Chem.SanitizeMol(mol, catchErrors=True)
        return mol
    except Exception:
        return None


def fingerprint_similarity(pred_path: Path, ref_path: Path) -> FingerprintResult | None:
    """Compute RDK fingerprint similarity for a single SDF pair."""

    pred_mol = load_first_molecule(pred_path)
    ref_mol = load_first_molecule(ref_path)
    if pred_mol is None or ref_mol is None:
        return None

    pred_fp = Chem.RDKFingerprint(pred_mol)
    ref_fp = Chem.RDKFingerprint(ref_mol)
    similarity = DataStructs.FingerprintSimilarity(pred_fp, ref_fp)
    return FingerprintResult(prediction=pred_path, reference=ref_path, similarity=similarity)


def evaluate_directory(
    sdf_dir: Path,
    pred_keyword: str = "pred",
    gt_keyword: str = "gt",
) -> List[FingerprintResult]:
    """Evaluate all available SDF pairs in ``sdf_dir``.

    Parameters
    ----------
    sdf_dir:
        Directory that contains predicted/ground-truth SDF files.
    pred_keyword, gt_keyword:
        Tokens used to convert a predicted file name into its reference name.

    Returns
    -------
    list[FingerprintResult]
        Per-pair results for downstream aggregation.
    """

    results: List[FingerprintResult] = []
    for pred_path, ref_path in iter_prediction_pairs(sdf_dir, pred_keyword, gt_keyword):
        result = fingerprint_similarity(pred_path, ref_path)
        if result is not None:
            results.append(result)
    return results


def summarise(results: Sequence[FingerprintResult]) -> dict[str, float]:
    """Return mean and standard deviation for a list of similarities."""

    if not results:
        return {"mean": float("nan"), "std": float("nan"), "count": 0}

    scores = np.array([r.similarity for r in results], dtype=float)
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=0)),
        "count": int(len(scores)),
    }


def format_table(results: Sequence[FingerprintResult]) -> str:
    """Create a readable table of per-molecule similarities."""

    lines = ["Prediction,Reference,Similarity"]
    for r in results:
        lines.append(f"{r.prediction},{r.reference},{r.similarity:.4f}")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute fingerprint similarity statistics for SDF predictions.")
    parser.add_argument("sdf_dir", type=Path, help="Directory containing predicted and reference SDF files.")
    parser.add_argument("--pred-keyword", default="pred", help="Token identifying prediction files (default: %(default)s).")
    parser.add_argument("--gt-keyword", default="gt", help="Token identifying reference files (default: %(default)s).")
    parser.add_argument("--show-table", action="store_true", help="Print a CSV table of per-molecule similarities.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    results = evaluate_directory(args.sdf_dir, pred_keyword=args.pred_keyword, gt_keyword=args.gt_keyword)
    stats = summarise(results)

    print(f"Evaluated {stats['count']} SDF pairs in {args.sdf_dir}.")
    print(f"Mean fingerprint similarity: {stats['mean']:.4f}")
    print(f"Std. dev.: {stats['std']:.4f}")

    if args.show_table:
        print("\nPer-molecule similarities (CSV):")
        print(format_table(results))


if __name__ == "__main__":
    main()
