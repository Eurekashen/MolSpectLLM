"""Assess steric clashes in generated 3D molecular structures."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

# Empirical van der Waals radii (Å) for common elements. Unknown elements fall back to 1.8 Å.
VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
    "B": 1.92,
    "He": 1.40,
    "Ne": 1.54,
    "Ar": 1.88,
    "Li": 1.82,
    "Be": 1.53,
    "Na": 2.27,
    "Mg": 1.73,
    "Al": 1.84,
    "Si": 2.10,
    "K": 2.75,
    "Ca": 2.31,
    "Fe": 1.84,
    "Cu": 1.40,
    "Zn": 1.39,
}

TOLERANCE_FACTOR = 0.4  # Relax clash threshold compared to the raw VdW sum.


def count_clashes_in_sdf(sdf_path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Return the number of van-der-Waals clashes in the first molecule of ``sdf_path``."""

    try:
        supplier = Chem.SDMolSupplier(str(sdf_path))
        mol = next(iter(supplier), None)
    except Exception as exc:  # pragma: no cover - RDKit error path
        print(f"Failed to read {sdf_path}: {exc}")
        return None, None

    if mol is None:
        return None, None

    conf = mol.GetConformer()
    positions = np.array([
        list(conf.GetAtomPosition(i))
        for i in range(mol.GetNumAtoms())
    ])

    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    radii = np.array([VDW_RADII.get(sym, 1.8) for sym in symbols])

    dist_matrix = np.linalg.norm(positions[:, None] - positions, axis=2)
    vdw_sum_matrix = radii[:, None] + radii

    clash_matrix = (dist_matrix < TOLERANCE_FACTOR * vdw_sum_matrix) & (dist_matrix > 0)
    clash_count = int(np.sum(clash_matrix) // 2)
    num_atoms = mol.GetNumAtoms()
    return clash_count, num_atoms


def analyse_folder(folder_path: Path, pattern: str = "molecule_*_pred.sdf") -> Tuple[pd.DataFrame, dict[str, float]]:
    """Analyse all predicted SDF files in ``folder_path`` and compute clash statistics."""

    sdf_files = sorted(folder_path.glob(pattern))
    if not sdf_files:
        raise FileNotFoundError(f"No SDF files matching '{pattern}' found in {folder_path}")

    rows = []
    for sdf_file in tqdm(sdf_files, desc="Analysing SDF files"):
        clash_count, num_atoms = count_clashes_in_sdf(sdf_file)
        if clash_count is None or num_atoms is None:
            continue
        rows.append(
            {
                "file": sdf_file.name,
                "clash_count": clash_count,
                "num_atoms": num_atoms,
                "clash_density": clash_count / num_atoms if num_atoms > 0 else float("nan"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid SDF files were processed.")

    stats = {
        "total_files": float(len(df)),
        "total_atoms": float(df["num_atoms"].sum()),
        "total_clashes": float(df["clash_count"].sum()),
        "mean_clash_count": float(df["clash_count"].mean()),
        "std_clash_count": float(df["clash_count"].std(ddof=0)),
        "mean_clash_density": float(df["clash_density"].mean()),
        "std_clash_density": float(df["clash_density"].std(ddof=0)),
        "min_clash_count": float(df["clash_count"].min()),
        "max_clash_count": float(df["clash_count"].max()),
        "min_clash_density": float(df["clash_density"].min()),
        "max_clash_density": float(df["clash_density"].max()),
    }

    return df, stats


def render_summary(stats: dict[str, float]) -> str:
    """Create a human-readable summary string for the clash statistics."""

    return (
        "===== Clash statistics =====\n"
        f"Analysed files: {int(stats['total_files'])}\n"
        f"Total atoms: {int(stats['total_atoms'])}\n"
        f"Total clashes: {int(stats['total_clashes'])}\n\n"
        "Collision counts:\n"
        f"  mean = {stats['mean_clash_count']:.2f}, std = {stats['std_clash_count']:.2f}\n"
        f"  min = {stats['min_clash_count']:.0f}, max = {stats['max_clash_count']:.0f}\n\n"
        "Clash density (per atom):\n"
        f"  mean = {stats['mean_clash_density']:.4f}, std = {stats['std_clash_density']:.4f}\n"
        f"  min = {stats['min_clash_density']:.4f}, max = {stats['max_clash_density']:.4f}\n"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute steric clash statistics for predicted SDF structures.")
    parser.add_argument("folder", type=Path, help="Directory containing predicted SDF files.")
    parser.add_argument("--pattern", default="molecule_*_pred.sdf", help="Glob pattern used to locate SDF files.")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV path for detailed results.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    df, stats = analyse_folder(args.folder, pattern=args.pattern)
    print(render_summary(stats))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Detailed results written to {args.output}")


if __name__ == "__main__":
    main()
