"""Validate 3D structures derived from SMILES-to-coordinates workflows."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms


def load_first_molecule(path: Path):
    """Return the first molecule from an SDF file."""

    supplier = Chem.SDMolSupplier(str(path))
    mol = next(iter(supplier), None)
    if mol is None:
        raise ValueError(f"No molecule could be read from {path}")
    return mol


def collect_atomic_radii(mol: Chem.Mol) -> Tuple[dict[int, float], dict[int, float]]:
    """Return van der Waals and covalent radii keyed by atom index."""

    periodic_table = Chem.GetPeriodicTable()
    vdw_radii: dict[int, float] = {}
    covalent_radii: dict[int, float] = {}
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        idx = atom.GetIdx()
        vdw_radii[idx] = periodic_table.GetRvdw(atomic_num)
        covalent_radii[idx] = periodic_table.GetRcovalent(atomic_num)
    return vdw_radii, covalent_radii


def identify_adjacent_pairs(mol: Chem.Mol) -> Tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Identify 1-2 (bonded) and 1-3 (angle) atom relationships."""

    bonded_12: set[tuple[int, int]] = set()
    bonded_13: set[tuple[int, int]] = set()
    n_atoms = mol.GetNumAtoms()

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bonded_12.add((i, j))
        bonded_12.add((j, i))

        for k in range(n_atoms):
            if k in (i, j):
                continue
            if mol.GetBondBetweenAtoms(i, k) and mol.GetBondBetweenAtoms(j, k):
                bonded_13.add((i, j))
                bonded_13.add((j, i))

    return bonded_12, bonded_13


def detect_steric_clashes(mol: Chem.Mol, conf_id: int = 0) -> List[tuple[tuple[int, int], float, float]]:
    """Return atom pairs whose distance falls below a radius-scaled threshold."""

    conf = mol.GetConformer(conf_id)
    vdw_radii, _ = collect_atomic_radii(mol)
    bonded_12, bonded_13 = identify_adjacent_pairs(mol)

    n_atoms = mol.GetNumAtoms()
    clashes: List[tuple[tuple[int, int], float, float]] = []
    for i in range(n_atoms):
        pos_i = np.array(conf.GetAtomPosition(i))
        for j in range(i + 1, n_atoms):
            if (i, j) in bonded_12 or (i, j) in bonded_13:
                continue
            pos_j = np.array(conf.GetAtomPosition(j))
            distance = float(np.linalg.norm(pos_i - pos_j))
            min_allowed = 0.656 * (vdw_radii[i] + vdw_radii[j])
            if distance < min_allowed:
                clashes.append(((i, j), distance, min_allowed))
    return clashes


def detect_bond_length_violations(mol: Chem.Mol, conf_id: int = 0) -> List[tuple[tuple[int, int], float, float, float]]:
    """Return bonds whose length falls outside relaxed covalent limits."""

    conf = mol.GetConformer(conf_id)
    _, covalent_radii = collect_atomic_radii(mol)
    violations: List[tuple[tuple[int, int], float, float, float]] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        length = rdMolTransforms.GetBondLength(conf, i, j)

        bond_order = bond.GetBondTypeAsDouble()
        cov_ref = covalent_radii[i] + covalent_radii[j]
        if bond_order >= 1.5:
            cov_ref *= 0.9
        if bond_order >= 2.5:
            cov_ref *= 0.8

        lower_bound = cov_ref * 0.8
        upper_bound = cov_ref * 1.2
        allowed_lower = 0.75 * lower_bound
        allowed_upper = 1.25 * upper_bound

        if length < allowed_lower or length > allowed_upper:
            violations.append(((i, j), float(length), float(allowed_lower), float(allowed_upper)))

    return violations


def check_steric_clash_and_bond_lengths(sdf_file: Path) -> tuple[List[tuple[tuple[int, int], float, float]], List[tuple[tuple[int, int], float, float, float]]]:
    """Analyse a single SDF file for steric clashes and bond length issues."""

    mol = load_first_molecule(sdf_file)
    return detect_steric_clashes(mol), detect_bond_length_violations(mol)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate bond lengths and steric clashes in SDF files.")
    parser.add_argument("root", type=Path, help="Directory containing predicted SDF files.")
    parser.add_argument("--pattern", default="*.sdf", help="Glob pattern for SDF files (default: %(default)s).")
    parser.add_argument("--exclude-gt", action="store_true", help="Skip files ending with '_gt.sdf'.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    pattern = str(args.root / args.pattern)
    sdf_files = [
        Path(f)
        for f in glob.glob(pattern)
        if not (args.exclude_gt and f.endswith("_gt.sdf"))
    ]

    if not sdf_files:
        raise FileNotFoundError(f"No SDF files found for pattern {pattern}")

    from tqdm import tqdm

    clash_counts: List[int] = []
    violation_counts: List[int] = []
    failed = 0

    for sdf_file in tqdm(sdf_files, desc="Validating SDF files"):
        try:
            clashes, violations = check_steric_clash_and_bond_lengths(sdf_file)
            clash_counts.append(len(clashes))
            violation_counts.append(len(violations))
        except Exception as exc:
            print(f"Error processing {sdf_file}: {exc}")
            failed += 1

    if clash_counts:
        print(f"Mean steric clashes: {np.mean(clash_counts):.2f} ± {np.std(clash_counts):.2f}")
    if violation_counts:
        print(f"Mean bond violations: {np.mean(violation_counts):.2f} ± {np.std(violation_counts):.2f}")
    print(f"Failed to validate {failed} file(s).")
    print(f"Processed {len(sdf_files)} SDF files in {args.root}.")


if __name__ == "__main__":
    main()
