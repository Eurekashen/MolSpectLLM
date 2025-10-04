#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprehensive accuracy metrics for spectrum-to-SMILES evaluation.

The utilities in this module compute token-level accuracy, sequence accuracy,
fingerprint similarity, structural similarity, MCES overlap, functional-group
agreement, and a Fraggle-style fragment similarity.
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional
import numpy as np

# -------- RDKit availability --------
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdMolDescriptors, rdFMCS, Descriptors, Crippen
    HAS_RDKIT = True
    print("RDKit loaded successfully; structural metrics are available.")
except Exception as e:
    HAS_RDKIT = False
    print(f"RDKit import failed: {e}")
    print("Fingerprint and structural similarities will be skipped.")


def extract_smiles_from_text(text: str) -> str:
    """Extract a candidate SMILES string from free-form text."""
    if not text or text.startswith("Error"):
        return ""

    # Prefer explicit <smiles>...</smiles> tags when present.
    smiles_match = re.search(r'<smiles>(.*?)</smiles>',
                             text, re.IGNORECASE | re.DOTALL)
    if smiles_match:
        extracted = smiles_match.group(1).strip()
        # Ensure the captured string is not another nested tag.
        if extracted and not extracted.startswith('<'):
            return extracted

    # Fallback: collect plausible SMILES tokens (elements, bonds, brackets).
    chemical_pattern = r'[A-Z][a-z]?\d*|[a-z]\d*|[=#@+-]|\(|\)|\[|\]|\.'
    potential_smiles = re.findall(chemical_pattern, text)
    if potential_smiles:
        return ''.join(potential_smiles)

    return ""


def normalize_smiles(smiles: str) -> str:
    """Lightweight SMILES normalisation that strips whitespace and noise."""
    if not smiles:
        return ""

    # Remove extraneous whitespace.
    smiles = re.sub(r'\s+', '', smiles)

    # Drop characters that are unlikely to belong to SMILES strings.
    smiles = re.sub(r'[^\w=#@+\-\(\)\[\]\.]', '', smiles)

    return smiles


def clean_llm_answer(ans: str) -> str:
    """Strip auxiliary formatting from LLM-generated responses."""
    if not ans:
        return ""

    # Remove common tags.
    ans = ans.strip()

    # Drop <think>...</think> annotations if present.
    ans = re.sub(r'<think>.*?</think>', '', ans, flags=re.DOTALL)

    # Remove any remaining XML/HTML-like tags.
    ans = re.sub(r'<[^>]+>', '', ans)

    # Collapse repeated whitespace.
    ans = re.sub(r'\n+', ' ', ans)
    ans = re.sub(r'\s+', ' ', ans)

    return ans.strip()


def normalize_smiles_with_rdkit(smiles_str: str) -> str:
    """Canonicalise SMILES via RDKit while retaining stereochemistry."""
    if not smiles_str:
        return ""
    if not HAS_RDKIT:
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles_str, sanitize=True)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def is_error_ans(ans: str) -> bool:
    """Return ``True`` when the model answer explicitly indicates failure."""
    if ans is None:
        return True
    a = ans.strip().lower()
    if a == "":
        return True
    return a.startswith("error") or "invalid" in a or "failed" in a


# Robust SMILES tokenizer
SMILES_TOKEN_PATTERN = r"""
    \%\d{2}               | # %10, %99 ring indices
    Cl|Br|Si|Se|Na|Li|Mg  | # common two-letter elements (extend if needed)
    [A-Z][a-z]?           | # elements
    [bcnops]              | # aromatic elements (lowercase)
    \[[^\]]+\]            | # bracket atoms
    @@?                   | # @ or @@ stereo
    [#=\-]                | # bond orders
    [+/\\()]              | # branches/directional
    \.                    | # dot (disconnected)
    \d                      # single-digit ring indices
"""
TOKEN_RE = re.compile(SMILES_TOKEN_PATTERN, re.VERBOSE)


def smiles_tokens(s: str) -> List[str]:
    """Split a SMILES string into lexical tokens."""
    return TOKEN_RE.findall(s)


def structural_equal_smiles(a: str, b: str) -> bool:
    """Check structural equivalence after RDKit canonicalisation (stereo-aware)."""
    if not HAS_RDKIT:
        return False
    ca = normalize_smiles_with_rdkit(a)
    cb = normalize_smiles_with_rdkit(b)
    return bool(ca and cb and ca == cb)


def calculate_fingerprint_similarity(smiles1: str, smiles2: str) -> float:
    """Return the Tanimoto similarity between RDKit fingerprints."""
    if not HAS_RDKIT:
        return 0.0

    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return 0.0

        # Compute RDKit bit fingerprints.
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)

        # Report the Tanimoto overlap.
        similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
        return similarity

    except Exception:
        return 0.0


def calculate_torsion_similarity(smiles1: str, smiles2: str) -> float:
    """Compute hashed topological torsion fingerprint similarity."""
    if not HAS_RDKIT:
        return 0.0

    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return 0.0

        tt_fp1 = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
            mol1)
        tt_fp2 = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
            mol2)
        return DataStructs.FingerprintSimilarity(tt_fp1, tt_fp2)

    except Exception:
        return 0.0


def calculate_atom_pair_similarity(smiles1: str, smiles2: str) -> float:
    """Compute atom-pair fingerprint similarity."""
    if not HAS_RDKIT:
        return 0.0

    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return 0.0

        ap_fp1 = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol1)
        ap_fp2 = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol2)
        return DataStructs.FingerprintSimilarity(ap_fp1, ap_fp2)

    except Exception:
        return 0.0


def calculate_mces_similarity(smiles1: str, smiles2: str) -> float:
    """Estimate similarity via the maximum common edge substructure (MCES)."""
    if not HAS_RDKIT:
        return 0.0

    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return 0.0

        # Derive the maximum common substructure.
        mcs = rdFMCS.FindMCS([mol1, mol2], 
                           bondCompare=rdFMCS.BondCompare.CompareOrder,
                           atomCompare=rdFMCS.AtomCompare.CompareElements,
                           timeout=30)
        
        if mcs.smartsString == "":
            return 0.0
            
        # Build the common substructure molecule for evaluation.
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        if mcs_mol is None:
            return 0.0
            
        # Similarity is the ratio between common and maximal edge counts.
        num_bonds_mcs = mcs_mol.GetNumBonds()
        num_bonds_mol1 = mol1.GetNumBonds()
        num_bonds_mol2 = mol2.GetNumBonds()
        
        if max(num_bonds_mol1, num_bonds_mol2) == 0:
            return 1.0 if num_bonds_mcs == 0 else 0.0
            
        return num_bonds_mcs / max(num_bonds_mol1, num_bonds_mol2)

    except Exception:
        return 0.0


def get_functional_groups(mol):
    """Return functional-group labels identified by simple SMARTS patterns."""
    if mol is None:
        return set()
    
    # Functional group SMARTS patterns.
    functional_groups = {
        'alcohol': '[OH]',
        'aldehyde': '[CX3H1](=O)[#6]',
        'ketone': '[#6][CX3](=O)[#6]',
        'carboxylic_acid': '[CX3](=O)[OX2H1]',
        'ester': '[#6][CX3](=O)[OX2H0][#6]',
        'ether': '[OD2]([#6])[#6]',
        'amine_primary': '[NX3;H2,H1;!$(NC=O)]',
        'amine_secondary': '[NX3;H1;!$(NC=O)]',
        'amine_tertiary': '[NX3;H0;!$(NC=O)]',
        'amide': '[NX3][CX3](=[OX1])[#6]',
        'nitrile': '[NX1]#[CX2]',
        'nitro': '[NX3+]([OX1-])[OX1-]',
        'aromatic_ring': 'c1ccccc1',
        'halogen_F': '[F]',
        'halogen_Cl': '[Cl]',
        'halogen_Br': '[Br]',
        'halogen_I': '[I]',
        'sulfonic_acid': '[SX4](=[OX1])(=[OX1])([OX2H,OX1H0-])[#6]',
        'thiol': '[SH]',
        'phenol': 'c[OH]',
    }
    
    found_groups = set()
    for group_name, smarts in functional_groups.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            found_groups.add(group_name)
    
    return found_groups


def calculate_functional_group_similarity(smiles1: str, smiles2: str) -> float:
    """Compute Jaccard similarity between detected functional groups."""
    if not HAS_RDKIT:
        return 0.0

    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return 0.0

        groups1 = get_functional_groups(mol1)
        groups2 = get_functional_groups(mol2)
        
        if len(groups1) == 0 and len(groups2) == 0:
            return 1.0
        
        if len(groups1) == 0 or len(groups2) == 0:
            return 0.0
        
        # Compute the Jaccard similarity coefficient.
        intersection = len(groups1.intersection(groups2))
        union = len(groups1.union(groups2))
        
        return intersection / union if union > 0 else 0.0

    except Exception:
        return 0.0


def calculate_fraggle_similarity(smiles1: str, smiles2: str) -> float:
    """Approximate a Fraggle-style fragment similarity metric."""
    if not HAS_RDKIT:
        return 0.0

    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return 0.0

        # Use bit-vector Morgan fingerprints as a fragment-based approximation.
        morgan_fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        morgan_fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        
        # Compute the Tanimoto similarity.
        similarity = DataStructs.FingerprintSimilarity(morgan_fp1, morgan_fp2)
        
        # Apply a penalty based on relative molecule size.
        size1 = mol1.GetNumAtoms()
        size2 = mol2.GetNumAtoms()
        size_penalty = min(size1, size2) / max(size1, size2) if max(size1, size2) > 0 else 1.0
        
        # Combine similarity with the size penalty.
        fraggle_sim = similarity * (0.7 + 0.3 * size_penalty)
        
        return fraggle_sim

    except Exception:
        return 0.0


def calculate_sequence_accuracy(gt_smiles: str, pred_smiles: str) -> float:
    """Binary sequence-level accuracy (1.0 for an exact match, otherwise 0.0)."""
    if not gt_smiles or not pred_smiles:
        return 0.0

    # Normalise SMILES strings prior to comparison.
    gt_norm = normalize_smiles(gt_smiles)
    pred_norm = normalize_smiles(pred_smiles)

    if not gt_norm or not pred_norm:
        return 0.0

    # Exact lexical match.
    if gt_norm == pred_norm:
        return 1.0

    # fall back to RDKit structural comparison when available.
    if HAS_RDKIT:
        if structural_equal_smiles(gt_norm, pred_norm):
            return 1.0

    # Otherwise treat the prediction as incorrect.
    return 0.0


def calculate_token_accuracy(gt_smiles: str, pred_smiles: str) -> float:
    """Token-level accuracy based on aligned SMILES tokens."""
    if not gt_smiles or not pred_smiles:
        return 0.0

    # Normalise simple formatting differences.
    gt_norm = normalize_smiles(gt_smiles)
    pred_norm = normalize_smiles(pred_smiles)

    if not gt_norm or not pred_norm:
        return 0.0

    # Canonicalise via RDKit if possible.
    if HAS_RDKIT:
        gt_canon = normalize_smiles_with_rdkit(gt_norm) or gt_norm
        pred_canon = normalize_smiles_with_rdkit(pred_norm) or pred_norm
    else:
        gt_canon = gt_norm
        pred_canon = pred_norm

    # Tokenise each sequence.
    gt_tokens = smiles_tokens(gt_canon)
    pred_tokens = smiles_tokens(pred_canon)

    if not gt_tokens or not pred_tokens:
        return 0.0

    # Scale by the reference token count.
    total_tokens = len(gt_tokens)
    correct_tokens = sum(1 for a, b in zip(gt_tokens, pred_tokens) if a == b)

    return correct_tokens / total_tokens if total_tokens > 0 else 0.0


def analyze_file(file_path: str) -> Dict[str, Any]:
    """Analyse a single JSONL result file and return aggregate metrics."""
    print(f"Analysing file: {file_path}")

    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    if not results:
        print(f"Warning: file is empty or malformed: {file_path}")
        return {}

    # Aggregate per-record metrics.
    total = len(results)
    exact_matches = 0
    partial_matches = 0
    errors = 0
    empty_responses = 0

    # Accumulate detailed metric values for averaging.
    seq_accuracies = []
    token_accuracies = []
    fp_similarities = []
    torsion_similarities = []
    atom_pair_similarities = []
    mces_similarities = []
    functional_group_similarities = []
    fraggle_similarities = []

    for result in results:
        # Extract ground-truth and predicted SMILES strings.
        gt_answer = result.get('label', '')
        llm_answer = result.get('predict', '')

        gt_smiles = extract_smiles_from_text(gt_answer)
        pred_smiles = extract_smiles_from_text(llm_answer)

        if not pred_smiles:
            empty_responses += 1
            continue

        if is_error_ans(llm_answer):
            errors += 1
            continue

        # Compute individual metrics.
        seq_acc = calculate_sequence_accuracy(gt_smiles, pred_smiles)
        token_acc = calculate_token_accuracy(gt_smiles, pred_smiles)
        fp_sim = calculate_fingerprint_similarity(gt_smiles, pred_smiles)
        torsion_sim = calculate_torsion_similarity(gt_smiles, pred_smiles)
        atom_pair_sim = calculate_atom_pair_similarity(gt_smiles, pred_smiles)
        mces_sim = calculate_mces_similarity(gt_smiles, pred_smiles)
        fg_sim = calculate_functional_group_similarity(gt_smiles, pred_smiles)
        fraggle_sim = calculate_fraggle_similarity(gt_smiles, pred_smiles)

        seq_accuracies.append(seq_acc)
        token_accuracies.append(token_acc)
        fp_similarities.append(fp_sim)
        torsion_similarities.append(torsion_sim)
        atom_pair_similarities.append(atom_pair_sim)
        mces_similarities.append(mces_sim)
        functional_group_similarities.append(fg_sim)
        fraggle_similarities.append(fraggle_sim)

        if seq_acc == 1.0:
            exact_matches += 1
        elif seq_acc > 0.5:
            partial_matches += 1

    # Compute aggregated statistics.
    avg_seq_acc = np.mean(seq_accuracies) if seq_accuracies else 0.0
    avg_token_acc = np.mean(token_accuracies) if token_accuracies else 0.0
    avg_fp_sim = np.mean(fp_similarities) if fp_similarities else 0.0
    avg_torsion_sim = np.mean(
        torsion_similarities) if torsion_similarities else 0.0
    avg_atom_pair_sim = np.mean(
        atom_pair_similarities) if atom_pair_similarities else 0.0
    avg_mces_sim = np.mean(mces_similarities) if mces_similarities else 0.0
    avg_fg_sim = np.mean(functional_group_similarities) if functional_group_similarities else 0.0
    avg_fraggle_sim = np.mean(fraggle_similarities) if fraggle_similarities else 0.0

    return {
        'file': file_path,
        'total': total,
        'exact_matches': exact_matches,
        'partial_matches': partial_matches,
        'errors': errors,
        'empty_responses': empty_responses,
        'avg_seq_accuracy': avg_seq_acc,
        'avg_token_accuracy': avg_token_acc,
        'avg_fp_similarity': avg_fp_sim,
        'avg_torsion_similarity': avg_torsion_sim,
        'avg_atom_pair_similarity': avg_atom_pair_sim,
        'avg_mces_similarity': avg_mces_sim,
        'avg_functional_group_similarity': avg_fg_sim,
        'avg_fraggle_similarity': avg_fraggle_sim,
        'exact_match_rate': exact_matches / total if total > 0 else 0.0,
        'partial_match_rate': partial_matches / total if total > 0 else 0.0
    }


def generate_summary_table(all_results: List[Dict[str, Any]]) -> None:
    """Print a tabulated comparison of all evaluated models."""

    print(f"\n{'='*200}")
    print("Summary Report – Model Comparison")
    print(f"{'='*200}")

    header = (
        f"{'Model':<25} {'SeqAcc':<10} {'TokenAcc':<10} {'FP_Sim':<10} "
        f"{'Torsion_Sim':<11} {'AtomPair_Sim':<11} {'MCES_Sim':<10} "
        f"{'FG_Sim':<11} {'Fraggle_Sim':<11} {'ExactRate':<10} {'PartialRate':<10}"
    )
    print(header)
    print("-" * 200)

    sorted_results = sorted(all_results, key=lambda x: x['avg_seq_accuracy'], reverse=True)

    for result in sorted_results:
        model_name = Path(result['file']).name.replace('results_', '').replace('.jsonl', '')
        if len(model_name) > 24:
            model_name = model_name[:21] + "..."

        row = (
            f"{model_name:<25} {result['avg_seq_accuracy']:<10.4f} "
            f"{result['avg_token_accuracy']:<10.4f} {result['avg_fp_similarity']:<10.4f} "
            f"{result['avg_torsion_similarity']:<11.4f} {result['avg_atom_pair_similarity']:<11.4f} "
            f"{result['avg_mces_similarity']:<10.4f} {result['avg_functional_group_similarity']:<11.4f} "
            f"{result['avg_fraggle_similarity']:<11.4f} {result['exact_match_rate']:<10.4f} "
            f"{result['partial_match_rate']:<10.4f}"
        )
        print(row)

    print("-" * 200)

    if all_results:
        avg_seq_acc = np.mean([r['avg_seq_accuracy'] for r in all_results])
        avg_token_acc = np.mean([r['avg_token_accuracy'] for r in all_results])
        avg_fp_sim = np.mean([r['avg_fp_similarity'] for r in all_results])
        avg_torsion_sim = np.mean([r['avg_torsion_similarity'] for r in all_results])
        avg_atom_pair_sim = np.mean([r['avg_atom_pair_similarity'] for r in all_results])
        avg_mces_sim = np.mean([r['avg_mces_similarity'] for r in all_results])
        avg_fg_sim = np.mean([r['avg_functional_group_similarity'] for r in all_results])
        avg_fraggle_sim = np.mean([r['avg_fraggle_similarity'] for r in all_results])
        avg_exact_rate = np.mean([r['exact_match_rate'] for r in all_results])
        avg_partial_rate = np.mean([r['partial_match_rate'] for r in all_results])

        avg_row = (
            f"{'Average':<25} {avg_seq_acc:<10.4f} {avg_token_acc:<10.4f} "
            f"{avg_fp_sim:<10.4f} {avg_torsion_sim:<11.4f} {avg_atom_pair_sim:<11.4f} "
            f"{avg_mces_sim:<10.4f} {avg_fg_sim:<11.4f} {avg_fraggle_sim:<11.4f} "
            f"{avg_exact_rate:<10.4f} {avg_partial_rate:<10.4f}"
        )
        print(avg_row)

    print(f"{'='*200}")


def main():
    parser = argparse.ArgumentParser(
        description="Spectrum-to-SMILES comprehensive evaluation utility")
    parser.add_argument(
        "--input", type=str, default="eval_results/bench_spec2smiles/generated_predictions.jsonl",
        help="Single JSONL file to analyse")
    parser.add_argument(
        "--results_dir", type=str, default="/Users/shenshuaike/Documents/Research/TrimoLLM/eval_results/bench_spec2smiles/",
        help="Directory containing JSONL results for aggregation")

    args = parser.parse_args()

    if args.input:
        # Analyse a single file when --input is provided.
        if not Path(args.input).exists():
            print(f"Input file not found: {args.input}")
            return

        result = analyze_file(args.input)
        if result:
            print("\nPer-file metrics:")
            print(f"  File: {result['file']}")
            print(f"  Samples: {result['total']}")
            print(
                f"  Exact matches: {result['exact_matches']} ({result['exact_match_rate']*100:.2f}%)")
            print(
                f"  Partial matches: {result['partial_matches']} ({result['partial_match_rate']*100:.2f}%)")
            print(
                f"  Error responses: {result['errors']} ({result['errors']/result['total']*100:.2f}%)")
            print(
                f"  Empty responses: {result['empty_responses']} ({result['empty_responses']/result['total']*100:.2f}%)")
            print(f"  Avg. sequence accuracy: {result['avg_seq_accuracy']:.4f}")
            print(f"  Avg. token accuracy: {result['avg_token_accuracy']:.4f}")
            print(f"  Avg. FP similarity: {result['avg_fp_similarity']:.4f}")
            print(f"  Avg. torsion similarity: {result['avg_torsion_similarity']:.4f}")
            print(
                f"  Avg. atom-pair similarity: {result['avg_atom_pair_similarity']:.4f}")
            print(f"  Avg. MCES similarity: {result['avg_mces_similarity']:.4f}")
            print(f"  Avg. functional-group similarity: {result['avg_functional_group_similarity']:.4f}")
            print(f"  Avg. Fraggle similarity: {result['avg_fraggle_similarity']:.4f}")

    else:
        # Analyse every JSONL file inside the results directory.
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"Results directory not found: {results_dir}")
            return

        jsonl_files = list(results_dir.glob("*.jsonl"))
        if not jsonl_files:
            print(f"No JSONL files found in {results_dir}")
            return

        print(f"Found {len(jsonl_files)} result file(s) in {results_dir}")

        all_results = []
        for file_path in jsonl_files:
            result = analyze_file(str(file_path))
            if result:
                all_results.append(result)

        if not all_results:
            print("No files could be analysed successfully.")
            return

        print("\nSummary across files")
        print("="*80)

        total_samples = sum(r['total'] for r in all_results)
        total_exact = sum(r['exact_matches'] for r in all_results)
        total_partial = sum(r['partial_matches'] for r in all_results)
        total_errors = sum(r['errors'] for r in all_results)
        total_empty = sum(r['empty_responses'] for r in all_results)

        print("Overall statistics:")
        print(f"  Samples: {total_samples}")
        print(f"  Exact matches: {total_exact} ({total_exact/total_samples*100:.2f}%)")
        print(
            f"  Partial matches: {total_partial} ({total_partial/total_samples*100:.2f}%)")
        print(
            f"  Error responses: {total_errors} ({total_errors/total_samples*100:.2f}%)")
        print(f"  Empty responses: {total_empty} ({total_empty/total_samples*100:.2f}%)")

        print("\nPer-file details:")
        print("-"*80)
        all_results.sort(key=lambda x: x['avg_seq_accuracy'], reverse=True)

        for result in all_results:
            file_label = Path(result['file']).name
            print(f"File {file_label}:")
            print(
                f"  Samples: {result['total']:4d} | "
                f"SeqAcc: {result['avg_seq_accuracy']:6.4f} | "
                f"TokenAcc: {result['avg_token_accuracy']:6.4f} | "
                f"FP_Sim: {result['avg_fp_similarity']:6.4f} | "
                f"Torsion_Sim: {result['avg_torsion_similarity']:6.4f} | "
                f"AtomPair_Sim: {result['avg_atom_pair_similarity']:6.4f}"
            )
            print(
                f"           MCES_Sim: {result['avg_mces_similarity']:6.4f} | "
                f"FG_Sim: {result['avg_functional_group_similarity']:6.4f} | "
                f"Fraggle_Sim: {result['avg_fraggle_similarity']:6.4f}"
            )

        # Produce a comparative table for all models.
        generate_summary_table(all_results)

        # Persist the summary to disk.
        summary_file = results_dir / "comprehensive_accuracy_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Spectrum-to-SMILES summary report\n")
            f.write("="*60 + "\n\n")
            f.write(f"Samples: {total_samples}\n")
            f.write(
                f"Exact matches: {total_exact} ({total_exact/total_samples*100:.2f}%)\n")
            f.write(
                f"Partial matches: {total_partial} ({total_partial/total_samples*100:.2f}%)\n")
            f.write(
                f"Error responses: {total_errors} ({total_errors/total_samples*100:.2f}%)\n")
            f.write(
                f"Empty responses: {total_empty} ({total_empty/total_samples*100:.2f}%)\n\n")

            f.write("Per-file details:\n")
            f.write("-"*40 + "\n")
            for result in all_results:
                f.write(f"{Path(result['file']).name}:\n")
                f.write(
                    f"  Samples: {result['total']} | "
                    f"SeqAcc: {result['avg_seq_accuracy']:.4f} | "
                    f"TokenAcc: {result['avg_token_accuracy']:.4f} | "
                    f"FP_Sim: {result['avg_fp_similarity']:.4f} | "
                    f"Torsion_Sim: {result['avg_torsion_similarity']:.4f} | "
                    f"AtomPair_Sim: {result['avg_atom_pair_similarity']:.4f} | "
                    f"MCES_Sim: {result['avg_mces_similarity']:.4f} | "
                    f"FG_Sim: {result['avg_functional_group_similarity']:.4f} | "
                    f"Fraggle_Sim: {result['avg_fraggle_similarity']:.4f}\n"
                )

        print(f"\nSummary written to: {summary_file}")


if __name__ == "__main__":
    main()
