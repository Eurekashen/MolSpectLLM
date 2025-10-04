# Metrics Toolkit

Utilities under `metrics/` evaluate model outputs for spectrum prediction,
SMILES generation, and 3D structure recovery. The tools are grouped by task:

- `3d_gen/` – Quality checks for SMILES-to-3D conversions (fingerprint
  similarity, clash detection, geometry validation).
- `molecule/` – Text-based accuracy metrics for molecule QA and name
  conversion tasks.
- `spectrum/` – Quantitative comparisons between predicted spectra/structures
  and ground-truth annotations.

## Quick Usage

1. **Fingerprint overlap for 3D predictions**
   ```bash
   python metrics/3d_gen/fingerprint.py path/to/sdf_dir --show-table
   ```

2. **Steric clash statistics**
   ```bash
   python metrics/3d_gen/clash.py path/to/sdf_dir --output clash_stats.csv
   ```

3. **Bond and clash validation**
   ```bash
   python metrics/3d_gen/validation.py path/to/sdf_dir --pattern "*_pred.sdf" --exclude-gt
   ```

4. **Molecule QA accuracy**
   ```bash
   python metrics/molecule/molecule_qa.py path/to/results.jsonl
   ```

5. **Spectrum-to-SMILES evaluation**
   ```bash
   python metrics/spectrum/eval_spec2smiles_updated.py --input path/to/predictions.jsonl
   ```

The spectrum utilities (`eval_ir.py`, `eval_ms.py`, `eval_cnmr.py`) accept JSONL
files containing `<IR>`, `<ms_*>`, `<13C_NMR>`, or `<1H_NMR>` tags and produce
CSV summaries with cosine similarity, matching statistics, and per-sample
breakdowns.

All scripts provide `--help` with the full list of arguments.
