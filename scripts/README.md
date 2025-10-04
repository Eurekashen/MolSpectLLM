# Spectroscopic Utilities (`scripts/`)

This directory groups small utilities that convert raw peak lists into a compact
text representation and a chemically meaningful narrative. Each module follows
the same high-level pattern:

1. Accept data exported from analytical software (lists of peaks or spectra).
2. Produce a tag-style, machine-readable summary string.
3. Generate an English report that highlights key structural information.

The sections below summarise every module and demonstrate typical use.

---

## `positive_ms.py`

**Purpose**: Analyse positive-mode MS/MS peak lists collected at 10, 20, or 40 eV
collision energies, returning both an XML-like summary tag and an extended
written interpretation.

**Key functions**:

- `process_msms_10ev(peaks)`, `process_msms_20ev(peaks)`, `process_msms_40ev(peaks)`
  – Convenience wrappers that call the shared internal pipeline with the
  corresponding collision energy label.

**Input format**: Any sequence of peaks where each peak provides `(m/z, intensity)`.
Values may be Python lists, tuples, or NumPy arrays. Intensities are assumed to
be scaled to percentage values, as commonly exported by vendor software.

**Output**:

1. Compact string such as `<ms_positive_10ev>411.01088:24, 429.02145:100</ms_positive_10ev>`.
2. Markdown-formatted analysis covering base peaks, neutral losses, and
   energy-dependent fragmentation behaviour.

**Typical usage**:

```python
from scripts.positive_ms import process_msms_40ev

# Peaks can originate from mzML/mgf parsing; here we hard-code two values
peaks = [
    (314.99334, 100.0),
    (339.04390, 46.97),
    (429.02145, 14.51),
]

standard_tag, report = process_msms_40ev(peaks)
print(standard_tag)
print(report[:300])  # Preview the narrative
```

---

## `c_nmr.py`

**Purpose**: Format and interpret 13C-NMR peak tables. The helper returns a
compact tag listing every chemical shift along with a descriptive report that
categorises peaks into structural regions.

**Key functions**:

- `process_c13_nmr_data(peaks, frequency="unknown", solvent="unknown")`
  – Build the `<13C_NMR>` tag string plus a markdown report.
- `parse_c13_nmr_standard(tag)` – Convert the compact representation back into a
  list of peak dictionaries (intensity/integral values are placeholders because
  the abbreviated form does not retain them).

**Input format**: A sequence of dictionaries produced by NMR processing
software. Each dictionary should contain the keys `"delta (ppm)"`, `"intensity"`,
`"integral"`, and `"width (ppm)"`.

**Output**: Tuple `(standard_tag, report_text)` for the formatter; list of peak
dictionaries for the parser.

**Typical usage**:

```python
from scripts.c_nmr import process_c13_nmr_data

peaks = [
    {"delta (ppm)": 162.39, "intensity": 0.0248, "integral": 0.00047, "width (ppm)": 0.0122},
    {"delta (ppm)": 130.13, "intensity": 0.0430, "integral": 0.00082, "width (ppm)": 0.0122},
]

standard_tag, report = process_c13_nmr_data(peaks, frequency="400 MHz", solvent="CDCl3")
```

---

## `h_nmr.py`

**Purpose**: Perform the same tag-plus-report transformation for 1H-NMR peak
lists, including interpretation of multiplicity, coupling constants, and proton
counts.

**Key functions**:

- `process_hnmr_data(peaks, frequency="unknown", solvent="unknown")`
  – Format `<1H_NMR>` tags and produce a narrative analysis.
- `parse_hnmr_standard(tag)` – Reconstruct peak dictionaries from the compact
  representation.

**Input format**: Sequence of peak dictionaries. Expected keys include
`"category"` (multiplicity code, e.g. `'d'` or `'m'`), `"centroid"`, `"nH"`, and
optional `"j_values"` strings where coupling constants are separated by
underscores, e.g. `'7.2_1.5_'`.

**Output**: Tuple `(standard_tag, report_text)` or list of peak dictionaries.

**Typical usage**:

```python
from scripts.h_nmr import process_hnmr_data

peaks = [
    {"category": "d", "centroid": 7.45, "nH": 1, "j_values": "4.3_"},
    {"category": "m", "centroid": 7.06, "nH": 3, "j_values": None},
]

standard_tag, report = process_hnmr_data(peaks, frequency="400 MHz", solvent="CDCl3")
```

---

## `ir.py`

**Purpose**: Detect peaks from a raw infrared spectrum and propose functional
group assignments. The module builds an `<IR>` tag summarising peak positions and
writes a markdown report detailing possible structural motifs.

**Key functions**:

- `extract_peaks_from_spectrum(spectrum, freqs, ...)` – Find peak positions
  using SciPy's `find_peaks`, returning sorted `(wavenumber, intensity)` pairs.
- `assign_functional_groups(peaks)` – Map each peak to likely functional groups
  based on tabulated wavenumber windows.
- `generate_ir_analysis(peaks, num_peaks)` – Produce the detailed report used in
  downstream workflows.
- `process_ir_data(row)` – High-level helper that combines all steps for a single
  spectrum row, returning `(standard_tag, report_text)`.

**Input format**: For `process_ir_data`, provide a one-dimensional iterable of
intensities ordered from 500 to 4000 cm^-1. The helper internally constructs the
wavenumber axis using `numpy.linspace`.

**Output**: Compact `<IR>` string and markdown report.

**Typical usage**:

```python
from scripts.ir import process_ir_data
import numpy as np

# Synthetic spectrum with two Gaussian absorptions
freqs = np.linspace(500, 4000, 1800)
spectrum = np.exp(-0.5 * ((freqs - 1710) / 20) ** 2) + 0.6 * np.exp(-0.5 * ((freqs - 3300) / 40) ** 2)

standard_tag, report = process_ir_data(spectrum)
```

---

## Testing and Validation Tips

- Each module contains a `__main__` block with small smoke tests you can run via
  `python scripts/<module>.py`.
- To integrate the formatters into data pipelines, ensure your intensity values
  are scaled consistently (percentages for MS/MS and NMR, arbitrary units for IR).
- The generated reports are Markdown strings suitable for direct rendering in
  notebooks, documentation, or LLM prompts.

