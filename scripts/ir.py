"""Utility functions for extracting IR spectral peaks and annotating functional groups."""

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks

# Mapping between characteristic IR bands and representative functional group assignments.
IR_BAND_ASSIGNMENTS = {
    (3700, 3200): "O-H stretching (alcohols, phenols, carboxylic acids)",
    (3300, 3000): "N-H stretching (amines, amides), ≡C-H stretching",
    (3000, 2850): "C-H stretching (alkanes)",
    (2260, 2220): "C≡N stretching (nitriles)",
    (1760, 1665): "C=O stretching (carbonyls: ketones ~1715, aldehydes ~1730, carboxylic acids ~1720, esters ~1735, amides ~1680)",
    (1650, 1580): "C=C stretching (alkenes, aromatics)",
    (1600, 1450): "C-C stretching (aromatics)",
    (1550, 1500): "N-O stretching (nitro compounds)",
    (1470, 1350): "C-H bending (alkanes)",
    (1300, 1000): "C-O stretching (alcohols, carboxylic acids, esters, ethers)",
    (1000, 650): "=C-H bending (alkenes), C-H bending (aromatics)",
    (950, 910): "O-H bending (carboxylic acids)",
    (850, 550): "C-Cl stretching (alkyl chlorides)"
}

# Common functional group signatures expressed as characteristic wavenumber windows.
FUNCTIONAL_GROUP_SIGNATURES = {
    "Alcohol": [(3200, 3600), (1000, 1200)],
    "Carboxylic Acid": [(2500, 3300), (1700, 1725), (1210, 1320)],
    "Ketone": [(1705, 1720)],
    "Aldehyde": [(1720, 1740), (2700, 2800), (2800, 2900)],
    "Ester": [(1735, 1750), (1000, 1300)],
    "Amide": [(1640, 1690), (3100, 3500)],
    "Amine": [(3300, 3500)],
    "Nitrile": [(2200, 2260)],
    "Alkyne": [(2100, 2260), (3250, 3350)],
    "Aromatic": [(1500, 1600), (1580, 1600), (3000, 3100)],
    "Alkene": [(1620, 1680), (3000, 3100)],
    "Nitro": [(1500, 1600), (1300, 1400)]
}

def extract_peaks_from_spectrum(
    spectrum: Sequence[float],
    freqs: Sequence[float],
    min_height: float = 0.1,
    min_distance: int = 10,
    min_prominence: float = 0.1,
) -> Tuple[List[Tuple[float, float]], int]:
    """Detect significant absorption bands from a spectrum and return sorted peaks."""

    max_intensity = np.max(spectrum)
    abs_min_height = max_intensity * min_height

    peaks, properties = find_peaks(
        spectrum,
        height=abs_min_height,
        distance=min_distance,
        prominence=min_prominence * max_intensity,
    )

    peak_freqs = freqs[peaks]
    peak_heights = properties["peak_heights"]

    return sorted(zip(peak_freqs, peak_heights)), len(peaks)


def assign_functional_groups(
    peaks: Iterable[Tuple[float, float]]
) -> Dict[float, List[str]]:
    """Associate each detected peak with plausible functional group assignments."""

    assignments: Dict[float, List[str]] = {}

    for freq, _ in peaks:
        freq = float(freq)
        possible_groups: List[str] = []

        for band, assignment in IR_BAND_ASSIGNMENTS.items():
            if band[0] <= freq <= band[1]:
                possible_groups.append(assignment)

        if not possible_groups:
            for group, bands in FUNCTIONAL_GROUP_SIGNATURES.items():
                for band in bands:
                    if band[0] <= freq <= band[1]:
                        possible_groups.append(group)
                        break

        assignments[freq] = possible_groups

    return assignments


def generate_ir_analysis(
    peaks: Sequence[Tuple[float, float]],
    num_peaks: int,
) -> str:
    """Produce a narrative IR analysis that highlights major bands and implications."""

    if not peaks:
        return "No significant peaks detected in the IR spectrum."

    strongest_peak = max(peaks, key=lambda x: x[1])
    assignments = assign_functional_groups(peaks)

    functional_groups: Dict[str, int] = {}
    for groups in assignments.values():
        for group in groups:
            functional_groups[group] = functional_groups.get(group, 0) + 1

    sorted_groups = sorted(functional_groups.items(), key=lambda x: x[1], reverse=True)

    analysis = "## Comprehensive Infrared Spectroscopy Analysis\n\n"
    analysis += "### Spectrum Overview\n"
    analysis += f"- **Total peaks**: {num_peaks} significant absorption bands\n"
    analysis += f"- **Strongest absorption**: {strongest_peak[0]:.0f} cm^-1 (intensity: {strongest_peak[1]:.2f})\n"
    analysis += "- **Spectral range**: 500-4000 cm^-1 (mid-infrared region)\n\n"

    analysis += "### Major Absorption Bands and Assignments\n"
    for i, (freq, intensity) in enumerate(peaks, 1):
        freq_groups = assignments.get(float(freq), [])
        group_desc = ", ".join(freq_groups) if freq_groups else "Unassigned band"
        analysis += (
            f"{i}. **{freq:.0f} cm^-1** (Intensity: {intensity:.2f}): {group_desc}\n"
        )

    analysis += "\n### Functional Group Analysis\n"
    if sorted_groups:
        analysis += "The IR spectrum suggests the presence of:\n"
        for group, count in sorted_groups[:5]:
            analysis += f"- **{group}** ({count} characteristic bands)\n"
    else:
        analysis += (
            "No definitive functional group assignments could be made based on the observed bands.\n"
        )

    analysis += "\n### Structural Implications\n"

    group_types = [group for group, _ in sorted_groups]
    if "Carboxylic Acid" in group_types:
        analysis += (
            "- Strong evidence of carboxylic acid functional groups (broad O-H stretch ~3000 cm^-1, "
            "C=O stretch ~1710 cm^-1)\n"
        )
    if "Alcohol" in group_types:
        analysis += (
            "- Presence of hydroxyl groups indicated by broad O-H stretch between 3200-3600 cm^-1 "
            "and C-O stretch around 1100 cm^-1\n"
        )
    if "Aromatic" in group_types:
        analysis += (
            "- Characteristic aromatic bands detected between 1450-1600 cm^-1 and 3000-3100 cm^-1, "
            "suggesting phenyl rings\n"
        )
    if "Amine" in group_types or "Amide" in group_types:
        analysis += (
            "- Nitrogen-containing functional groups detected, likely amines or amides\n"
        )

    if any(1650 <= freq <= 1750 for freq, _ in peaks):
        analysis += (
            "- Prominent carbonyl stretch observed, indicating possible ketone, aldehyde, "
            "carboxylic acid, ester, or amide functionality\n"
        )

    if any(3200 <= freq <= 3550 for freq, _ in peaks) and any(
        1000 <= freq <= 1300 for freq, _ in peaks
    ):
        analysis += (
            "- Combination of broad O-H stretch and C-O stretch suggests alcoholic functionality\n"
        )

    if any(3300 <= freq <= 3500 for freq, _ in peaks) and not any(
        1650 <= freq <= 1750 for freq, _ in peaks
    ):
        analysis += (
            "- N-H stretches without carbonyl absorption may indicate primary or secondary amines\n"
        )

    analysis += "\n### Analytical Summary\n"
    analysis += (
        "This IR spectrum reveals characteristic absorption bands that provide insights into the molecular structure. "
    )

    if sorted_groups:
        primary_groups = ", ".join([group for group, _ in sorted_groups[:3]])
        analysis += f"The most prominent functional groups appear to be {primary_groups}. "

    analysis += "The spectral features are consistent with an organic compound containing heteroatoms. "
    analysis += (
        "For definitive structural assignment, correlation with other spectroscopic data (e.g., NMR, MS) is recommended."
    )

    return analysis


def process_ir_data(row: Sequence[float]) -> Tuple[str, str]:
    """Return the compact IR tag and descriptive analysis for a single spectrum."""

    freqs = np.linspace(500, 4000, 1800)
    spectrum = np.array(row, dtype=float)

    peaks, num_peaks = extract_peaks_from_spectrum(spectrum, freqs)

    peaks_str = "; ".join([f"{freq:.4f}({intensity:.4f})" for freq, intensity in peaks])
    standard_rep = f"<IR>(500~4000) {peaks_str}</IR>"

    text_description = generate_ir_analysis(peaks, num_peaks)

    return standard_rep, text_description
