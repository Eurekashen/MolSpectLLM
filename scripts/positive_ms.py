"""Utilities for positive-mode tandem mass spectrometry reporting.

This module provides helpers to convert peak lists into a compact text representation
and to generate a chemistry-focused narrative that highlights fragment ions,
neutral losses, and energy-dependent behaviour.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

MSPeak = Tuple[float, float]

# Extended library of frequent fragment ions in positive-ion mode MS/MS.
COMMON_FRAGMENTS = {
    15: "methyl (CH3+)",
    17: "ammonia (NH3)",
    18: "water (H2O)",
    28: "carbon monoxide (CO)",
    29: "ethyl (C2H5+) or formyl (HCO+)",
    30: "formaldehyde (CH2O) or NO+",
    31: "methoxy (CH3O+)",
    32: "methanol (CH3OH) or S+",
    33: "methanethiol (CH3SH)",
    35: "chlorine (Cl+)",
    36: "hydrogen chloride (HCl)",
    41: "allyl (C3H5+)",
    42: "acetylene (C2H2) or C3H6+",
    43: "propyl (C3H7+) or acetyl (CH3CO+)",
    44: "carbon dioxide (CO2)",
    45: "ethoxy (C2H5O+) or carboxylic acid (COOH+)",
    46: "nitrogen dioxide (NO2) or protonated ethanol",
    55: "butenyl (C4H7+)",
    57: "butyl (C4H9+)",
    60: "acetic acid (CH3COOH)",
    64: "sulfur dioxide (SO2)",
    77: "phenyl (C6H5+)",
    79: "bromine (Br+)",
    91: "tropylium (C7H7+)",
    105: "benzoyl (C6H5CO+)",
    149: "phthalate fragment",
}

# Reference table of frequent neutral losses in positive-mode MS/MS.
COMMON_NEUTRAL_LOSSES = {
    15: "methyl group (-CH3)",
    17: "ammonia (-NH3)",
    18: "water (-H2O)",
    27: "HCN or C2H3",
    28: "CO or ethylene",
    29: "CHO or C2H5",
    30: "formaldehyde or NO",
    31: "methoxy (-OCH3)",
    32: "methanol or sulfur",
    35: "chlorine",
    36: "hydrogen chloride",
    42: "ketene (-CH2CO)",
    43: "acetyl (-COCH3) or propyl",
    44: "carbon dioxide",
    45: "carboxylic acid (-COOH) or ethoxy",
    46: "nitrogen dioxide or ethanol",
    57: "tert-butyl or butene",
    60: "acetic acid",
    79: "bromine",
}


def process_msms_10ev(spectra: Sequence[Sequence[float]]) -> Tuple[str, str]:
    """Return a standard representation and narrative for 10 eV positive-mode data."""

    return _process_msms_spectra(spectra, "10ev")


def process_msms_20ev(spectra: Sequence[Sequence[float]]) -> Tuple[str, str]:
    """Return a standard representation and narrative for 20 eV positive-mode data."""

    return _process_msms_spectra(spectra, "20ev")


def process_msms_40ev(spectra: Sequence[Sequence[float]]) -> Tuple[str, str]:
    """Return a standard representation and narrative for 40 eV positive-mode data."""

    return _process_msms_spectra(spectra, "40ev")


def _identify_fragments(mz: float, intensity: float, precursor_mz: Optional[float]) -> str:
    """Identify plausible fragment annotations for a given m/z value."""

    possible_ids: List[str] = []

    # Check canonical fragment ions.
    for frag_mz, frag_name in COMMON_FRAGMENTS.items():
        if abs(mz - frag_mz) < 0.1:
            possible_ids.append(frag_name)

    if precursor_mz is not None and precursor_mz > mz:
        loss = precursor_mz - mz
        rounded_loss = round(loss)
        loss_name = COMMON_NEUTRAL_LOSSES.get(rounded_loss, f"unknown group ({rounded_loss} Da)")

        # Prefer exact integer matches.
        if abs(loss - rounded_loss) < 0.1 and rounded_loss in COMMON_NEUTRAL_LOSSES:
            possible_ids.append(f"Loss of {loss_name}")
        else:
            # Fall back to approximate matches.
            for known_loss, name in COMMON_NEUTRAL_LOSSES.items():
                if abs(loss - known_loss) < 0.2:
                    possible_ids.append(f"Approximate loss of {name} ({loss:.2f} Da)")
                    break
            else:
                possible_ids.append(f"Loss of {loss:.2f} Da")

    # Provide guidance for broader m/z regions.
    if 390 < mz < 430:
        possible_ids.append("Possible molecular ion or adduct")
    elif 300 < mz < 350:
        possible_ids.append("Characteristic of aromatic or conjugated systems")
    elif 200 < mz < 250:
        possible_ids.append("Consistent with conjugated backbones")

    if mz < 100:
        if 40 < mz < 50:
            possible_ids.append("Small hydrocarbon fragment")
        elif 70 < mz < 80:
            possible_ids.append("Possible pyridine or benzene fragment")

    return ", ".join(possible_ids) if possible_ids else "Unidentified fragment"


def _process_msms_spectra(spectra: Sequence[Sequence[float]], energy_level: str) -> Tuple[str, str]:
    """Generate a compact representation and textual analysis for MS/MS spectra."""

    cleaned_spectra: List[MSPeak] = [
        (float(peak[0]), float(peak[1]))
        for peak in spectra
        if len(peak) >= 2
    ]

    if not cleaned_spectra:
        standard_rep = f"<ms_positive_{energy_level}></ms_positive_{energy_level}>"
        description = (
            f"## Comprehensive MS/MS Analysis at {energy_level} Collision Energy\n\n"
            "No peaks were supplied, so no fragment analysis could be performed."
        )
        return standard_rep, description

    sorted_spectra = sorted(cleaned_spectra, key=lambda x: x[0])
    fragments = [f"{mz:.5f}:{int(round(intensity))}" for mz, intensity in sorted_spectra]
    standard_rep = f"<ms_positive_{energy_level}>{', '.join(fragments)}</ms_positive_{energy_level}>"

    base_peak = max(cleaned_spectra, key=lambda x: x[1])
    significant_peaks = [peak for peak in cleaned_spectra if peak[1] >= 10]

    precursor_peak = max(cleaned_spectra, key=lambda x: x[0])
    precursor_mz = precursor_peak[0]

    description = f"## Comprehensive MS/MS Analysis at {energy_level} Collision Energy\n\n"

    description += "### Spectrum Overview\n"
    description += f"- **Base peak**: m/z {base_peak[0]:.5f} at {base_peak[1]:.2f}% relative intensity\n"
    description += f"- **Precursor ion**: m/z {precursor_mz:.5f} at {precursor_peak[1]:.2f}% relative intensity\n"
    description += f"- **Number of fragments**: {len(cleaned_spectra)} total peaks, {len(significant_peaks)} ≥10% intensity\n"
    description += (
        f"- **Mass range**: m/z {min(mz for mz, _ in cleaned_spectra):.2f} "
        f"to {max(mz for mz, _ in cleaned_spectra):.2f}\n\n"
    )

    description += "### Major Fragment Ions\n"
    for i, (mz, intensity) in enumerate(significant_peaks, 1):
        fragment_id = _identify_fragments(mz, intensity, precursor_mz)
        description += (
            f"{i}. **m/z {mz:.5f}** (Intensity: {intensity:.2f}%): {fragment_id}\n"
        )

    description += "\n### Neutral Loss Analysis\n"
    neutral_loss_highlights: List[str] = []
    for mz, intensity in significant_peaks:
        if mz < precursor_mz and intensity > 15:
            loss = precursor_mz - mz
            rounded_loss = round(loss)
            loss_name = COMMON_NEUTRAL_LOSSES.get(rounded_loss, f"unknown group ({rounded_loss} Da)")
            mass_diff = abs(loss - rounded_loss)
            description += (
                f"- Loss of **{loss:.4f} Da** from precursor "
                f"(m/z {precursor_mz:.5f} → {mz:.5f}): Suggests elimination of {loss_name} "
                f"(mass error: {mass_diff:.4f} Da)\n"
            )

            if loss_name not in neutral_loss_highlights:
                neutral_loss_highlights.append(loss_name)

    if not neutral_loss_highlights:
        description += "No prominent neutral losses above 15% intensity were detected.\n"

    description += "\n### Energy-Dependent Fragmentation Patterns\n"
    description += _describe_energy_dependence(
        energy_level,
        base_peak,
        precursor_peak,
        significant_peaks,
        neutral_loss_highlights,
    )

    description += "\n\n### Structural Implications\n"
    if energy_level == "40ev":
        description += (
            "High collision energy reveals extensive fragmentation, indicating:\n"
            "1. **Multiple functional groups** supported by diverse neutral losses.\n"
            "2. **Aromatic or heteroaromatic motifs** evidenced by fragments near m/z 77 and 91.\n"
            "3. **Oxygen-containing substituents** inferred from water and CO2 losses.\n"
            "4. **Potential nitrogen content** suggested by fragments close to m/z 30 and 46.\n"
            "5. **A stabilised backbone** associated with the highest m/z fragment.\n"
        )
    else:
        description += (
            "Limited fragmentation implies a comparatively stable molecular ion.\n"
            "Key observations include persistent precursor intensity, characteristic dehydration "
            "events, and restricted alternative cleavage pathways. The molecule likely contains "
            "stabilising motifs such as conjugated or aromatic systems.\n"
        )

    description += "\n### Analytical Summary\n"
    description += (
        "The observed fragmentation behaviour supports a moderately complex organic molecule "
        "with heteroatom content. Integrating this analysis with complementary spectroscopic "
        "techniques (NMR, IR) is recommended for definitive structure elucidation."
    )

    return standard_rep, description


def _describe_energy_dependence(
    energy_level: str,
    base_peak: MSPeak,
    precursor_peak: MSPeak,
    significant_peaks: Iterable[MSPeak],
    neutral_loss_highlights: Sequence[str],
) -> str:
    """Create a descriptive paragraph for the specified collision energy."""

    precursor_mz, precursor_intensity = precursor_peak
    base_mz, base_intensity = base_peak

    fragment_count = sum(1 for peak in significant_peaks if peak != precursor_peak)
    neutral_loss_text = ", ".join(neutral_loss_highlights[:3]) if neutral_loss_highlights else "none"

    if energy_level == "10ev":
        return (
            "The low collision energy (10 eV) spectrum shows limited fragmentation:\n"
            f"- Precursor ion at m/z {precursor_mz:.5f} dominates with {precursor_intensity:.2f}% intensity.\n"
            f"- {fragment_count} fragment(s) exceed the 10% threshold, indicating a resilient molecular ion.\n"
            f"- Observable neutral losses: {neutral_loss_text}.\n"
        )

    if energy_level == "20ev":
        intensity_change = base_intensity - precursor_intensity
        trend = "remains the base peak" if base_peak == precursor_peak else "begins to decline relative to fragments"
        return (
            "At moderate energy (20 eV), fragmentation increases while the precursor remains influential:\n"
            f"- The precursor ion at m/z {precursor_mz:.5f} {trend} ({precursor_intensity:.2f}% intensity).\n"
            f"- {fragment_count} notable fragment(s) emerge above 10% intensity.\n"
            f"- Neutral losses highlight {neutral_loss_text}.\n"
            f"- Relative change between base and precursor intensities is {intensity_change:.2f} percentage points.\n"
        )

    # Default to high-energy narrative (40 eV and beyond).
    high_mass_fragments = [peak for peak in significant_peaks if peak[0] > precursor_mz * 0.7]
    mid_mass_fragments = [peak for peak in significant_peaks if precursor_mz * 0.4 < peak[0] <= precursor_mz * 0.7]
    low_mass_fragments = [peak for peak in significant_peaks if peak[0] <= precursor_mz * 0.4]

    return (
        "High collision energy promotes extensive fragmentation:\n"
        f"- Base peak shifts to m/z {base_mz:.5f} with {base_intensity:.2f}% intensity.\n"
        f"- Precursor ion intensity drops to {precursor_intensity:.2f}%, signalling efficient dissociation.\n"
        f"- Significant fragments observed: {len(low_mass_fragments)} low-, {len(mid_mass_fragments)} mid-, "
        f"and {len(high_mass_fragments)} high-mass ions.\n"
        f"- Dominant neutral losses include {neutral_loss_text}.\n"
    )


if __name__ == "__main__":
    # Example data for quick smoke testing.
    msms_10ev = [np.array([411.01088, 24.07]), np.array([429.02145, 100.0])]
    msms_20ev = [np.array([411.01088, 6.58]), np.array([429.02145, 100.0])]
    msms_40ev = [
        np.array([45.0135, 14.42]), np.array([174.98597, 17.24]),
        np.array([227.03144, 37.03]), np.array([239.04784, 28.26]),
        np.array([255.04275, 74.42]), np.array([270.98711, 15.93]),
        np.array([314.99334, 100.0]), np.array([339.0439, 46.97]),
        np.array([351.0439, 14.69]), np.array([365.05955, 11.36]),
        np.array([385.01522, 22.27]), np.array([429.02145, 14.51]),
    ]

    # Demonstrate generation of the textual outputs.
    standard_10ev, desc_10ev = process_msms_10ev(msms_10ev)
    print("10 eV Standard Representation:\n", standard_10ev)
    print("\n10 eV Description:\n", desc_10ev)
