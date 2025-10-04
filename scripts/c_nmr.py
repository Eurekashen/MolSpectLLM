"""Routines for formatting and analysing 13C NMR peak data."""

import re
from typing import Dict, List, Sequence, Tuple

C13Peak = Dict[str, float]


def process_c13_nmr_data(
    c13_data: Sequence[C13Peak],
    frequency: str = "unknown",
    solvent: str = "unknown",
) -> Tuple[str, str]:
    """Convert 13C-NMR peak information into a standard tag and an analysis block."""

    # 1. Build the compact representation.
    standard_rep = f"<13C_NMR>({frequency}, {solvent}) δ "

    shifts = [peak["delta (ppm)"] for peak in c13_data]
    sorted_shifts = sorted(shifts, reverse=True)

    formatted_shifts = [f"{shift:.4f}" for shift in sorted_shifts]
    standard_rep += ", ".join(formatted_shifts)
    standard_rep += "</13C_NMR>"

    # 2. Generate the narrative report.
    analysis_text = "## 13C-NMR Spectrum Analysis\n\n"

    num_peaks = len(c13_data)
    min_shift, max_shift = min(shifts), max(shifts)

    analysis_text += (
        f"The 13C-NMR spectrum contains {num_peaks} distinct signals, "
        f"spanning a chemical shift range from {min_shift:.2f} to {max_shift:.2f} ppm.\n\n"
    )

    analysis_text += "### Chemical Shift Region Analysis:\n"

    regions = [
        {
            "name": "Carbonyl",
            "range": (160, 220),
            "description": "carbonyl carbons (aldehydes, ketones, carboxylic acids, esters, amides)",
        },
        {
            "name": "Aromatic",
            "range": (110, 160),
            "description": "aromatic carbons and heteroaromatic carbons",
        },
        {
            "name": "Olefinic",
            "range": (100, 150),
            "description": "alkene carbons (sp2 hybridised)",
        },
        {
            "name": "Alkynic",
            "range": (65, 90),
            "description": "alkyne carbons",
        },
        {
            "name": "Oxygen-substituted",
            "range": (50, 90),
            "description": "carbons adjacent to oxygen (alcohols, ethers)",
        },
        {
            "name": "Nitrogen-substituted",
            "range": (30, 65),
            "description": "carbons adjacent to nitrogen (amines, amides)",
        },
        {
            "name": "Alkyl",
            "range": (0, 50),
            "description": "alkyl carbons (sp3 hybridised)",
        },
    ]

    region_counts = {region["name"]: 0 for region in regions}

    for shift in shifts:
        for region in regions:
            low, high = region["range"]
            if low <= shift <= high:
                region_counts[region["name"]] += 1
                break

    for region in regions:
        count = region_counts[region["name"]]
        if count > 0:
            analysis_text += (
                f"- **{region['name']} region** (δ {region['range'][0]}-{region['range'][1]} ppm): "
                f"{count} signals. Characteristic of {region['description']}.\n"
            )

    analysis_text += "\n### Characteristic Signal Analysis:\n"

    carbonyl_signals = [shift for shift in shifts if 160 <= shift <= 220]
    aromatic_signals = [shift for shift in shifts if 110 <= shift <= 160]
    alkyl_signals = [shift for shift in shifts if shift < 50]

    if carbonyl_signals:
        carbonyl_str = ", ".join(f"{s:.2f}" for s in sorted(carbonyl_signals, reverse=True))
        analysis_text += (
            f"- Carbonyl signals detected at {carbonyl_str} ppm. These could indicate "
            "ketone, aldehyde, carboxylic acid, ester, or amide functional groups.\n"
        )

    if aromatic_signals:
        aromatic_count = len(aromatic_signals)
        analysis_text += (
            f"- {aromatic_count} aromatic/olefinic signals present, suggesting "
            "an aromatic ring system or conjugated double bonds.\n"
        )

    if alkyl_signals:
        alkyl_count = len(alkyl_signals)
        alkyl_str = ", ".join(f"{s:.2f}" for s in sorted(alkyl_signals, reverse=True))
        analysis_text += (
            f"- Alkyl signals detected at {alkyl_str} ppm, indicating "
            "methyl, methylene, or methine groups.\n"
        )

    intensities = [peak["intensity"] for peak in c13_data]
    max_intensity = max(intensities)
    min_intensity = min(intensities)

    analysis_text += (
        f"\n- Signal intensities range from {min_intensity:.4f} to {max_intensity:.4f}. "
        "Note that in 13C-NMR, signal intensity does not directly correlate with carbon count "
        "due to relaxation and nuclear Overhauser effects.\n"
    )

    analysis_text += "\n### Structural Insights:\n"

    if carbonyl_signals:
        analysis_text += (
            "- Carbonyl signals suggest the presence of carbonyl-containing functional groups. "
        )
        if any(190 <= shift <= 220 for shift in carbonyl_signals):
            analysis_text += "Signals above 190 ppm may indicate aldehydes or ketones.\n"
        elif any(160 <= shift <= 175 for shift in carbonyl_signals):
            analysis_text += "Signals around 160-175 ppm may indicate carboxylic acids, esters, or amides.\n"

    if aromatic_signals:
        aromatic_count = len(aromatic_signals)
        if aromatic_count >= 6:
            analysis_text += (
                "- Multiple aromatic signals suggest a substituted benzene ring or polyaromatic system.\n"
            )
        elif aromatic_count in {4, 5}:
            analysis_text += (
                "- Several aromatic signals may indicate a heteroaromatic system or substituted benzene.\n"
            )

    # All carbons give signals in 13C-NMR, but quaternary carbons often appear weaker.
    quaternary_candidates = [peak for peak in c13_data if peak["intensity"] < (max_intensity * 0.5)]
    if quaternary_candidates:
        quat_shifts = [f"{peak['delta (ppm)']:.2f}" for peak in quaternary_candidates]
        quat_str = ", ".join(quat_shifts)
        analysis_text += (
            f"- Potential quaternary carbons detected at {quat_str} ppm "
            "(based on lower signal intensity).\n"
        )

    analysis_text += (
        "\n### Interpretation Summary:\n"
        "13C-NMR spectroscopy provides detailed information about the carbon skeleton of a molecule. "
        "Chemical shifts reveal the electronic environment of each carbon atom, while the number of signals "
        "indicates the number of chemically distinct carbon environments. For complete structure determination, "
        "correlate these data with complementary experiments such as 1H-NMR, HSQC, HMBC, and mass spectrometry."
    )

    return standard_rep, analysis_text


def parse_c13_nmr_standard(standard_rep: str) -> List[C13Peak]:
    """Reconstruct peak dictionaries from a compact 13C-NMR string representation."""

    if not standard_rep.startswith("<13C_NMR>") or not standard_rep.endswith("</13C_NMR>"):
        raise ValueError("Invalid 13C-NMR standard representation format")

    core_text = standard_rep.replace("<13C_NMR>", "").replace("</13C_NMR>", "").strip()

    if "(" in core_text and ")" in core_text:
        core_text = core_text.split(")", 1)[1].strip()

    shifts_str = core_text.replace("δ", "").strip()
    shift_values = [float(shift) for shift in re.findall(r"[\d.]+", shifts_str)]

    c13_data: List[C13Peak] = []
    for shift in shift_values:
        peak_data: C13Peak = {
            "delta (ppm)": shift,
            "integral": 0.0,
            "intensity": 0.0,
            "width (ppm)": 0.01,
        }
        c13_data.append(peak_data)

    return c13_data


if __name__ == "__main__":
    # Example dataset (truncated 13C-NMR table).
    c13_data = [
        {'delta (ppm)': 162.3939, 'integral': 0.00047, 'intensity': 0.02482, 'width (ppm)': 0.0122},
        {'delta (ppm)': 160.4215, 'integral': 0.00047, 'intensity': 0.02482, 'width (ppm)': 0.0122},
        {'delta (ppm)': 156.2097, 'integral': 0.00048, 'intensity': 0.02493, 'width (ppm)': 0.0122},
        {'delta (ppm)': 154.1781, 'integral': 0.00048, 'intensity': 0.02494, 'width (ppm)': 0.0122},
        {'delta (ppm)': 148.8389, 'integral': 0.00048, 'intensity': 0.02507, 'width (ppm)': 0.0122},
        {'delta (ppm)': 141.9831, 'integral': 0.00048, 'intensity': 0.02543, 'width (ppm)': 0.0122},
        {'delta (ppm)': 130.1334, 'integral': 0.00082, 'intensity': 0.04305, 'width (ppm)': 0.0122},
        {'delta (ppm)': 125.2806, 'integral': 0.00047, 'intensity': 0.02479, 'width (ppm)': 0.0122},
        {'delta (ppm)': 121.4140, 'integral': 0.00098, 'intensity': 0.04409, 'width (ppm)': 0.0155},
        {'delta (ppm)': 114.3332, 'integral': 0.00024, 'intensity': 0.01241, 'width (ppm)': 0.0122},
        {'delta (ppm)': 105.8465, 'integral': 0.00083, 'intensity': 0.04374, 'width (ppm)': 0.0122},
        {'delta (ppm)': 100.3585, 'integral': 0.00081, 'intensity': 0.04235, 'width (ppm)': 0.0122}
    ]
    
    # Convert to standard text and narrative.
    standard_rep, analysis_text = process_c13_nmr_data(
        c13_data, frequency="400 MHz", solvent="CDCl3"
    )
    print("Standard Representation:")
    print(standard_rep)
    print("\nAnalysis Text:")
    print(analysis_text[:500] + "...")  # Display the first 500 characters

    # Parse back to the structured dict format.
    parsed_data = parse_c13_nmr_standard(standard_rep)
    print("\nParsed Data:")
    for i, peak in enumerate(parsed_data, 1):
        print(f"Peak {i}: δ = {peak['delta (ppm)']:.2f} ppm")
