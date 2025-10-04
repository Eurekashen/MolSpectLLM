"""Helpers for formatting and interpreting 1H NMR spectral peak lists."""

import re
from typing import Dict, List, Sequence, Tuple

HNMRPeak = Dict[str, object]


def process_hnmr_data(
    hnmr_data: Sequence[HNMRPeak],
    frequency: str = "unknown",
    solvent: str = "unknown",
) -> Tuple[str, str]:
    """Generate the compact string representation and descriptive analysis for 1H-NMR."""

    standard_rep = f"<1H_NMR>({frequency}, {solvent}) "

    for peak in hnmr_data:
        centroid = float(peak["centroid"])
        nH = int(peak["nH"])
        category = str(peak["category"])

        # Pass through the shape code; reserved for potential expansion.
        shape_map = {
            "s": "s",
            "d": "d",
            "t": "t",
            "q": "q",
            "m": "m",
            "dd": "dd",
            "dt": "dt",
            "td": "td",
            "dq": "dq",
            "tt": "tt",
            "br": "br",
        }
        shape = shape_map.get(category, category)

        j_str = ""
        if peak.get("j_values"):
            j_values = str(peak["j_values"]).rstrip("_").split("_")
            j_formatted = ", ".join(j_values)
            j_str = f", J= {j_formatted} Hz"

        standard_rep += f"δ {centroid:.2f} ({shape}{j_str}, {nH}H), "

    standard_rep = standard_rep.rstrip(", ") + "</1H_NMR>"

    analysis_text = "## 1H-NMR Spectrum Analysis\n\n"

    num_peaks = len(hnmr_data)
    shifts = [float(peak["centroid"]) for peak in hnmr_data]
    min_shift, max_shift = min(shifts), max(shifts)
    total_protons = sum(int(peak["nH"]) for peak in hnmr_data)

    analysis_text += (
        f"The 1H-NMR spectrum contains {num_peaks} distinct signals, "
        f"spanning a chemical shift range from {min_shift:.2f} to {max_shift:.2f} ppm. "
        f"The total integration corresponds to {total_protons} protons.\n\n"
    )

    analysis_text += "### Signal-by-Signal Analysis:\n"
    for i, peak in enumerate(hnmr_data, 1):
        centroid = float(peak["centroid"])
        nH = int(peak["nH"])
        category = str(peak["category"])
        j_values = peak.get("j_values")

        shape_desc = {
            "s": "singlet (no adjacent protons)",
            "d": "doublet (coupled to one proton)",
            "t": "triplet (coupled to two equivalent protons)",
            "q": "quartet (coupled to three equivalent protons)",
            "m": "multiplet (complex coupling pattern)",
            "dd": "doublet of doublets (coupled to two different protons)",
            "dt": "doublet of triplets",
            "td": "triplet of doublets",
            "br": "broad singlet (often due to exchangeable protons)",
        }.get(category, f"complex pattern ({category})")

        shift_desc = ""
        if 0.5 <= centroid <= 3.0:
            shift_desc = "typical of aliphatic protons (alkyl chains, methyl groups)"
        elif 3.0 <= centroid <= 4.5:
            shift_desc = "characteristic of protons adjacent to oxygen (alcohols, ethers)"
        elif 4.5 <= centroid <= 6.0:
            shift_desc = "indicative of vinylic protons (alkenes, allylic systems)"
        elif 6.0 <= centroid <= 8.5:
            shift_desc = "aromatic region (benzene derivatives, heterocycles)"
        elif 9.0 <= centroid <= 10.5:
            shift_desc = "aldehyde protons"
        elif 10.5 <= centroid <= 12.0:
            shift_desc = "carboxylic acid protons"

        j_desc = ""
        if j_values:
            j_values_list = str(j_values).rstrip("_").split("_")
            j_desc = f" with coupling constants of {', '.join(j_values_list)} Hz"

        nh_desc = {
            1: "a methine proton (CH)",
            2: "a methylene group (CH2)",
            3: "a methyl group (CH3)",
        }.get(nH, f"{nH} equivalent protons")

        peak_text = (
            f"- **Signal {i}**: δ = {centroid:.2f} ppm, {shape_desc}{j_desc}. "
            f"Integrates for {nH}H, suggesting {nh_desc}. "
            f"This chemical shift is {shift_desc}.\n"
        )

        analysis_text += peak_text

    analysis_text += "\n### Chemical Shift Region Analysis:\n"

    aromatic_count = sum(1 for p in hnmr_data if 6.0 <= float(p["centroid"]) <= 8.5)
    aliphatic_count = sum(1 for p in hnmr_data if 0.5 <= float(p["centroid"]) <= 3.0)
    oxygenated_count = sum(1 for p in hnmr_data if 3.0 <= float(p["centroid"]) <= 4.5)
    vinylic_count = sum(1 for p in hnmr_data if 4.5 <= float(p["centroid"]) <= 6.0)
    aldehyde_count = sum(1 for p in hnmr_data if 9.0 <= float(p["centroid"]) <= 10.5)

    analysis_text += (
        f"- **Aromatic region** (δ 6.0-8.5 ppm): {aromatic_count} signals. "
        "These typically correspond to protons on benzene rings or heterocyclic aromatic systems.\n"
        f"- **Aliphatic region** (δ 0.5-3.0 ppm): {aliphatic_count} signals. "
        "Characteristic of alkyl groups such as methyl, methylene, and methine protons.\n"
        f"- **Oxygen-substituted region** (δ 3.0-4.5 ppm): {oxygenated_count} signals. "
        "Suggests protons adjacent to oxygen atoms in alcohols, ethers, or esters.\n"
        f"- **Vinylic region** (δ 4.5-6.0 ppm): {vinylic_count} signals. "
        "Indicates protons attached to sp2 hybridised carbons in alkenes.\n"
    )

    if aldehyde_count > 0:
        analysis_text += (
            f"- **Aldehyde region** (δ 9.0-10.5 ppm): {aldehyde_count} signal(s). "
            "Characteristic of formyl protons in aldehydes.\n"
        )

    analysis_text += "\n### Structural Insights:\n"

    methyl_groups = sum(1 for p in hnmr_data if int(p["nH"]) == 3)
    if methyl_groups > 0:
        analysis_text += (
            f"- Presence of {methyl_groups} methyl group(s) (CH3), indicated by 3H signals. "
            "These could be terminal alkyl groups or attached to heteroatoms.\n"
        )

    methylene_groups = sum(1 for p in hnmr_data if int(p["nH"]) == 2)
    if methylene_groups > 0:
        analysis_text += (
            f"- {methylene_groups} methylene group(s) (CH2) detected. "
            "These often appear in chains, rings, or as bridging groups between functional groups.\n"
        )

    if aromatic_count > 0:
        analysis_text += (
            f"- Aromatic protons present ({aromatic_count} signals), suggesting benzene or heteroaromatic rings. "
            "The coupling patterns can provide information about substitution patterns.\n"
        )

    broad_signals = sum(1 for p in hnmr_data if str(p["category"]) == "br")
    if broad_signals > 0:
        analysis_text += (
            f"- Broad signals detected ({broad_signals}), which may indicate exchangeable protons "
            "such as OH (alcohols, phenols), NH (amines, amides), or COOH (carboxylic acids).\n"
        )

    analysis_text += (
        "\n### Interpretation Summary:\n"
        "1H-NMR spectroscopy provides detailed information about the chemical environment of protons in a molecule. "
        "Chemical shifts reveal the electronic environment, coupling patterns show connectivity to adjacent protons, "
        "and integration values indicate the number of equivalent protons. For complete structure determination, "
        "correlate these findings with 13C-NMR, COSY, HSQC, and complementary spectroscopic techniques."
    )

    return standard_rep, analysis_text


def parse_hnmr_standard(standard_rep: str) -> List[HNMRPeak]:
    """Parse the compact string back into structured peak dictionaries."""

    if not standard_rep.startswith("<1H_NMR>") or not standard_rep.endswith("</1H_NMR>"):
        raise ValueError("Invalid 1H-NMR standard representation format")

    core_text = standard_rep.replace("<1H_NMR>", "").replace("</1H_NMR>", "").strip()

    if "(" in core_text and ")" in core_text:
        core_text = core_text.split(")", 1)[1].strip()

    pattern = r"δ\s*([\d.]+)\s*\(([a-zA-Z]+)(?:,\s*J=\s*([\d.\s,Hz]+))?,\s*(\d+)H\)"
    matches = re.findall(pattern, standard_rep)

    hnmr_data: List[HNMRPeak] = []
    for match in matches:
        centroid = float(match[0])
        category = match[1].lower()
        j_values = match[2].replace("Hz", "").strip() if match[2] else None
        nH = int(match[3])

        shape_map = {
            "s": "s",
            "d": "d",
            "t": "t",
            "q": "q",
            "m": "m",
            "dd": "dd",
            "dt": "dt",
            "td": "td",
            "dq": "dq",
            "tt": "tt",
            "br": "br",
        }
        reverse_shape_map = {v: k for k, v in shape_map.items()}
        category_abbr = reverse_shape_map.get(category, category)

        if j_values:
            j_values = j_values.replace(" ", "").replace(",", "_") + "_"

        peak_data: HNMRPeak = {
            "category": category_abbr,
            "centroid": centroid,
            "j_values": j_values,
            "nH": float(nH),
            "delta": centroid,
            "rangeMin": centroid - 0.05,
            "rangeMax": centroid + 0.05,
        }

        hnmr_data.append(peak_data)

    return hnmr_data


if __name__ == "__main__":
    # Example 1H-NMR dataset.
    hnmr_data = [
        {'category': 'dt', 'centroid': 7.522086492824546, 'delta': 7.522243962013214, 
         'j_values': '4.4661865234375_7.798828125_', 'nH': 2, 
         'rangeMax': 7.556898795087911, 'rangeMin': 7.487564376867245},
        {'category': 'd', 'centroid': 7.455149151109405, 'delta': 7.455161993700052, 
         'j_values': '4.324462890625_', 'nH': 1, 
         'rangeMax': 7.477480819807289, 'rangeMin': 7.432843167592817},
        {'category': 's', 'centroid': 7.15130955401033, 'delta': 7.151309554010332, 
         'j_values': None, 'nH': 1, 
         'rangeMax': 7.1693049472083885, 'rangeMin': 7.133314160812276},
        {'category': 'm', 'centroid': 7.057730479974367, 'delta': 7.056546574535754, 
         'j_values': None, 'nH': 3, 
         'rangeMax': 7.094927997803475, 'rangeMin': 7.01558316456887},
        {'category': 'td', 'centroid': 6.944663507357883, 'delta': 6.944397458167322, 
         'j_values': '2.740478515625_12.0855712890625_', 'nH': 2, 
         'rangeMax': 6.9892782332245815, 'rangeMin': 6.8995099885201565}
    ]
    
    # Convert to standard representation and descriptive text.
    standard_rep, analysis_text = process_hnmr_data(hnmr_data, frequency="400 MHz", solvent="CDCl3")
    print("Standard Representation:")
    print(standard_rep)
    print("\nAnalysis Text:")
    print(analysis_text[:500] + "...")  # Display the first 500 characters
    
    # Parse back into the peak dictionaries.
    parsed_data = parse_hnmr_standard(standard_rep)
    print("\nParsed Data:")
    for peak in parsed_data:
        print(peak)
