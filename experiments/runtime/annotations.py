"""Coarse cell types of author labels, and the donor identifiers that mean "unknown", for the
analyses that pool studies."""

MISSING_DONORS = {"", "unknown", "nan", "none", "na", "not applicable"}
LADDER_LABEL_CORRECTIONS = {"mast cell": "Granulocytes", "malignant cell": ""}


def harmonize_cell_type(tissue, label):
    text = label.lower()
    if text in {
        "",
        "unknown",
        "nan",
        "none",
        "unassigned",
        "unannotated",
        "blood cell",
        "cell",
        "lymphocyte",
        "kidney cell",
        "kidney epithelial cell",
        "progenitor cell",
    }:
        return None
    if tissue == "lung":
        return label
    if tissue == "blood":
        if any(value in text for value in ("progenitor", "stem cell", "precursor")):
            return "Progenitors"
        if "plasma cell" in text or "plasmablast" in text:
            return "Plasma cells"
        if "b cell" in text:
            return "B cells"
        if "t cell" in text or "thymocyte" in text or "t regulatory" in text:
            return "T cells"
        if "natural killer" in text or "innate lymphoid" in text:
            return "NK and ILC"
        if "monocyte" in text or "macrophage" in text:
            return "Monocytes"
        if "dendritic" in text:
            return "Dendritic cells"
        if "platelet" in text or "megakaryo" in text:
            return "Platelets"
        if "erythro" in text or "reticulo" in text:
            return "Erythroid"
        if any(
            value in text
            for value in ("neutrophil", "granulocyte", "eosinophil", "basophil", "mast cell")
        ):
            return "Granulocytes"
    if tissue == "kidney":
        if any(value in text for value in ("neural cell", "neuron", "schwann")):
            return "Neural cells"
        if "urothelial" in text:
            return "Urothelial cells"
        if "proximal" in text and "tubul" in text:
            return "Proximal tubule"
        if "thin ascending" in text:
            return "Thin ascending limb"
        if "thin descending" in text:
            return "Thin descending limb"
        if "ascending limb" in text:
            return "Thick ascending limb"
        if "distal convoluted" in text:
            return "Distal convoluted tubule"
        if "connecting" in text:
            return "Connecting tubule"
        if "intercalated" in text:
            return "Intercalated cells"
        if "principal" in text:
            return "Principal cells"
        if "podocyte" in text:
            return "Podocytes"
        if "parietal" in text:
            return "Parietal epithelium"
        if "endothelial" in text or "vasa recta" in text:
            return "Endothelial cells"
        if any(
            value in text
            for value in (
                "fibroblast",
                "interstitial cell",
                "pericyte",
                "smooth muscle",
                "mesangial",
            )
        ):
            return "Stromal cells"
        if "b cell" in text or "plasma cell" in text or "plasmablast" in text:
            return "B lineage"
        if "t cell" in text or "natural killer" in text or "lymphoid" in text:
            return "T and NK cells"
        if any(
            value in text
            for value in ("macrophage", "monocyte", "dendritic", "myeloid", "mast", "neutrophil")
        ):
            return "Myeloid cells"
    return None


def blood_cell_type(label: str) -> str:
    """Coarse blood cell type of an author label, or "" when the label is ambiguous.

    T-helper subsets and ILC1 cells, which the shared map leaves unlabelled, join T cells
    and NK and ILC respectively.
    """
    harmonized = harmonize_cell_type("blood", label)
    if harmonized is not None:
        return harmonized
    text = label.lower()
    if text.startswith("t-helper") or text == "t follicular helper cell":
        return "T cells"
    if text.startswith("ilc"):
        return "NK and ILC"
    return ""


def ladder_cell_type(label: str) -> str:
    """Coarse blood cell type of an author label, or "" when the label is ambiguous.

    The shared blood map matches "t cell" inside "mast cell" and "malignant cell"; mast cells
    join granulocytes, as that map intends, and malignant cells stay unlabelled.
    """
    if label in LADDER_LABEL_CORRECTIONS:
        return LADDER_LABEL_CORRECTIONS[label]
    return blood_cell_type(label)
