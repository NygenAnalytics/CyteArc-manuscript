from manuscript.tables.render import read_table, write


def tabular(spec: str, header: list[str], body: list[str], pre: str = "") -> str:
    lines = [f"{pre}\\begin{{tabular}}{{{spec}}}", "    \\toprule"]
    lines += [f"    {h}" for h in header]
    lines.append("    \\midrule")
    lines += [f"    {b}" for b in body]
    lines += ["    \\bottomrule", "  \\end{tabular}"]
    return "\n".join(lines) + "\n"


def memory_cell(tier: float, marks: str) -> str:
    """Container memory for the grey second line; marks hang right so the number stays centred."""
    return f"\\tabmem{{{tier:.0f}" + (f"\\rlap{{{marks}}}" if marks else "") + "}"


DATASETS = (
    ("embryo_1000000", "Embryo", "", 8),
    ("brain_1306127", "Mouse", "brain", 8),
    ("tabula_sapiens", "Tabula", "Sapiens", 8),
    ("hbca_neurons", "Brain-atlas", "neurons", 16),
    ("mssm_cohort", "PFC", "cohort", 16),
    ("embryo_10000000", "Embryo", "", 16),
)
FAILURE_MARK = {"OOM": "\\oommark", "TL": "\\tlmark"}
MATRIX_LIMIT = "matrix limit"
RUNS_PER_ARM = 3
TOOLKIT_ARMS = (
    ("cytearc", "CyteArc", "24 GiB budget"),
    ("bpcells_cached", "BPCells", ""),
    ("seurat_bpcells", "Seurat", "(BPCells)"),
    ("seurat", "Seurat", "(memory)"),
    ("scanpy", "Scanpy", "(memory)"),
    ("scanpy_dask_covariance_csc", "Scanpy", "(Dask)"),
    ("census_scanpy", "Scanpy", "(Census)"),
)
STEP_SHORT = {
    "read": "read",
    "readLazy": "read",
    "convertMatrix": "conversion",
    "filterCells": "filter",
    "runNormalization": "normalize",
    "markHvgs": "variable genes",
    "runPca": "PCA",
    "queryNeighbors": "neighbors",
    "runUmap": "UMAP",
    "runLeiden": "Leiden",
    "findMarkers": "markers",
    "materializeMarkerInput": "marker input",
}


def render_toolkits():
    rows = read_table("extended_table_1/data")
    lookup = {(r["tool"], r["dataset"]): r for r in rows}
    arms = []
    for tool, line1, line2 in TOOLKIT_ARMS:
        first, second = ([line1], [f"\\quad {line2}" if line2 else ""])
        for dataset, *_ in DATASETS:
            r = lookup[tool, dataset]
            if r["status"] == "not run":
                first.append("--")
                second.append("")
            elif r["status"] == "completed":
                count = "" if r["runs"] == RUNS_PER_ARM else f" ({r['runs']})"
                value = f"{r['mean_minutes']:.1f}"
                if r["fastest"]:
                    value = f"\\textbf{{{value}}}"
                first.append(value + count)
                star = (
                    "*" if r["ladder_start_gib"] == r["memory_gib"] and r["memory_gib"] > 32 else ""
                )
                second.append(
                    memory_cell(r["memory_gib"], star + ("\\dag" if r["scaled_pca"] else ""))
                )
            elif r["status"] == "matrix limit":
                first.append(MATRIX_LIMIT)
                second.append(MATRIX_LIMIT)
            else:
                step = STEP_SHORT.get(r["failed_step"], r["failed_step"] or "")
                first.append(f"{FAILURE_MARK.get(r['status'], r['status'])}~{step}")
                second.append(memory_cell(r["memory_gib"], "\\dag" if r["scaled_pca"] else ""))
        if MATRIX_LIMIT in first:
            k = first.index(MATRIX_LIMIT)
            if not all((c == MATRIX_LIMIT for c in first[k:])):
                raise ValueError("noncontiguous matrix-limit columns")
            count = len(first) - k
            first[k:] = [
                f"\\multicolumn{{{count}}}{{c}}{{\\itshape not possible: more than $2^{{31}}-1$ nonzero values for a dgCMatrix}}"
            ]
            second[k:] = [f"\\multicolumn{{{count}}}{{c}}{{}}"]
        arms.append((tool, first, second))
    body = []
    for index, (tool, first, second) in enumerate(arms):
        if index:
            body.append("\\addlinespace[2pt]")
        shade = (
            "cytearc" if tool.startswith("cytearc") else "tablezebra" if index % 2 == 1 else None
        )
        prefix = f"\\rowcolor{{{shade}}}" if shade else ""
        body.extend([prefix + " & ".join(first) + " \\\\", prefix + " & ".join(second) + " \\\\"])
    cells = [lookup["cytearc", d]["cells"] for d, *_ in DATASETS]
    header = [
        "\\rowcolor{tableband} & \\multicolumn{3}{c}{\\sffamily\\bfseries 8 CPUs} & \\multicolumn{3}{c}{\\sffamily\\bfseries 16 CPUs} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}",
        " & " + " & ".join((d[1] for d in DATASETS)) + " \\\\",
        " & " + " & ".join((d[2] for d in DATASETS)) + " \\\\",
        "\\textbf{Toolkit} & "
        + " & ".join(
            (
                f"{n / 1000000.0:.0f}M cells" if n >= 10000000.0 else f"{n / 1000000.0:.1f}M cells"
                for n in cells
            )
        )
        + " \\\\",
    ]
    return tabular(
        "@{}>{\\raggedright\\arraybackslash}p{0.16\\textwidth}*{6}{>{\\centering\\arraybackslash}p{0.11\\textwidth}}@{}",
        header,
        body,
        pre="\\sffamily\\setstretch{1}\\color{ink}\\setlength{\\tabcolsep}{3pt}\n",
    )


def render_all():
    write("edtable1_toolkits", render_toolkits())
