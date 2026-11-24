from manuscript.tables.render import read_metrics, write

P = r">{\raggedright\arraybackslash}p"


def sci(value: float, digits: int = 1) -> str:
    """A number in LaTeX scientific notation, for example $5.1 \\times 10^{-4}$."""
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    exponent = int(exponent)
    if float(mantissa) == 1.0:
        return f"$10^{{{exponent}}}$"
    return f"${mantissa} \\times 10^{{{exponent}}}$"


def tabular(spec: str, header: str, rows: list[str], pre: str = "") -> str:
    body = "\n    ".join(rows)
    return (
        f"{pre}\\begin{{tabular}}{{{spec}}}\n    \\toprule\n{header}    \\midrule\n    "
        + body
        + "\n    \\bottomrule\n  \\end{tabular}"
    )


def group_heading(cols: int, title: str, detail: str = "") -> str:
    """A group heading as a shaded band, as in Table 2; a detail after the title is in normal weight."""
    text = (
        f"{{\\sffamily\\bfseries {title}}} {detail}" if detail else f"\\sffamily\\bfseries {title}"
    )
    return f"\\rowcolor{{tableband}}\\multicolumn{{{cols}}}{{@{{\\hspace{{3pt}}}}l}}{{{text}}} \\\\"


def agreement_table(v: dict) -> str:
    rows = []

    def section(title):
        if rows:
            rows.append(r"\addlinespace[3pt]")
        rows.append(group_heading(4, title))

    def add(comparison, reference, data, result):
        rows.append(" & ".join((comparison, reference, data, result)) + r" \\")

    def difference(value):
        return "Exact" if value == 0 else f"Max $|\\Delta|$ {sci(value)}"

    pca, leiden = v["pca"], v["leiden"]
    pipeline, lsi = v["pipelineRepeatability"], v["lsi"]
    genes, wnn, scores = v["geneScores"], v["wnn"], v["aucell"]
    section("Numerical agreement with matched settings")
    add(
        "PCA (Gram and Krylov)",
        "Dense float64 SVD",
        f"{pca['cells']:,} cells; {pca['minimumFeatures']:,}--{pca['maximumFeatures']:,} genes; {pca['dimensions']} PCs",
        f"Max angle ${pca['maxSubspaceAngleDegrees']:.3g}^\\circ$; relative variance error {sci(pca['maxRelativeVarianceError'])}",
    )
    add(
        "Native Leiden",
        "igraph; thread repetitions",
        f"{leiden['cells']:,} cells; "
        + ", ".join(str(value) for value in leiden["threads"].values())
        + " threads",
        (
            "Partitions identical across threads; "
            if leiden["nativePartitionsIdentical"]
            else "Partitions differ across threads; "
        )
        + f"modularity {leiden['nativeModularity']:.5f}/{leiden['igraphModularity']:.5f}",
    )
    add(
        "Complete RNA pipeline",
        "Fresh runs without reuse",
        f"{pipeline['cells']:,} cells; {pipeline['runsPerMode']} deterministic runs",
        (
            "Sequential arrays identical; "
            if pipeline["allArraysIdentical"]
            else "Sequential arrays differ; "
        )
        + f"ARI {pipeline['deterministicARI']:.2f}",
    )
    add(
        "Batchwise LSI",
        "Dense SVD; first component omitted",
        f"{lsi['cells']:,} cells; {lsi['peaks']:,} peaks; {lsi['dimensions']} components",
        f"{lsi['iterations']} iterations, {lsi['oversamples']} oversamples; "
        + f"max angle ${lsi['maxSubspaceAngleDegrees']:.3g}^\\circ$",
    )
    add(
        "WNN",
        f"Seurat {wnn['seuratVersion']}; shared candidate pools and bandwidths",
        f"{wnn['citeCells']:,} CITE-seq cells; {wnn['teaCells']:,} TEA-seq cells",
        ("Neighbors identical; " if wnn["neighborsIdentical"] else "Neighbor indices differ; ")
        + difference(wnn["maxAbs"]),
    )
    symphony = v["symphony"]
    harmony = v["harmony"]
    steps, fits = harmony["steps"], harmony["fullFits"]
    harmony_data = (
        " and ".join(f"{cells:,} {name.capitalize()}" for name, cells in harmony["cells"].items())
        + f" PBMCs; {harmony['seeds']} seeds per tool"
    )

    own_fits = fits["harmonypy"]["ownStoppingRules"]
    add(
        "Harmony2",
        f"harmonypy {harmony['harmonypy']}; R harmony {harmony['rHarmony']}; shared PCs and states",
        harmony_data,
        f"Assignment step max $|\\Delta|$ {sci(steps['assignmentPenaltyOff']['harmonypy'])} (harmonypy) "
        + f"and {sci(steps['assignmentPenaltyOff']['rHarmony'])} (R), penalty off; "
        + f"full fits with each tool's stopping rule within harmonypy's seed spread on {own_fits['datasetsPassed']} of {own_fits['datasets']} datasets",
    )
    add(
        "Symphony query correction",
        "R Symphony 0.1.3; unclipped shared scaling",
        f"{symphony['referenceCells']:,} reference and {symphony['cells']:,} query HLCA cells; {symphony['genes']:,} genes",
        f"{difference(symphony['maxAbs'])}; largest correction {symphony['correctionShift']:.2f}",
    )
    add(
        "GeneScores",
        "Dense interval overlap and TF-IDF aggregation",
        f"{genes['cells']:,} cells; {genes['peaks']:,} peaks; {genes['genes']} genes",
        f"{difference(genes['maxAbsoluteError'])}; {genes['genesWithoutOverlaps']} nonoverlapping genes retained",
    )
    chromvar = v["chromvar"]
    add(
        "chromVAR",
        f"R chromVAR {chromvar['referenceVersion']}; shared motif/background peaks",
        f"{chromvar['cells']:,} cells; {chromvar['peaks']:,} peaks; {chromvar['motifs']} motifs",
        f"Max $|\\Delta|$: deviations {sci(chromvar['deviationMaxAbsoluteError'])}; z-scores {sci(chromvar['zScoreMaxAbsoluteError'])}",
    )
    qc = v["dropletQc"]
    add(
        "EmptyDrops",
        f"DropletUtils {qc['emptyDropsVersion']}; shared model precision",
        "/".join(f"{case['testedBarcodes']:,}" for case in qc["datasets"].values())
        + " tested barcodes",
        "Cell-call Jaccard "
        + "/".join(f"{case['cellCallJaccard']:.3f}" for case in qc["datasets"].values())
        + f"; ambient {difference(qc['emptyDropsAmbientMaxAbsoluteError'])}",
    )
    add(
        "Ambient RNA correction",
        f"SoupX {qc['soupXVersion']}; matched pool and contamination fraction",
        "/".join(f"{case['filteredCells']:,}" for case in qc["datasets"].values())
        + " filtered cells",
        f"Contamination {difference(qc['rhoMaxAbsoluteError'])}; corrected-count difference "
        + (
            "$<1$"
            if qc["correctedCountMaxAbsoluteError"] < 1
            else sci(qc["correctedCountMaxAbsoluteError"])
        ),
    )
    hto = v["hto"]
    add(
        "HTO demultiplexing",
        f"Seurat {hto['seuratVersion']}; k-means",
        "; ".join(
            f"{case['cells']:,} cells, {case['hashtags']} tags" for case in hto["datasets"].values()
        ),
        (
            "Global classes identical; "
            if hto["globalClassesIdentical"]
            else "Global classes differ; "
        )
        + f"{hto['identityDifferences']} singlet identity differs",
    )
    add(
        "AUCell; WAGGR mean",
        f"decoupler {scores['version']}",
        f"{scores['cells']:,} cells; {scores['sets']} gene sets",
        f"Max $|\\Delta|$ {sci(scores['aucell'])}; {sci(scores['waggr'])}, respectively",
    )
    grn = v["grnboost2"]
    matched_error = (
        "0.0"
        if grn["matchedMaxAbsoluteDifference"] == 0
        else sci(grn["matchedMaxAbsoluteDifference"])
    )
    add(
        "GRNBoost2",
        f"arboreto {grn['arboretoVersion']}; scikit-learn {grn['sklearnVersion']}",
        f"{grn['cells']:,} cells; {grn['genes']:,} genes; {grn['factors']:,} factors",
        f"Consecutive stopping: max $|\\Delta|$ {matched_error}; $r={grn['matchedPearson']:.3g}$",
    )
    scenic = v["scenic"]
    add(
        "cisTarget recovery",
        f"pySCENIC {scenic['referenceVersion']}; shared ranks and weights",
        f"{scenic['regulators']} regulators; {scenic['seeds']} seeds; {scenic['coreCases']} modules",
        f"AUC: {difference(scenic['aucMaxAbsoluteError'])}; curves: {difference(scenic['curveMaxAbsoluteError'])}",
    )

    m, f = v["markers"], v["fate"]
    section("Markers, trajectories and supporting scores")
    add(
        "Marker statistics",
        "presto; Scanpy; BPCells",
        "Three datasets of about one million cells",
        f"AUROC: {difference(m['auroc'])}; $\\log_{{10}}P$ versus BPCells: {difference(m['log10P'])}",
    )
    add(
        "Fate probabilities",
        "CellRank 2; matched graphs",
        f"{f['pancreasCells']:,} pancreatic cells; {f['larryProgenitors']:,} LARRY progenitors",
        f"$1-r$: {sci(1 - f['pancreasPearsonMin'])}; {sci(1 - f['larryPearson'])}, respectively",
    )
    add(
        "LISI; cell-cycle scores",
        f"scib-metrics {v['lisi']['version']}; Scanpy {v['cellCycle']['version']}",
        "Shared neighbors; shared control genes and gene lists",
        f"{difference(v['lisi']['maxPerCell'])}; {difference(v['cellCycle']['maxAbsSameControls'])}, respectively",
    )
    magic, pba = v["magic"], v["pba"]
    add(
        "MAGIC imputation",
        f"magic-impute {magic['version']}; shared graph and diffusion time",
        f"{magic['cells']:,} PBMCs; {magic['genes']} genes",
        difference(magic["maxAbs"]),
    )
    add(
        "Population-balance pseudotime",
        "Dense truncated potential (self-check)",
        f"{pba['cells']:,} pancreatic cells; same {pba['modesUsed']} modes",
        f"Spearman $\\rho={pba['denseTruncatedSpearman']:.5f}$; "
        + f"max $|\\Delta|$ {sci(pba['versusDenseTruncated'], 3)}",
    )
    spec = "@{}" + "".join(f"{P}{{{x}\\textwidth}}" for x in AGREEMENT_WIDTHS) + "@{}"
    return tabular(
        spec,
        "    \\textbf{Comparison} & \\textbf{Reference} & \\textbf{Data} & \\textbf{Result} \\\\\n",
        rows,
        pre="\\sffamily\\setstretch{1}\\color{ink}\\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.12}\n",
    )


AGREEMENT_WIDTHS = (0.22, 0.19, 0.235, 0.30)


def render_all():
    write("table2_agreement", agreement_table(read_metrics("table_2/data")) + "\n")
