# CyteArc manuscript

Rebuild the figures, tables and PDFs of the CyteArc paper from the records of its experiments, trace any number in the paper to the run that produced it, and rerun the experiments.

Read the paper in [`publication/manuscript.pdf`](publication/manuscript.pdf) and the [Supplementary Information](publication/supplementary_information.pdf). CyteArc, the datastore and analysis framework the paper presents, is developed at [NygenAnalytics/CyteArc](https://github.com/NygenAnalytics/CyteArc) and released on [PyPI](https://pypi.org/project/cytearc/); the experiments used release 1.3.0.

## What is in this repository

| Folder | Holds |
|---|---|
| `experiments/` | One folder per experiment: the scripts that ran it, a README with its question, commands and inputs, and `records/`, the run records the paper reads, one per run, replicate or donor |
| `manuscript/` | The LaTeX of the paper and the Supplementary Information, the bibliography, and the code that prepares and draws each figure and table |
| `publication/` | Build output: both PDFs, each figure as PDF, PNG and SVG, the tables as CSV, and the plotted data of every figure panel in `figure_source_data/` |
| `tests/` | Tests of the experiment, figure and table code |

## Find the code and data behind a figure or table

Each figure has a folder in `manuscript/figures/`, named for what the figure shows. Its `prepare.py` reads experiment records and the summaries derived from them and writes one JSON file per panel, and its `draw.py` draws the panels in letter order, one function per panel. Each table has a folder in `manuscript/tables/` with its data in `data.csv`: Table 1's is written by hand, and `manuscript/tables/prepare.py` writes the others from the records.

| In the paper | Folder |
|---|---|
| Figure 1 | `manuscript/figures/analysis_scaling/` |
| Figure 2 | `manuscript/figures/zero_copy_analysis/` |
| Figure 3 | `manuscript/figures/artifact_reuse/` |
| Supplementary Fig. 1 | `manuscript/figures/layout_scaling/` |
| Supplementary Fig. 2 | `manuscript/figures/artifact_reuse_measurements/` |
| Supplementary Fig. 3 | `manuscript/figures/in_place_access/` |
| Supplementary Fig. 4 | `manuscript/figures/reference_mapping_measurements/` |
| Supplementary Fig. 5 | `manuscript/figures/multimodal_graph_agreement/` |
| Table 1 | `manuscript/tables/table_1/` |
| Table 2 | `manuscript/tables/table_2/` |
| Extended Data Table 1 | `manuscript/tables/extended_table_1/` |

To trace a number in the text, find the panel or table that reports it, open the panel's JSON file, and follow `prepare.py` to its source: a summary in `.build/derived/<experiment>/`, which `make source-data` writes from the records, or a record in `experiments/<experiment>/records/`. The experiment's README describes each record and the commands and settings that produced it.

## Rebuild the paper

You need [uv](https://docs.astral.sh/uv/), which installs Python 3.14 and the locked packages, and a TeX distribution with `pdflatex`, `bibtex` and the TeX Gyre fonts. The figures use TeX Gyre Heros, so install its OpenType files where matplotlib finds them. From the repository root:

```bash
uv sync --locked --group experiment
make source-data
make
```

The `experiment` dependency group adds CyteArc 1.3.0 and the other packages that the analysis code imports; `make` on its own needs only `uv sync --locked`. `make source-data` recomputes the summaries in `.build/derived/` from the records and writes the figure inputs and table data again; on an unchanged checkout the files it writes are identical to the ones in the repository. `make` draws the figures, renders the tables and builds both PDFs into `publication/`; the rebuilt PDF and SVG files differ from the committed ones only in embedded dates and generated identifiers. `make TARGET=naturemethods` builds the format for peer review, with 1.5 line spacing and line numbers, and `make clean` removes `.build/`. Neither command runs an experiment.

## Run the tests

```bash
uv run --locked --group experiment python -B -m unittest discover -s tests
```

The tests import the experiment code, so they need the `experiment` group too. Two slower tests run only with `CYTEARC_SLOW_TESTS=1` set, and tests that need raw run outputs, which are not in the repository, or network access skip when those are missing.

## Rerun the experiments

The experiments ran on [Modal](https://modal.com) in container images built by `experiments/images.py` from the lock files in `experiments/runtime/`, and read and wrote their data in a Cloudflare R2 bucket of Nygen Analytics through R2's S3-compatible API, so the code names its bucket variables `R2_*`; any S3-compatible bucket works in its place. Raw run outputs and intermediate stores are held by Nygen Analytics and are available from the corresponding author on request. [`experiments/README.md`](experiments/README.md) gives the order to run them in and the environment variables that point the scripts at your own bucket and folders, and each experiment's README gives its commands, payloads and the records they write. `experiments/modal_runner.py` launches a job on Modal, and `uv run python -m experiments MODULE FUNCTION payload.json` runs one on your own machine.

## Data and licences

This repository is released under the [BSD 3-Clause licence](LICENSE). The datasets it reads are public and keep their sources' licences. [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) lists each one the experiments read, with its source and the licence the source states. CyteBase release 2610, the converted CZ CELLxGENE Discover datasets that several experiments read in place, is at [huggingface.co/buckets/Nygen/cytebase-2610](https://huggingface.co/buckets/Nygen/cytebase-2610).
