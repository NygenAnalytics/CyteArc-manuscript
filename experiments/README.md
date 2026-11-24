# Experiments

Run the experiments in the order of this table: each experiment comes after the ones it needs. Each folder's README gives the experiment's question, the commands that run it, what it needs and what each record in its `records/` holds. Then `make source-data` at the repository root writes the summaries derived from the records to `.build/derived/<experiment>/`.

All experiments run in the images that `experiments/images.py` builds from the lock files in `experiments/runtime/`: `python-cytearc.txt` for CyteArc and the other `python-*.txt` files for the tools it is compared with.

| Experiment | Measures | Needs |
|---|---|---|
| [lung](lung/) | The cost of updating stored lung analyses to a revised reference | None |
| [integration](integration/) | Harmony and WNN against harmonypy and Seurat, multimodal neighbor graphs on CITE-seq and TEA-seq data of up to 1,457,527 cells, and T-cell subtypes from WNN graphs as references for RNA-only cells | None |
| [methods](methods/) | CyteArc's methods against reference implementations on matched inputs, and its doublet scores, fate probabilities and regulons against experimental labels | None |
| [benchmarks](benchmarks/) | RNA workflow time and memory from 10,000 to 95,624,334 cells, ranged reads per count layout, memory budget compliance, and Scanpy, Seurat, BPCells and Census on six atlases | None |
| [catalogue](catalogue/) | The contents and object sizes of the CyteBase release, its coverage across studies, and in-place interferon screens of its human blood stores | `benchmarks` |
| [reuse](reuse/) | In-place queries of published stores against Census, the time that stored artifacts save, virtual composites against materialized and concatenated copies and BPCells, and circulating progenitors pulled out of every human blood store | `benchmarks`, `catalogue` |
| [interferon_contrasts](interferon_contrasts/) | Disease-versus-healthy contrasts of donor interferon scores in five immune lineages across the human blood stores of CyteBase | `catalogue`, `reuse` |
| [mapping](mapping/) | Query mapping onto CyteArc references against CellTypist, Seurat and Symphony, reference sizes of 0.5 to 4.29 million cells, and in-place annotation of blood stores | `benchmarks`, `reuse` |

## Data locations

The experiments kept their prepared datasets, stores and run records under the key prefix `experiments/` of a Cloudflare R2 bucket of Nygen Analytics, which the scripts reach through R2's S3-compatible API; this is why the bucket variables are named `R2_*`. Raw run outputs and intermediate stores are held by Nygen Analytics and are available from the corresponding author on request. To rerun an experiment, set `R2_ENDPOINT_URL`, `R2_BUCKET`, `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY` to the endpoint, name and credentials of your own S3-compatible bucket.

Scripts keep local data outside this folder, in the places that these environment variables name. A script that needs a variable that is not set, the `R2_*` ones included, stops with an error that names it.

| Variable | Names |
|---|---|
| `CYTEARC_DOWNLOADS` | The folder for the public datasets that scripts download, with one subfolder per experiment |
| `CYTEARC_RUNS` | The folder for runs on this machine: each experiment's work stores in `<experiment>/workspace/` and results in `<experiment>/results/`, and the source archive of each Modal launch in `harness/` |
| `CYTEARC_RESULTS` | A results folder for a single run, used instead of `$CYTEARC_RUNS/<experiment>/results/`; `runtime.method_checks` sets it for the check it runs |

The benchmark configurations in `benchmarks/configs/` name objects by their keys in the bucket. The Modal runner passes `R2_ENDPOINT_URL` and `R2_BUCKET` into each job, the images set `CYTEARC_DOWNLOADS` and `CYTEARC_RUNS` to folders on the job's own disk, and jobs read the bucket credentials from the Modal secret that the runner's `--secret` option names.
