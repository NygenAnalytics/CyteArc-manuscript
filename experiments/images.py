import modal

from experiments.runtime.common import ROOT

RUNTIME = ROOT / "experiments" / "runtime"
GIB = 1024**3
DISK_MB = 524288
CRAN_SNAPSHOT = "https://packagemanager.posit.co/cran/__linux__/noble/2026-09-15"
R_BASE = "rocker/r-ver:4.6.1"
BPCELLS_REF = "v0.3.1"
PRESTO_SHA = "b5df6ee6097eb62522f2e557aa93ac21bab2d05f"
BOTO3 = "boto3==1.43.102"
python_base = modal.Image.debian_slim(python_version="3.14.4").uv_pip_install(
    requirements=[str(RUNTIME / "python-tools.txt")]
)
python_image = python_base
OBSTORE = "obstore==0.11.1"
python_zarr_image = python_base.uv_pip_install(OBSTORE)
r_base = (
    modal.Image.from_registry(R_BASE, add_python="3.14")
    .apt_install(
        "libhdf5-dev",
        "libglpk-dev",
        "libgsl-dev",
        "libxml2-dev",
        "libcurl4-openssl-dev",
        "libssl-dev",
        "libfontconfig1-dev",
        "libfreetype-dev",
        "libpng-dev",
        "libharfbuzz-dev",
        "libfribidi-dev",
        "libicu-dev",
        "libopenblas0-pthread",
        "zlib1g-dev",
        "libzstd-dev",
        "libbz2-dev",
        "liblzma-dev",
        "cmake",
        "git",
        "pkg-config",
    )
    .run_commands(
        f'''echo 'options(repos = c(CRAN = "{CRAN_SNAPSHOT}"))' >> /usr/local/lib/R/etc/Rprofile.site''',
        'Rscript -e \'install.packages(c("remotes", "Seurat", "SeuratObject", "hdf5r", "jsonlite", "leidenbase", "uwot", "RcppHNSW", "RhpcBLASctl", "future"), Ncpus = 8)\'',
        f"""Rscript -e 'remotes::install_github("bnprks/BPCells/r@{BPCELLS_REF}", upgrade = "never", Ncpus = 8)'""",
        f"""Rscript -e 'remotes::install_github("immunogenomics/presto@{PRESTO_SHA}", upgrade = "never")'""",
        'Rscript -e \'for (p in c("Seurat", "SeuratObject", "BPCells", "presto", "leidenbase", "uwot", "hdf5r", "RhpcBLASctl")) stopifnot(requireNamespace(p, quietly = TRUE)); print(sessionInfo())\'',
    )
    .uv_pip_install(BOTO3)
)
r_image = r_base.uv_pip_install("h5py==3.16.0")
CENSUS_VERSION = "2025-11-08"
EMBRYO_DATASET_ID = "dcfd4feb-18a3-4b30-81d7-1b0c544a8ab3"
# The Census arm runs on the Python its lock was compiled for: TileDB-SOMA publishes no Python 3.14 builds.
python_census_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    requirements=[str(RUNTIME / "python-census.txt")]
)
SYMPHONY_SHA = "7c5905988734d9cfe6e1e97a658664717c4ba7b7"
HLCA_CORE_URL = (
    "https://datasets.cellxgene.cziscience.com/688185ad-11c2-4172-a53a-f4f1f4076860.h5ad"
)
python_mapping_image = modal.Image.debian_slim(python_version="3.14.4").uv_pip_install(
    requirements=[str(RUNTIME / "python-mapping.txt")]
)
BIOC_VERSION = "3.23"
DOUBLETFINDER_SHA = "1b244d8f0d54b4b1cb4365639931bbb16f01e1cd"
DOUBLET_DATA_URL = "https://zenodo.org/api/records/4062232/files/real_datasets.zip/content"
r_doublet_image = r_base.run_commands(
    f"""Rscript -e 'install.packages("BiocManager"); BiocManager::install(version = "{BIOC_VERSION}", ask = FALSE, update = FALSE); BiocManager::install(c("scDblFinder", "SingleCellExperiment", "BiocParallel"), ask = FALSE, update = FALSE, Ncpus = 8)'""",
    f"""Rscript -e 'remotes::install_github("chris-mcginnis-ucsf/DoubletFinder@{DOUBLETFINDER_SHA}", upgrade = "never")'""",
    'Rscript -e \'for (p in c("scDblFinder", "DoubletFinder")) stopifnot(requireNamespace(p, quietly = TRUE))\'',
).uv_pip_install(requirements=[str(RUNTIME / "python-tools.txt")])
CYTEARC_APT = ("git", "build-essential", "libfftw3-dev", "libmetis-dev", "libtbb-dev")
python_cytearc_base = (
    modal.Image.debian_slim(python_version="3.14.4")
    .apt_install(*CYTEARC_APT)
    .env({"HNSWLIB_NO_NATIVE": "1"})
    .uv_pip_install(
        requirements=[str(RUNTIME / "python-cytearc.txt")],
        extra_options="--no-binary hnswlib",
    )
)
python_cytearc_image = python_cytearc_base
r_multimodal_image = (
    r_base.apt_install(*CYTEARC_APT)
    .env({"HNSWLIB_NO_NATIVE": "1", "CC": "gcc", "CXX": "g++"})
    .uv_pip_install(
        requirements=[str(RUNTIME / "python-cytearc.txt")],
        extra_options="--no-binary hnswlib",
    )
)
r_mapping_image = r_multimodal_image.run_commands(
    'Rscript -e \'install.packages(c("harmony", "irlba", "class", "purrr", "dplyr"), Ncpus = 8)\'',
    f"""Rscript -e 'remotes::install_github("immunogenomics/symphony@{SYMPHONY_SHA}", upgrade = "never")'""",
    "Rscript -e 'stopifnot(requireNamespace(\"symphony\", quietly = TRUE))'",
)
r_qc_image = r_multimodal_image.run_commands(
    'Rscript -e \'install.packages(c("BiocManager", "SoupX"), Ncpus = 8)\'',
    f'''Rscript -e 'BiocManager::install(version = "{BIOC_VERSION}", ask = FALSE, update = FALSE); BiocManager::install(c("DropletUtils", "BiocParallel"), ask = FALSE, update = FALSE, Ncpus = 8)' '''.strip(),
)
r_chromvar_image = r_qc_image.run_commands(
    'Rscript -e \'BiocManager::install("chromVAR", ask = FALSE, update = FALSE, Ncpus = 8); stopifnot(as.character(packageVersion("chromVAR")) == "1.34.1")\'',
)
CYTEARC_BUDGET_SHARE = 0.75
RDATA = "rdata==1.1.0"
r_doublet_image = (
    r_doublet_image.apt_install(*CYTEARC_APT)
    .env({"HNSWLIB_NO_NATIVE": "1", "CC": "gcc", "CXX": "g++"})
    .uv_pip_install(
        requirements=[str(RUNTIME / "python-cytearc.txt")],
        extra_options="--no-binary hnswlib",
    )
    .uv_pip_install(RDATA)
)
python_cytearc_doublet_image = python_cytearc_base.uv_pip_install(RDATA)
comparison_image = python_cytearc_base.uv_pip_install(
    "scanpy==1.12.4",
    "celltypist==1.7.1",
    "cellrank==2.0.7",
    "leidenalg==0.12.0",
    "dask[array]==2026.8.0",
    "anndata==0.13.4",
)
IMAGES = {
    "cytearc": python_cytearc_image,
    "python": python_image,
    "zarr": python_zarr_image,
    "r": r_image,
    "census": python_census_image,
    "mapping": python_mapping_image,
    "r_mapping": r_mapping_image,
    "doublets": r_doublet_image,
    "cytearc_doublets": python_cytearc_doublet_image,
    "comparisons": comparison_image,
    "r_multimodal": r_multimodal_image,
    "qc": r_qc_image,
    "chromvar": r_chromvar_image,
}
EXTRAS = {
    "regulatory": ("pyscenic==0.12.1", "arboreto==0.1.6", "setuptools==80.9.0", "polars==2.1.0"),
    "harmonypy": ("harmonypy==2.0.2",),
    "scanpy": ("scanpy==1.12.4", "anndata==0.13.4"),
    "magic": ("magic-impute==3.0.0", "graphtools==2.1.0", "scprep==1.1.0", "anndata==0.13.4"),
    "doublet_check": ("scanpy==1.12.4", "scikit-image==0.26.0", RDATA),
    "lisi": ("scib-metrics==0.6.1", "jax==0.11.2", "harmonypy==0.0.10"),
    "decoupler": ("decoupler==2.2.0",),
    "harmonypy_classic": ("harmonypy==0.0.10",),
    "boto3": ("boto3==1.43.104",),
}
# harmonypy 2.0.2 publishes no Python 3.14 wheel, and its source build links BLAS and LAPACK.
EXTRA_APT = {"harmonypy": ("libopenblas-dev", "liblapack-dev")}
for name, packages in EXTRAS.items():
    base = (
        python_cytearc_base.apt_install(*EXTRA_APT[name])
        if name in EXTRA_APT
        else python_cytearc_base
    )
    IMAGES[name] = base.uv_pip_install(*packages)
# Inside a job, downloads, work stores and results go to the container's own disk.
JOB_FOLDERS = {"CYTEARC_DOWNLOADS": "/workspace/downloads", "CYTEARC_RUNS": "/workspace/runs"}
for name, image in IMAGES.items():
    IMAGES[name] = image.env(
        {"PYTHONPATH": "/workspace", "PYTHONDONTWRITEBYTECODE": "1", **JOB_FOLDERS}
    )
