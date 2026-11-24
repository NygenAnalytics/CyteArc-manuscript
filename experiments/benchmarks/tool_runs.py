WORKFLOW_TOOLS = ("scanpy", "scanpy_dask_materialized_pca", "seurat", "seurat_bpcells", "bpcells")
ALL_TIERS = (32, 64, 128, 256, 336)
QUALITY_TAG = "rna_cluster_quality"
QUALITY_DATASET = "embryo_1000000"
QUALITY_CPU = 8
QUALITY_TOOLS = {"cytearc": 32, "bpcells_cached": 32, "scanpy_csc": 64}
