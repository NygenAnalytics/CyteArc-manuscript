# R workflows for the comparison: Seurat in memory, Seurat with BPCells, and BPCells.
#
# Usage: Rscript workflow_r.R MODE WORKERS H5AD EVENTS SUMMARY WORKDIR
#
# MODE
#   seurat          Seurat v5 on a dgCMatrix in memory, with presto for the Wilcoxon test.
#   seurat_bpcells  Seurat v5 on counts held on disk by BPCells, following the Seurat
#                   BPCells workflow; markers with BPCells' streaming Wilcoxon test.
#   bpcells         The BPCells workflow of its authors: normalisation, variable genes and
#                   PCA on the matrix on disk, HNSW neighbours, Leiden, uwot UMAP and
#                   BPCells' streaming Wilcoxon test.
#   seurat_bpcells_spca, bpcells_spca
#                   The same, with uwot's scaled PCA initialisation ("spca") instead of its
#                   default spectral initialisation, which crashed (a segmentation fault in
#                   RSpectra) on 2.4 and 3.9 million cells. These are variants reported beside
#                   the default runs, not replacements.
#   bpcells_cached, bpcells_cached_spca
#                   The bpcells arm with the normalised variable genes written to disk before
#                   PCA, as the BPCells tutorial does "to improve performance of the downstream
#                   PCA" (write_matrix_dir, its choice for large data): svds then reads stored
#                   values instead of recomputing the normalisation at every iteration. The
#                   bpcells arm without this step is kept to reproduce the earlier records.
#   bpcells_snn, bpcells_igraph
#                   The BPCells arm clustered with BPCells' igraph-based Leiden
#                   (cluster_graph_leiden) on the shared-nearest-neighbour graph of the BPCells
#                   tutorial, or on the geodesic graph, both built inside the Leiden step. Both
#                   were killed for memory at one million cells, which is why the bpcells arm
#                   clusters through cluster_graph_seurat; kept to reproduce those records.
#
# The steps and parameters match the RNA workflow benchmark (see workflow_scanpy.py). The disk
# arms write the counts twice, cell-major for cell-wise steps and feature-major for the
# marker test, as BPCells requires; the two writes are timed as conversion steps.
#
# With KEEP_OUTPUTS=1 (modal_tools.py --keep-outputs, the quality comparison; the BPCells arms
# only), a last step, keepOutputs, writes the per-cell outputs to WORKDIR/outputs.h5 in the layout
# of tools/kept_outputs.py: every input cell's identifier, the retained cells, their Leiden
# clusters, principal components and UMAP coordinates. Without it nothing changes.

args <- commandArgs(trailingOnly = TRUE)
mode <- args[[1]]
workers <- as.integer(args[[2]])
h5ad <- args[[3]]
events <- args[[4]]
summary_path <- args[[5]]
workdir <- args[[6]]
umap_init <- "spectral"
if (endsWith(mode, "_spca")) {
  umap_init <- "spca"
  mode <- sub("_spca$", "", mode)
}
cache_pca <- FALSE
if (endsWith(mode, "_cached")) {
  cache_pca <- TRUE
  mode <- sub("_cached$", "", mode)
}
leiden_graph <- NULL
if (mode %in% c("bpcells_snn", "bpcells_igraph")) {
  leiden_graph <- sub("^bpcells_", "", mode)
  mode <- "bpcells"
}
stopifnot(mode %in% c("seurat", "seurat_bpcells", "bpcells"))
keep_outputs <- identical(Sys.getenv("KEEP_OUTPUTS"), "1")
if (keep_outputs && mode != "bpcells") stop("KEEP_OUTPUTS applies to the BPCells arms only")
# WORKFLOW_SETTINGS (the bpcells arm only) replaces parameters as in workflow_scanpy.py: filterCells (false skips
# cell and gene filtering), variableGenes, variableGeneExclusionPattern (a regular expression on gene symbols;
# matching genes are not candidates), principalComponents, neighbors, umap and markers (false skips the step).
settings <- if (nzchar(Sys.getenv("WORKFLOW_SETTINGS"))) jsonlite::fromJSON(Sys.getenv("WORKFLOW_SETTINGS")) else list()
known_settings <- c("filterCells", "variableGenes", "variableGeneExclusionPattern", "principalComponents", "neighbors", "umap", "markers")
if (length(setdiff(names(settings), known_settings))) stop(sprintf("WORKFLOW_SETTINGS keys must be among %s", paste(known_settings, collapse = ", ")))
if (length(settings) && mode != "bpcells") stop("WORKFLOW_SETTINGS applies to the bpcells arm only")
setting <- function(name, default) if (is.null(settings[[name]])) default else settings[[name]]

suppressPackageStartupMessages({
  library(jsonlite)
  library(Matrix)
  library(BPCells)
  library(hdf5r)
  if (mode != "bpcells") {
    library(Seurat)
    library(SeuratObject)
  }
})
RhpcBLASctl::blas_set_num_threads(workers)
options(future.globals.maxSize = Inf, Seurat.presto.wilcox.msg = FALSE, warn = 1)
# Seurat runs with its default sequential plan: forked future workers duplicate the count
# matrix and ran the one-million-cell normalisation out of memory. UMAP alone takes its
# thread count from the plan's worker count, so it runs under a multicore plan, which
# multithreads uwot without forking.
if (mode != "bpcells") future::plan("sequential")
with_threads <- function(expr) {
  future::plan("multicore", workers = workers)
  on.exit(future::plan("sequential"))
  expr
}
set.seed(4444)

summary_data <- list()
save_summary <- function() write(toJSON(summary_data, auto_unbox = TRUE, digits = NA, null = "null"), summary_path)
emit <- function(step, event) {
  line <- toJSON(list(step = step, event = event, t = as.numeric(Sys.time())), auto_unbox = TRUE, digits = NA)
  cat(line, "\n", file = events, append = TRUE, sep = "")
}
# `expr` is a promise, evaluated in the caller's frame after the start event, so
# assignments inside a step land where the step was called.
step <- function(name, expr) {
  emit(name, "start")
  value <- expr
  emit(name, "end")
  invisible(value)
}

# H5AD may also be a staged Tahoe-100M revision, a directory whose data/ holds Parquet files: a first
# preparation step, convertParquet, writes them as uncompressed H5AD shards with tools/tahoe_to_h5ad.py,
# which reads them as CyteArc's import does and deletes each Parquet file once converted. The shards
# are then read as a directory of H5AD files.
parquet <- file.path(h5ad, "data")
if (dir.exists(parquet) && length(list.files(parquet, pattern = "\\.parquet$"))) {
  shards <- file.path(workdir, "shards")
  step("convertParquet", {
    status <- system2(Sys.getenv("PYTHON", "python3"), c("-m", "experiments.benchmarks.tools.tahoe_to_h5ad", h5ad, shards, workers))
    if (status != 0) stop("converting the Parquet files to H5AD shards failed")
  })
  h5ad <- shards
}

# H5AD may also be a directory of H5AD files that share their genes in one order, such as one file per study of a
# cohort: the files are read as one matrix, their cells bound together with cbind, so no merged copy is written.
inputs <- if (dir.exists(h5ad)) sort(list.files(h5ad, pattern = "\\.h5ad$", full.names = TRUE)) else h5ad
if (length(inputs) == 0) stop("the input directory holds no H5AD files")

# Gene symbols: CELLxGENE's var/feature_name, or var/gene_short_name in files exported from a CyteArc store.
read_feature_names <- function(path) {
  f <- H5File$new(path, mode = "r")
  on.exit(f$close_all())
  node <- if (f$exists("var/feature_name")) f[["var/feature_name"]] else f[["var/gene_short_name"]]
  if (inherits(node, "H5Group")) {
    codes <- node[["codes"]]$read()
    node[["categories"]]$read()[codes + 1L]
  } else {
    node$read()
  }
}

matrix_facts <- function(path) {
  f <- H5File$new(path, mode = "r")
  on.exit(f$close_all())
  shape <- h5attr(f[["X"]], "shape")
  list(nnz = f[["X/data"]]$dims, nCells = shape[[1]], nGenes = shape[[2]])
}

versions <- list(R = as.character(getRversion()), BPCells = as.character(packageVersion("BPCells")), Matrix = as.character(packageVersion("Matrix")))
if (mode != "bpcells") {
  for (p in c("Seurat", "SeuratObject", "presto", "uwot", "leidenbase")) versions[[p]] <- as.character(packageVersion(p))
} else {
  versions[["uwot"]] <- as.character(packageVersion("uwot"))
}
summary_data$versions <- versions
summary_data$mode <- mode
facts <- lapply(inputs, matrix_facts)
if (length(unique(vapply(facts, function(x) as.numeric(x$nGenes), 0))) != 1) stop("the input files differ in their genes")
summary_data <- c(summary_data, list(
  nnz = sum(vapply(facts, function(x) as.numeric(x$nnz), 0)),
  nCells = sum(vapply(facts, function(x) as.numeric(x$nCells), 0)),
  nGenes = facts[[1]]$nGenes
))
if (length(inputs) > 1) summary_data$inputFiles <- length(inputs)
summary_data$dgCMatrixLimit <- .Machine$integer.max
save_summary()

symbols <- toupper(read_feature_names(inputs[[1]]))
for (path in inputs[-1]) {
  if (!identical(toupper(read_feature_names(path)), symbols)) stop(sprintf("%s orders its genes differently", basename(path)))
}
is_mt <- which(startsWith(symbols, "MT-"))
is_ribo <- which(grepl("^RP[SL]", symbols))

# BPCells' AnnData reader is slow on the gzip-compressed benchmark samples (one pass over
# 10,000 cells: 335 s compressed, 0.6 s uncompressed), so every R arm first writes an
# uncompressed copy of the input and reads that; the copy is a timed step of the workflow.
plain <- if (length(inputs) == 1) file.path(workdir, "input_uncompressed.h5ad") else file.path(workdir, "uncompressed", basename(inputs))
step("uncompressInput", {
  dir.create(dirname(plain[[1]]), showWarnings = FALSE, recursive = TRUE)
  for (i in seq_along(inputs)) {
    status <- system2(Sys.getenv("PYTHON", "python3"), c("-m", "experiments.benchmarks.tools.uncompress_h5ad", inputs[[i]], plain[[i]]))
    if (status != 0) stop(sprintf("uncompressing %s failed", basename(inputs[[i]])))
  }
})
open_counts <- function(path = plain) {
  matrices <- lapply(path, open_matrix_anndata_hdf5)
  if (length(matrices) == 1) matrices[[1]] else do.call(cbind, matrices)
}

quantile_keep <- function(values) {
  q <- quantile(values, c(0.01, 0.99), type = 7, names = FALSE)
  values >= q[[1]] & values <= q[[2]]
}

# Counts on disk: cell-major for cell-wise steps, feature-major for the marker test.
write_disk_counts <- function() {
  step("convertMatrix", {
    counts <- convert_matrix_type(open_counts(), "uint32_t")
    write_matrix_dir(counts, file.path(workdir, "counts"), overwrite = TRUE)
  })
  counts <- open_matrix_dir(file.path(workdir, "counts"))
  # Nothing reads the input after conversion. Deleting it and its uncompressed copy frees the
  # disk that the transposition's sort files need (about 250 GB at ten million cells).
  unlink(c(plain, inputs))
  step("transposeMatrix", transpose_storage_order(counts, outdir = file.path(workdir, "counts_by_feature"), tmpdir = workdir))
  list(by_cell = counts, by_feature = open_matrix_dir(file.path(workdir, "counts_by_feature")))
}

disk_markers <- function(by_feature, genes, cells, totals, groups) {
  step("findMarkers", {
    normalised <- log1p(multiply_cols(by_feature[genes, cells], 1e4 / totals))
    markers <- marker_features(normalised, groups = as.character(groups), method = "wilcoxon")
    summary_data$markerRows <<- nrow(markers)
  })
}

# The per-cell outputs of a run with KEEP_OUTPUTS=1 (tools/kept_outputs.py): every input cell's
# identifier in input order, the 0-based input position of each retained cell (ascending), and
# its cluster, principal components and UMAP coordinates in that order. hdf5r stores a matrix
# with its dimensions reversed (21 x cells as h5py reads it); the reader orients it by the number
# of retained cells.
write_kept_outputs <- function(cell_ids, positions, clusters, pca, umap) {
  pca <- unname(as.matrix(pca))
  umap <- unname(as.matrix(umap))
  stopifnot(
    length(clusters) == length(positions), nrow(pca) == length(positions), nrow(umap) == length(positions),
    all(diff(positions) > 0), positions[[1]] >= 0, positions[[length(positions)]] < length(cell_ids)
  )
  path <- file.path(workdir, "outputs.h5")
  if (file.exists(path)) unlink(path)
  f <- H5File$new(path, mode = "w")
  on.exit(f$close_all())
  f[["cellIds"]] <- as.character(cell_ids)
  f[["positions"]] <- as.integer(positions)
  f[["clusters"]] <- as.character(clusters)
  f[["pca"]] <- pca
  f[["umap"]] <- umap
  # The attributes only describe the file; the reader does not need them.
  try(h5attr(f, "workflow") <- "workflow_r.R", silent = TRUE)
  try(h5attr(f, "mode") <- args[[1]], silent = TRUE)
  invisible(path)
}

# Seurat's RunUMAP, with uwot's initialisation changed only in the _spca variants.
run_umap_seurat <- function(obj) {
  if (umap_init == "spectral") {
    RunUMAP(obj, dims = 1:21, n.neighbors = 11, n.epochs = 300, seed.use = 4444, verbose = FALSE)
  } else {
    RunUMAP(obj, dims = 1:21, n.neighbors = 11, n.epochs = 300, seed.use = 4444, verbose = FALSE, uwot.init = umap_init)
  }
}
summary_data$umapInit <- umap_init
summary_data$cachedPca <- cache_pca
summary_data$leidenGraph <- switch(
  if (is.null(leiden_graph)) "default" else leiden_graph,
  default = "geodesic graph, cluster_graph_seurat",
  snn = "shared-nearest-neighbour graph, cluster_graph_leiden",
  igraph = "geodesic graph, cluster_graph_leiden"
)

if (mode == "seurat") {
  # A dgCMatrix holds at most 2^31 - 1 non-zero values, and BPCells loads every value before it
  # checks, so a larger matrix runs out of memory first; the limit is checked here instead,
  # with the same message.
  if (summary_data$nnz > .Machine$integer.max) {
    stop(sprintf("dgCMatrix objects cannot hold more than 2^31 non-zero entries; input matrix has %.0f entries", summary_data$nnz))
  }
  counts <- step("read", {
    m <- as(open_counts(), "dgCMatrix")
    rownames(m) <- make.unique(rownames(m))
    m
  })
  obj <- step("createObject", CreateSeuratObject(counts = counts, min.cells = 0, min.features = 0))
  rm(counts)
  step("filterCells", {
    counts <- LayerData(obj, layer = "counts")
    total <- obj$nCount_RNA
    detected <- obj$nFeature_RNA
    pct_mt <- 100 * Matrix::colSums(counts[is_mt, , drop = FALSE]) / pmax(total, 1)
    pct_ribo <- 100 * Matrix::colSums(counts[is_ribo, , drop = FALSE]) / pmax(total, 1)
    keep <- quantile_keep(total) & quantile_keep(detected) & quantile_keep(pct_mt) & quantile_keep(pct_ribo) & detected >= 10
    obj <- subset(obj, cells = colnames(obj)[keep])
    counts <- LayerData(obj, layer = "counts")
    cells_per_gene <- tabulate(counts@i + 1L, nbins = nrow(counts))
    obj <- subset(obj, features = rownames(obj)[cells_per_gene >= 20])
    rm(counts)
  })
  obj <- step("runNormalization", NormalizeData(obj, normalization.method = "LogNormalize", scale.factor = 1e4, verbose = FALSE))
  obj <- step("markHvgs", FindVariableFeatures(obj, selection.method = "vst", nfeatures = 1000, verbose = FALSE))
  obj <- step("runPca", {
    obj <- ScaleData(obj, features = VariableFeatures(obj), verbose = FALSE)
    RunPCA(obj, features = VariableFeatures(obj), npcs = 21, seed.use = 4466, verbose = FALSE)
  })
  obj <- step("queryNeighbors", FindNeighbors(obj, dims = 1:21, k.param = 11, verbose = FALSE))
  obj <- step("runUmap", with_threads(run_umap_seurat(obj)))
  obj <- step("runLeiden", FindClusters(obj, algorithm = 4, resolution = 1.0, random.seed = 4444, verbose = FALSE))
  step("findMarkers", {
    markers <- FindAllMarkers(obj, test.use = "wilcox", logfc.threshold = 0, min.pct = 0, return.thresh = 1, verbose = FALSE)
    summary_data$markerRows <- nrow(markers)
  })
  summary_data$cellsAfterFilter <- ncol(obj)
  summary_data$genesAfterFilter <- nrow(obj)
  summary_data$clusters <- length(unique(Idents(obj)))
}

if (mode == "seurat_bpcells") {
  disk <- write_disk_counts()
  obj <- step("createObject", CreateSeuratObject(counts = disk$by_cell, min.cells = 0, min.features = 0))
  step("filterCells", {
    counts <- LayerData(obj, layer = "counts")
    total <- obj$nCount_RNA
    detected <- obj$nFeature_RNA
    pct_mt <- 100 * colSums(counts[is_mt, ]) / pmax(total, 1)
    pct_ribo <- 100 * colSums(counts[is_ribo, ]) / pmax(total, 1)
    keep <- quantile_keep(total) & quantile_keep(detected) & quantile_keep(pct_mt) & quantile_keep(pct_ribo) & detected >= 10
    obj <- subset(obj, cells = colnames(obj)[keep])
    cells_per_gene <- rowSums(binarize(LayerData(obj, layer = "counts")))
    obj <- subset(obj, features = rownames(obj)[cells_per_gene >= 20])
  })
  obj <- step("runNormalization", NormalizeData(obj, normalization.method = "LogNormalize", scale.factor = 1e4, verbose = FALSE))
  obj <- step("markHvgs", FindVariableFeatures(obj, selection.method = "vst", nfeatures = 1000, verbose = FALSE))
  obj <- step("runPca", {
    obj <- ScaleData(obj, features = VariableFeatures(obj), verbose = FALSE)
    RunPCA(obj, features = VariableFeatures(obj), npcs = 21, seed.use = 4466, verbose = FALSE)
  })
  obj <- step("queryNeighbors", FindNeighbors(obj, dims = 1:21, k.param = 11, verbose = FALSE))
  obj <- step("runUmap", with_threads(run_umap_seurat(obj)))
  obj <- step("runLeiden", FindClusters(obj, algorithm = 4, resolution = 1.0, random.seed = 4444, verbose = FALSE))
  disk_markers(disk$by_feature, rownames(obj), colnames(obj), obj$nCount_RNA, Idents(obj))
  summary_data$cellsAfterFilter <- ncol(obj)
  summary_data$genesAfterFilter <- nrow(obj)
  summary_data$clusters <- length(unique(Idents(obj)))
}

if (mode == "bpcells") {
  summary_data$workflowSettings <- settings
  n_components <- setting("principalComponents", 21)
  n_neighbors <- setting("neighbors", 11)
  disk <- write_disk_counts()
  counts <- disk$by_cell
  if (setting("filterCells", TRUE)) {
    step("filterCells", {
      total <- colSums(counts)
      detected <- colSums(binarize(counts))
      pct_mt <- 100 * colSums(counts[is_mt, ]) / pmax(total, 1)
      pct_ribo <- 100 * colSums(counts[is_ribo, ]) / pmax(total, 1)
      keep <- quantile_keep(total) & quantile_keep(detected) & quantile_keep(pct_mt) & quantile_keep(pct_ribo) & detected >= 10
      cells <- which(keep)
      cells_per_gene <- rowSums(binarize(counts[, cells]))
      genes <- which(cells_per_gene >= 20)
      totals <- total[cells]
    })
  } else {
    step("cellTotals", {
      cells <- seq_len(ncol(counts))
      genes <- seq_len(nrow(counts))
      totals <- colSums(counts)
    })
  }
  normalised <- step("runNormalization", log1p(multiply_cols(counts[genes, cells], 1e4 / totals)))
  variable <- step("markHvgs", {
    stats <- matrix_stats(normalised, row_stats = "variance", threads = workers)
    variance <- stats$row_stats["variance", ]
    pattern <- setting("variableGeneExclusionPattern", NULL)
    if (!is.null(pattern)) variance[grepl(pattern, symbols[genes], perl = TRUE)] <- -Inf
    variable <- order(variance, decreasing = TRUE)[seq_len(setting("variableGenes", 1000))]
    means <- stats$row_stats["mean", variable]
    sds <- sqrt(variance[variable])
    variable
  })
  pca <- step("runPca", {
    variable_matrix <- normalised[variable, ]
    if (cache_pca) {
      # The tutorial's step before the normalisations that would make the matrix dense.
      variable_matrix <- write_matrix_dir(variable_matrix, file.path(workdir, "normalised_variable"), overwrite = TRUE)
    }
    scaled <- (variable_matrix - means) / sds
    # The threads argument belongs to BPCells' svds method for its matrices, not to the generic.
    decomposition <- BPCells::svds(scaled, k = n_components, threads = workers)
    multiply_cols(decomposition$v, decomposition$d)
  })
  # The neighbour step builds the graph too, as Scanpy's pp.neighbors and Seurat's
  # FindNeighbors do: BPCells' geodesic graph is its documented equivalent of the graph Scanpy
  # clusters (umap-learn's fuzzy simplicial set).
  knn <- step("queryNeighbors", {
    knn <- knn_hnsw(pca, k = n_neighbors, threads = workers, ef = 100, verbose = FALSE)
    if (is.null(leiden_graph)) {
      graph <- knn_to_geodesic_graph(knn, threads = workers)
      # Seurat's Graph class, used by cluster_graph_seurat below, needs cell names.
      dimnames(graph) <- list(colnames(counts)[cells], colnames(counts)[cells])
    }
    knn
  })
  if (setting("umap", TRUE)) {
    umap_coordinates <- step("runUmap", uwot::umap(pca, nn_method = list(idx = knn$idx, dist = knn$dist), n_epochs = 300, n_threads = workers, n_sgd_threads = workers, init = umap_init))
    # Only a run that keeps its outputs holds the coordinates beyond their step.
    if (!keep_outputs) rm(umap_coordinates)
  } else if (keep_outputs) {
    stop("KEEP_OUTPUTS needs the UMAP step")
  }
  # Leiden runs through cluster_graph_seurat, BPCells' route to Seurat's clustering: its
  # igraph-based cluster_graph_leiden was killed at one million cells at 32 and 64 GiB on the
  # geodesic graph, and at 32, 64 and 128 GiB on the shared-nearest-neighbour graph of the
  # BPCells tutorial.
  clusters <- if (is.null(leiden_graph)) {
    step("runLeiden", cluster_graph_seurat(graph, resolution = 1.0, algorithm = 4, random.seed = 4444, verbose = FALSE))
  } else if (leiden_graph == "snn") {
    step("runLeiden", cluster_graph_leiden(knn_to_snn_graph(knn), resolution = 1.0, seed = 4444))
  } else {
    step("runLeiden", cluster_graph_leiden(knn_to_geodesic_graph(knn, threads = workers), resolution = 1.0, seed = 4444))
  }
  if (setting("markers", TRUE)) disk_markers(disk$by_feature, genes, cells, totals, clusters)
  summary_data$cellsAfterFilter <- length(cells)
  summary_data$genesAfterFilter <- length(genes)
  summary_data$clusters <- length(unique(clusters))
  if (keep_outputs) {
    save_summary()
    step("keepOutputs", write_kept_outputs(colnames(counts), cells - 1L, clusters, pca, umap_coordinates))
  }
}

save_summary()
