# Reference mapping with Seurat v5 or Symphony on the held-out lung study.
#
# Usage: Rscript mapping_r.R MODE WORKERS REFERENCE EVENTS SUMMARY WORKDIR
# The query H5AD is given in the QUERY_H5AD environment variable; both files come from
# prepare_hlca.py. Predictions with a confidence for abstention, for every query cell and
# annotation level, are written to WORKDIR/predictions.csv.
#
# seurat_map  Seurat's reference-mapping workflow: log-normalisation, 2,000 variable genes,
#             30 principal components of the reference, FindTransferAnchors with the
#             reference PCA projected onto the query, TransferData per annotation level; the
#             confidence is prediction.score.max.
# symphony    Symphony: buildReference with Harmony over the study (2,000 variable genes,
#             20 dimensions, K = 100), mapQuery, and knnPredict with k = 5; the confidence is
#             the share of the five neighbours that carry the predicted label.

args <- commandArgs(trailingOnly = TRUE)
mode <- args[[1]]
workers <- as.integer(args[[2]])
reference_path <- args[[3]]
events <- args[[4]]
summary_path <- args[[5]]
workdir <- args[[6]]
query_path <- Sys.getenv("QUERY_H5AD")
stopifnot(mode %in% c("seurat_map", "symphony"))
levels <- strsplit(Sys.getenv("MAPPING_LABEL_COLUMNS", "ann_level_3,cell_type,ann_finest_level"), ",", fixed = TRUE)[[1]]
query_batch <- Sys.getenv("MAPPING_QUERY_BATCH", "")

suppressPackageStartupMessages({
  library(jsonlite)
  library(Matrix)
  library(BPCells)
  library(hdf5r)
  if (mode == "seurat_map") library(Seurat) else library(symphony)
})
RhpcBLASctl::blas_set_num_threads(workers)
options(future.globals.maxSize = Inf, warn = 1)
set.seed(4444)

summary_data <- list(mode = mode, labelColumns = as.list(levels), queryBatchColumn = query_batch)
save_summary <- function() write(toJSON(summary_data, auto_unbox = TRUE, digits = NA, null = "null"), summary_path)
emit <- function(step, event) {
  cat(toJSON(list(step = step, event = event, t = as.numeric(Sys.time())), auto_unbox = TRUE, digits = NA), "\n", file = events, append = TRUE, sep = "")
}
# `expr` is a promise, evaluated in the caller's frame after the start event.
step <- function(name, expr) {
  emit(name, "start")
  value <- expr
  emit(name, "end")
  invisible(value)
}

read_obs <- function(path) {
  f <- H5File$new(path, mode = "r")
  on.exit(f$close_all())
  obs <- f[["obs"]]
  columns <- setdiff(names(obs), "_index")
  frame <- as.data.frame(lapply(setNames(columns, columns), function(name) obs[[name]]$read()), stringsAsFactors = FALSE)
  rownames(frame) <- make.unique(obs[["_index"]]$read())
  frame
}
matrix_nonzeros <- function(path) {
  f <- H5File$new(path, mode = "r")
  on.exit(f$close_all())
  f[["X/data"]]$dims
}
read_counts <- function(path, nonzeros) {
  # A dgCMatrix holds at most 2^31 - 1 non-zero values, and BPCells loads every value before it
  # checks, so a larger matrix runs out of memory first; the limit is checked here instead,
  # with the same message.
  if (nonzeros > .Machine$integer.max) {
    stop(sprintf("dgCMatrix objects cannot hold more than 2^31 non-zero entries; input matrix has %.0f entries", nonzeros))
  }
  m <- as(open_matrix_anndata_hdf5(path, buffer_size = 4194304L), "dgCMatrix")
  rownames(m) <- make.unique(rownames(m))
  colnames(m) <- make.unique(colnames(m))
  m
}

summary_data$referenceNonzeroCounts <- matrix_nonzeros(reference_path)
summary_data$queryNonzeroCounts <- matrix_nonzeros(query_path)
summary_data$dgCMatrixLimit <- .Machine$integer.max
save_summary()
reference_counts <- step("readReference", read_counts(reference_path, summary_data$referenceNonzeroCounts))
reference_obs <- step("readReferenceMetadata", read_obs(reference_path))
stopifnot(all(levels %in% names(reference_obs)))
study <- intersect(c("dataset", "study", "donor_id"), colnames(reference_obs))[[1]]

if (mode == "seurat_map") {
  reference <- step("buildReference", {
    reference <- CreateSeuratObject(counts = reference_counts, meta.data = reference_obs)
    reference <- NormalizeData(reference, verbose = FALSE)
    reference <- FindVariableFeatures(reference, nfeatures = 2000, verbose = FALSE)
    reference <- ScaleData(reference, verbose = FALSE)
    RunPCA(reference, npcs = 30, seed.use = 4444, verbose = FALSE)
  })
  reference <- step("buildReferenceIndex", FindNeighbors(reference, reduction = "pca", dims = 1:30,
    k.param = 31, return.neighbor = TRUE, compute.SNN = FALSE, l2.norm = TRUE,
    cache.index = TRUE, graph.name = "reference_neighbors", verbose = FALSE))
  summary_data$versions <- list(Seurat = as.character(packageVersion("Seurat")), SeuratObject = as.character(packageVersion("SeuratObject")))
}

if (mode == "symphony") {
  reference <- step("buildReference", {
    expression <- symphony:::normalizeData(reference_counts, 1e4, "log")
    buildReference(expression, reference_obs, vars = study, K = 100, verbose = FALSE, do_umap = FALSE,
                   do_normalize = FALSE, vargenes_method = "vst", topn = 2000, d = 20, seed = 4444)
  })
  rm(expression)
  summary_data$versions <- list(symphony = as.character(packageVersion("symphony")), harmony = as.character(packageVersion("harmony")))
}
rm(reference_counts)
reference_file <- file.path(workdir, "reference.rds")
index_file <- file.path(workdir, "reference.annoy")
step("persistReference", {
  if (mode == "seurat_map") SaveAnnoyIndex(reference[["reference_neighbors"]], index_file)
  saveRDS(reference, reference_file, compress = FALSE)
})
summary_data$referenceRdsBytes <- file.info(reference_file)$size
summary_data$referenceIndexBytes <- if (mode == "seurat_map") file.info(index_file)$size else 0
summary_data$referencePersistence <- "Uncompressed RDS; Seurat also retains its reusable L2-normalized PCA Annoy index"
rm(reference)
invisible(gc())
reference <- step("loadPreparedReference", {
  restored <- readRDS(reference_file)
  if (mode == "seurat_map") restored[["reference_neighbors"]] <- LoadAnnoyIndex(restored[["reference_neighbors"]], index_file)
  restored
})
query_counts <- step("readQuery", read_counts(query_path, summary_data$queryNonzeroCounts))
query_obs <- step("readQueryMetadata", read_obs(query_path))
stopifnot(all(levels %in% names(query_obs)))
if (nzchar(query_batch)) stopifnot(query_batch %in% names(query_obs))
predictions <- data.frame(cell = rownames(query_obs))
for (level in levels) predictions[[paste0("true_", level)]] <- query_obs[[level]]

if (mode == "seurat_map") {
  query <- step("createQuery", CreateSeuratObject(counts = query_counts, meta.data = query_obs))
  query <- step("normalizeQuery", NormalizeData(query, verbose = FALSE))
  anchors <- step("findAnchors", FindTransferAnchors(reference = reference, query = query, dims = 1:30,
    reference.reduction = "pca", reference.neighbors = "reference_neighbors", verbose = FALSE))
  for (level in levels) {
    transferred <- step(paste0("transfer_", level), TransferData(anchorset = anchors, refdata = reference[[level, drop = TRUE]], dims = 1:30, verbose = FALSE))
    predictions[[paste0("pred_", level)]] <- as.character(transferred$predicted.id)
    predictions[[paste0("conf_", level)]] <- transferred$prediction.score.max
  }
}
if (mode == "symphony") {
  query <- step("mapQuery", {
    expression <- symphony:::normalizeData(query_counts, 1e4, "log")
    mapQuery(expression, query_obs, reference, vars = if (nzchar(query_batch)) query_batch else NULL,
             verbose = FALSE, do_normalize = FALSE, do_umap = FALSE)
  })
  for (level in levels) {
    query <- step(paste0("transfer_", level), knnPredict(query, reference, reference$meta_data[[level]], k = 5, save_as = "prediction", confidence = TRUE))
    predictions[[paste0("pred_", level)]] <- as.character(query$meta_data$prediction)
    predictions[[paste0("conf_", level)]] <- query$meta_data$prediction_prob
  }
}

write.csv(predictions, file.path(workdir, "predictions.csv"), row.names = FALSE)
summary_data$queryCells <- nrow(predictions)
for (level in levels) summary_data[[paste0("agreementNoAbstention_", level)]] <- mean(predictions[[paste0("true_", level)]] == predictions[[paste0("pred_", level)]])
save_summary()
