args <- commandArgs(trailingOnly = TRUE)
rds_file <- args[[1]]
out_dir <- args[[2]]
workers <- as.integer(args[[3]])
method <- if (length(args) >= 4) args[[4]] else "both"
stopifnot(method %in% c("scDblFinder", "doubletFinder", "both"))

suppressPackageStartupMessages({
  library(Matrix)
  library(jsonlite)
  if (method != "doubletFinder") {
    library(SingleCellExperiment)
    library(scDblFinder)
    library(BiocParallel)
  }
  if (method != "scDblFinder") {
    library(Seurat)
    library(DoubletFinder)
  }
})
RhpcBLASctl::blas_set_num_threads(workers)
set.seed(4444)

for (path in rds_file) {
  name <- sub("\\.rds$", "", basename(path))
  target <- file.path(out_dir, name)
  dir.create(target, recursive = TRUE, showWarnings = FALSE)
  data <- readRDS(path)
  counts <- as(data[[1]], "CsparseMatrix")
  labels <- as.character(data[[2]])
  if (is.null(colnames(counts))) colnames(counts) <- paste0("cell", seq_len(ncol(counts)))
  if (is.null(rownames(counts))) rownames(counts) <- paste0("gene", seq_len(nrow(counts)))
  rownames(counts) <- make.unique(rownames(counts))
  colnames(counts) <- make.unique(colnames(counts))
  cat(name, ":", nrow(counts), "genes x", ncol(counts), "cells;", sum(labels == "doublet"), "doublets\n")
  if (method == "both") {
    writeMM(t(counts), file.path(target, "counts.mtx"))
    write.csv(data.frame(cell = colnames(counts), label = labels), file.path(target, "cells.csv"), row.names = FALSE)
  }
  scores <- data.frame(cell = colnames(counts))
  timing <- list(cells = ncol(counts), genes = nrow(counts), doublets = sum(labels == "doublet"))

  if (method != "doubletFinder") {
  started <- Sys.time()
  sce <- scDblFinder(SingleCellExperiment(list(counts = counts)), BPPARAM = MulticoreParam(workers, RNGseed = 4444))
  scdbl <- sce$scDblFinder.score
  timing$scDblFinderSeconds <- as.numeric(difftime(Sys.time(), started, units = "secs"))
  scores$scDblFinder <- scdbl
  }
  if (method != "scDblFinder") {

  started <- Sys.time()
  seu <- CreateSeuratObject(counts = counts)
  seu <- NormalizeData(seu, verbose = FALSE)
  seu <- FindVariableFeatures(seu, selection.method = "vst", nfeatures = 2000, verbose = FALSE)
  seu <- ScaleData(seu, verbose = FALSE)
  seu <- RunPCA(seu, npcs = 10, verbose = FALSE)
  sweep <- paramSweep(seu, PCs = 1:10, sct = FALSE, num.cores = workers)
  best <- find.pK(summarizeSweep(sweep, GT = FALSE))
  pk <- as.numeric(as.character(best$pK[which.max(best$BCmetric)]))
  expected <- round(0.075 * ncol(seu))
  seu <- doubletFinder(seu, PCs = 1:10, pN = 0.25, pK = pk, nExp = expected, sct = FALSE)
  pann <- seu@meta.data[[grep("^pANN_", colnames(seu@meta.data), value = TRUE)[1]]]
  timing$doubletFinderSeconds <- as.numeric(difftime(Sys.time(), started, units = "secs"))
  timing$doubletFinderPk <- pk
  scores$doubletFinder <- pann
  }

  write.csv(scores, file.path(target, "r_scores.csv"), row.names = FALSE)
  write(toJSON(timing, auto_unbox = TRUE, digits = NA), file.path(target, "r_timing.json"))
}
versions <- list(R = as.character(getRversion()))
for (p in intersect(c("scDblFinder", "DoubletFinder", "Seurat", "SingleCellExperiment", "BiocParallel"), loadedNamespaces())) versions[[p]] <- as.character(packageVersion(p))
write(toJSON(versions, auto_unbox = TRUE), file.path(out_dir, "r_versions.json"))
