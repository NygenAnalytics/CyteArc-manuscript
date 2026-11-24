# The marker tests of BPCells and presto alone, on the common inputs of the marker-only
# comparison.
#
# Usage: Rscript markers_r.R MODE WORKERS H5AD EVENTS SUMMARY WORKDIR
#
# The input is the common H5AD of tools/markers_common.py: the cells and genes of the other
# tools' filter, raw counts in an uncompressed file, and the fixed groups in obs/group. Every
# group is tested against the rest on every gene, after normalisation to total counts and log1p.
#
# MODE
#   bpcells  BPCells' streaming Wilcoxon test (marker_features) on the counts written to disk
#            cell-major and then feature-major, as the workflow's BPCells arm does; the two
#            writes and the per-cell totals are the input preparation, each timed.
#   presto   presto's wilcoxauc on a dgCMatrix read with BPCells and normalised as it is read
#            (readMatrix, the input preparation); only where the matrix fits a dgCMatrix.
#
# Afterwards the statistics of a seeded sample of genes and groups (var/marker_sample,
# uns/marker_sample_groups) are written to sample.csv in WORKDIR: BPCells reports P values only,
# presto the AUROC, the U statistic and P values.

args <- commandArgs(trailingOnly = TRUE)
mode <- args[[1]]
workers <- as.integer(args[[2]])
h5ad <- args[[3]]
events <- args[[4]]
summary_path <- args[[5]]
workdir <- args[[6]]
stopifnot(mode %in% c("bpcells", "presto"))

suppressPackageStartupMessages({
  library(jsonlite)
  library(Matrix)
  library(BPCells)
  library(hdf5r)
})
RhpcBLASctl::blas_set_num_threads(workers)
options(warn = 1)

summary_data <- list()
save_summary <- function() write(toJSON(summary_data, auto_unbox = TRUE, digits = NA, null = "null"), summary_path)
emit <- function(step, event) {
  line <- toJSON(list(step = step, event = event, t = as.numeric(Sys.time())), auto_unbox = TRUE, digits = NA)
  cat(line, "\n", file = events, append = TRUE, sep = "")
}
# `expr` is a promise, evaluated after the start event, as in workflow_r.R.
step <- function(name, expr) {
  emit(name, "start")
  value <- expr
  emit(name, "end")
  invisible(value)
}

# Small arrays of the input: the groups, the genes and the sample.
read_h5 <- function(key) {
  f <- H5File$new(h5ad, mode = "r")
  on.exit(f$close_all())
  f[[key]]$read()
}
facts <- local({
  f <- H5File$new(h5ad, mode = "r")
  on.exit(f$close_all())
  list(shape = h5attr(f[["X"]], "shape"), nnz = f[["X/data"]]$dims)
})
versions <- list(R = as.character(getRversion()), BPCells = as.character(packageVersion("BPCells")), Matrix = as.character(packageVersion("Matrix")))
if (mode == "presto") versions[["presto"]] <- as.character(packageVersion("presto"))
summary_data$versions <- versions
summary_data$mode <- mode
summary_data$nCells <- facts$shape[[1]]
summary_data$nGenes <- facts$shape[[2]]
summary_data$nnz <- facts$nnz
groups <- as.character(read_h5("obs/group"))
gene_ids <- read_h5("var/_index")
sample_genes <- gene_ids[read_h5("var/marker_sample") == 1]
sample_groups <- as.character(read_h5("uns/marker_sample_groups"))
summary_data$cellsAfterFilter <- length(groups)
summary_data$genesAfterFilter <- length(gene_ids)
summary_data$groups <- length(unique(groups))
save_summary()

# The first of several possible column names that a result table has.
column <- function(table, names) table[[intersect(names, colnames(table))[[1]]]]

if (mode == "bpcells") {
  step("convertMatrix", {
    counts <- convert_matrix_type(open_matrix_anndata_hdf5(h5ad), "uint32_t")
    write_matrix_dir(counts, file.path(workdir, "counts"), overwrite = TRUE)
  })
  counts <- open_matrix_dir(file.path(workdir, "counts"))
  step("transposeMatrix", transpose_storage_order(counts, outdir = file.path(workdir, "counts_by_feature"), tmpdir = workdir))
  by_feature <- open_matrix_dir(file.path(workdir, "counts_by_feature"))
  totals <- step("computeTotals", colSums(counts))
  markers <- step("findMarkers", marker_features(log1p(multiply_cols(by_feature, 1e4 / totals)), groups = groups, method = "wilcoxon"))
  summary_data$markerRows <- nrow(markers)
  summary_data$markerColumns <- colnames(markers)
  save_summary()
  step("sampleStatistics", {
    group <- as.character(column(markers, c("foreground", "group")))
    gene <- as.character(column(markers, c("feature", "gene")))
    keep <- gene %in% sample_genes & group %in% sample_groups
    table <- data.frame(group = group[keep], gene = gene[keep], pval = column(markers, c("p_val_raw", "p_val", "pval"))[keep])
    write.csv(table, file.path(workdir, "sample.csv"), row.names = FALSE)
  })
}

if (mode == "presto") {
  # A dgCMatrix holds at most 2^31 - 1 non-zero values; the test runs only where it fits.
  if (summary_data$nnz > .Machine$integer.max) {
    stop(sprintf("dgCMatrix objects cannot hold more than 2^31 non-zero entries; input matrix has %.0f entries", summary_data$nnz))
  }
  normalised <- step("readMatrix", {
    m <- open_matrix_anndata_hdf5(h5ad)
    totals <- colSums(m)
    as(log1p(multiply_cols(m, 1e4 / totals)), "dgCMatrix")
  })
  markers <- step("findMarkers", presto::wilcoxauc(normalised, groups))
  summary_data$markerRows <- nrow(markers)
  summary_data$markerColumns <- colnames(markers)
  save_summary()
  step("sampleStatistics", {
    keep <- markers$feature %in% sample_genes & markers$group %in% sample_groups
    table <- data.frame(group = markers$group[keep], gene = markers$feature[keep], u = markers$statistic[keep], auc = markers$auc[keep], pval = markers$pval[keep])
    write.csv(table, file.path(workdir, "sample.csv"), row.names = FALSE)
  })
}

save_summary()
