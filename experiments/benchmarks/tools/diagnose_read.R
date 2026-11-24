# Time BPCells' AnnData reader operation by operation on one H5AD file.
# Usage: Rscript diagnose_read.R H5AD
suppressPackageStartupMessages(library(BPCells))
path <- commandArgs(trailingOnly = TRUE)[[1]]
time_it <- function(label, expr) {
  t <- system.time(v <- expr)[["elapsed"]]
  message(sprintf("%-32s %8.1f s", label, t))
  invisible(v)
}
m <- time_it("open (lazy)", open_matrix_anndata_hdf5(path))
time_it("dim", dim(m))
time_it("colnames", length(colnames(m)))
time_it("rownames", length(rownames(m)))
time_it("colSums (one pass)", sum(colSums(m)))
