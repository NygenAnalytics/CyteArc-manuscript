args <- commandArgs(trailingOnly=TRUE)
work <- args[[1]]
suppressPackageStartupMessages({library(Seurat); library(Matrix); library(jsonlite)})
counts <- read.csv(file.path(work, 'hto_counts.csv'), row.names=1, check.names=FALSE)
original <- colnames(counts)
safe <- sprintf('HTO%02d', seq_along(original))
x <- t(as.matrix(counts))
rownames(x) <- safe
object <- CreateSeuratObject(counts=Matrix(x, sparse=TRUE), assay='HTO')
object <- NormalizeData(object, assay='HTO', normalization.method='CLR', margin=1, verbose=FALSE)
object <- HTODemux(object, assay='HTO', positive.quantile=0.99, init=length(safe)+1,
                   nstarts=100, kfunc='kmeans', seed=0, verbose=FALSE)
restore <- function(values) {
  values <- as.character(values)
  positions <- match(values, safe)
  values[!is.na(positions)] <- original[positions[!is.na(positions)]]
  values
}
output <- data.frame(barcode=colnames(object), identity=restore(object$hash.ID),
                     classification=as.character(object$HTO_classification.global),
                     max_id=restore(object$HTO_maxID), second_id=restore(object$HTO_secondID),
                     margin=object$HTO_margin)
write.csv(output, file.path(work, 'reference_hto.csv'), row.names=FALSE)
versions <- list(R=R.version.string, Seurat=as.character(packageVersion('Seurat')),
                 SeuratObject=as.character(packageVersion('SeuratObject')),
                 fitdistrplus=as.character(packageVersion('fitdistrplus')))
write(toJSON(versions, auto_unbox=TRUE, digits=NA), file.path(work, 'reference.json'))
