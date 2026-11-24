args <- commandArgs(trailingOnly=TRUE)
work <- args[[1]]
workers <- as.integer(args[[2]])
suppressPackageStartupMessages({library(chromVAR); library(Matrix); library(BiocParallel); library(jsonlite)})
register(MulticoreParam(workers))
counts <- as(readMM(file.path(work, 'counts.mtx')), 'CsparseMatrix')
annotations <- as(readMM(file.path(work, 'motifs.mtx')) != 0, 'CsparseMatrix')
background <- as.matrix(read.table(file.path(work, 'background.tsv'), header=FALSE, sep='\t'))
started <- proc.time()[['elapsed']]
result <- computeDeviations(counts, annotations=annotations, background_peaks=background)
elapsed <- proc.time()[['elapsed']] - started
options(digits=17)
write.table(deviations(result), file.path(work, 'reference_deviations.csv'), sep=',',
            row.names=FALSE, col.names=FALSE)
write.table(deviationScores(result), file.path(work, 'reference_z.csv'), sep=',',
            row.names=FALSE, col.names=FALSE)
metadata <- list(package='chromVAR', version=as.character(packageVersion('chromVAR')),
                 R=R.version.string, seconds=elapsed)
write(toJSON(metadata, auto_unbox=TRUE, digits=NA), file.path(work, 'reference.json'))
