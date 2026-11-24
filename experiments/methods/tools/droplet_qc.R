args <- commandArgs(trailingOnly=TRUE)
work <- args[[1]]
check <- args[[2]]
workers <- as.integer(args[[3]])
suppressPackageStartupMessages({library(Seurat); library(Matrix); library(jsonlite)})
settings <- fromJSON(file.path(work, 'settings.json'))
raw <- Read10X_h5(file.path(work, 'raw.h5'), use.names=FALSE, unique.features=FALSE)
set.seed(settings$seed)
if (check == 'emptydrops') {
  suppressPackageStartupMessages({library(DropletUtils); library(BiocParallel); library(S4Vectors)})
  started <- proc.time()[['elapsed']]
  out <- emptyDrops(raw, lower=settings$lower, retain=settings$retain,
                    niters=settings$iterations, alpha=NULL,
                    BPPARAM=MulticoreParam(workers, RNGseed=settings$seed))
  elapsed <- proc.time()[['elapsed']] - started
  keep <- which(out$Total > settings$lower)
  write.csv(data.frame(barcode=colnames(raw)[keep], as.data.frame(out[keep,])),
            file.path(work, 'reference_calls.csv'), row.names=FALSE)
  ambient <- metadata(out)$ambient
  ambient_genes <- rownames(raw)[Matrix::rowSums(raw) > 0]
  stopifnot(length(ambient_genes) == length(ambient))
  write.csv(data.frame(gene=ambient_genes, ambient=unname(ambient)),
            file.path(work, 'reference_ambient.csv'), row.names=FALSE)
  result <- list(package='DropletUtils', version=as.character(packageVersion('DropletUtils')),
                 seconds=elapsed, alpha=metadata(out)$alpha, retain=metadata(out)$retain)
} else if (check == 'soupx') {
  suppressPackageStartupMessages(library(SoupX))
  probe <- Matrix(matrix(c(1, 0, 99, 0, 0, 100, 101, 0), nrow=2,
                         dimnames=list(c('A', 'B'), c('d1', 'd99', 'd100', 'd101'))), sparse=TRUE)
  constructor_profile <- SoupChannel(probe, probe[, 3, drop=FALSE], soupRange=c(0, 101))$soupProfile$est
  probe_channel <- SoupChannel(probe, probe[, 3, drop=FALSE], calcSoupProfile=FALSE)
  explicit_profile <- estimateSoup(probe_channel, soupRange=c(0, 101))$soupProfile$est
  stopifnot(isTRUE(all.equal(unname(explicit_profile), c(0.5, 0.5), tolerance=1e-15)))
  cells <- Read10X_h5(file.path(work, 'filtered.h5'), use.names=FALSE, unique.features=FALSE)
  sets <- fromJSON(file.path(work, 'gene_sets.json'), simplifyVector=FALSE)
  sets <- lapply(sets, unlist)
  sc <- SoupChannel(raw, cells, calcSoupProfile=FALSE)
  sc <- estimateSoup(sc, soupRange=c(0, settings$lower + 1))
  used <- estimateNonExpressingCells(sc, sets, clusters=FALSE,
                                    maximumContamination=settings$soupMaximumContamination,
                                    FDR=settings$soupFdr)
  sc <- calculateContaminationFraction(sc, sets, used, forceAccept=TRUE)
  corrected <- adjustCounts(sc, clusters=FALSE, method='subtraction', roundToInt=FALSE)
  write.csv(data.frame(gene=rownames(sc$soupProfile), fraction=sc$soupProfile$est,
                       count=sc$soupProfile$counts),
            file.path(work, 'reference_soup.csv'), row.names=FALSE)
  write.csv(used, file.path(work, 'reference_used.csv'))
  writeMM(corrected, file.path(work, 'reference_corrected.mtx'))
  result <- list(package='SoupX', version=as.character(packageVersion('SoupX')),
                 rho=unname(sc$metaData$rho[[1]]), method='subtraction', roundToInt=FALSE,
                 ambientDroplets=sum(sc$nDropUMIs > 0 & sc$nDropUMIs < settings$lower + 1),
                 profileBoundaryCheck=list(constructorRangeProfile=constructor_profile,
                                           explicitRangeProfile=explicit_profile))
} else stop('Unknown comparison')
result$R <- R.version.string
write(toJSON(result, auto_unbox=TRUE, digits=NA), file.path(work, 'reference.json'))
