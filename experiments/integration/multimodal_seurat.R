suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
input <- args[[1]]
output <- args[[2]]
mode <- args[[3]]
threads <- as.integer(args[[4]])
stopifnot(threads > 0)
RhpcBLASctl::blas_set_num_threads(threads)
RhpcBLASctl::omp_set_num_threads(threads)
future::plan(future::sequential)
manifest <- fromJSON(file.path(input, "manifest.json"))
settings <- manifest$settings
modalities <- manifest$modalities
cells <- readLines(file.path(input, "cells.txt"))
load_counts <- function() {
  lapply(modalities, function(assay) {
    matrix <- as(readMM(gzfile(file.path(input, paste0(assay, "_counts.mtx.gz")))), "CsparseMatrix")
    rownames(matrix) <- make.unique(readLines(file.path(input, paste0(assay, "_features.txt"))))
    colnames(matrix) <- cells
    matrix
  })
}
make_object <- function(counts) {
  object <- CreateSeuratObject(counts = counts[[1]], assay = modalities[[1]])
  for (a in seq_along(modalities)) {
    if (a > 1) object[[modalities[[a]]]] <- CreateAssayObject(counts = counts[[a]])
  }
  object
}

# The weights mode runs the workflow's first analysis only and keeps each cell's modality weights.
if (mode %in% c("workflow", "weights")) {
  stopifnot(identical(modalities, c("RNA", "ADT")))
  # Each stage appends its start and end time (seconds since the epoch) as it
  # happens, so a stopped process still shows how far it got and the caller
  # can attribute sampled memory to stages.
  progress <- file.path(output, "progress.tsv")
  mark <- function(stage, event) {
    cat(sprintf("%s\t%s\t%.3f\n", stage, event, as.numeric(Sys.time())), file = progress, append = TRUE)
  }
  mark("input_setup", "start")
  setup.started <- proc.time()[["elapsed"]]
  raw <- make_object(load_counts())
  setup.seconds <- proc.time()[["elapsed"]] - setup.started
  mark("input_setup", "end")
  fit_workflow <- function(raw, resolution, phase) {
    branch.started <- proc.time()[["elapsed"]]
    object <- raw
    timings <- list()
    timed <- function(name, expr) {
      mark(paste(phase, name, sep = ":"), "start")
      started <- proc.time()[["elapsed"]]
      value <- force(expr)
      timings[[name]] <<- proc.time()[["elapsed"]] - started
      mark(paste(phase, name, sep = ":"), "end")
      value
    }
    set.seed(settings$seed)
    DefaultAssay(object) <- "RNA"
    object <- timed("RNA.normalization", NormalizeData(object, normalization.method = "LogNormalize",
                                                       scale.factor = 10000, verbose = FALSE))
    object <- timed("RNA.features", FindVariableFeatures(object, selection.method = "vst",
                                                          nfeatures = settings$hvgCount, verbose = FALSE))
    object <- timed("RNA.scaling", ScaleData(object, features = VariableFeatures(object), verbose = FALSE))
    object <- timed("RNA.reduction", RunPCA(object, features = VariableFeatures(object),
                                           npcs = settings$rnaDimensions, reduction.name = "rna_pca",
                                           reduction.key = "RNApc_", seed.use = settings$seed,
                                           approx = TRUE, verbose = FALSE))
    object <- timed("RNA.neighbors", FindNeighbors(object, reduction = "rna_pca",
                                                    dims = seq_len(settings$rnaDimensions),
                                                    k.param = settings$neighborCount, nn.method = "annoy",
                                                    n.trees = 50, graph.name = c("RNA_nn", "RNA_snn"),
                                                    verbose = FALSE))
    object <- timed("RNA.clustering", FindClusters(object, graph.name = "RNA_snn", algorithm = 4,
                                                   leiden_method = "leidenbase", resolution = settings$resolution,
                                                   random.seed = settings$seed, verbose = FALSE))
    rna.labels <- as.integer(Idents(object))
    DefaultAssay(object) <- "ADT"
    object <- timed("ADT.normalization", NormalizeData(object, normalization.method = "CLR",
                                                       margin = 1, verbose = FALSE))
    object <- timed("ADT.scaling", ScaleData(object, features = rownames(object[["ADT"]]), verbose = FALSE))
    adt.dimensions <- min(settings$adtDimensions, nrow(object[["ADT"]]) - 1L)
    object <- timed("ADT.reduction", RunPCA(object, features = rownames(object[["ADT"]]),
                                           npcs = adt.dimensions, reduction.name = "adt_pca",
                                           reduction.key = "ADTpc_", seed.use = settings$seed,
                                           approx = TRUE, verbose = FALSE))
    object <- timed("ADT.neighbors", FindNeighbors(object, reduction = "adt_pca", dims = seq_len(adt.dimensions),
                                                    k.param = settings$neighborCount, nn.method = "annoy",
                                                    n.trees = 50, graph.name = c("ADT_nn", "ADT_snn"),
                                                    verbose = FALSE))
    object <- timed("ADT.clustering", FindClusters(object, graph.name = "ADT_snn", algorithm = 4,
                                                   leiden_method = "leidenbase", resolution = settings$resolution,
                                                   random.seed = settings$seed, verbose = FALSE))
    adt.labels <- as.integer(Idents(object))
    object <- timed("WNN.integration", FindMultiModalNeighbors(object,
                          reduction.list = list("rna_pca", "adt_pca"),
                          dims.list = list(seq_len(settings$rnaDimensions), seq_len(adt.dimensions)),
                          k.nn = settings$neighborCount, knn.range = 200, l2.norm = TRUE,
                          modality.weight.name = c("RNA.weight", "ADT.weight"), verbose = FALSE))
    object <- timed("WNN.clustering", FindClusters(object, graph.name = "wsnn", algorithm = 4,
                                                   leiden_method = "leidenbase", resolution = resolution,
                                                   random.seed = settings$seed, verbose = FALSE))
    list(object = object, timings = timings, rna = rna.labels, adt = adt.labels,
         wnn = as.integer(Idents(object)),
         dimensions = list(RNA = ncol(Embeddings(object, "rna_pca")), ADT = ncol(Embeddings(object, "adt_pca"))),
         variableGenes = VariableFeatures(object[["RNA"]]),
         wallSeconds = proc.time()[["elapsed"]] - branch.started)
  }
  fit <- fit_workflow(raw, settings$resolution, "initial")
  if (mode == "weights") {
    weights <- fit$object[[]][cells, c("RNA.weight", "ADT.weight")]
    write.csv(data.frame(cell = cells, rna_weight = weights[["RNA.weight"]], adt_weight = weights[["ADT.weight"]]),
              file.path(output, "modality_weights.csv"), row.names = FALSE, quote = FALSE)
    write_json(list(Seurat = as.character(packageVersion("Seurat")),
                    SeuratObject = as.character(packageVersion("SeuratObject")), R = R.version.string,
                    cells = length(cells), threads = threads, inputSetupSeconds = setup.seconds,
                    initialStages = fit$timings, initialWallSeconds = fit$wallSeconds),
               file.path(output, "weights.json"), auto_unbox = TRUE, digits = 16, pretty = TRUE)
    quit(save = "no")
  }
  initial <- fit$timings
  initial.wall <- fit$wallSeconds
  dimensions <- fit$dimensions
  writeLines(fit$variableGenes, file.path(output, "RNA_variable_features.txt"))
  for (name in c("RNA", "ADT", "WNN")) {
    write.table(fit[[tolower(name)]], file.path(output, paste0(name, "_clusters.csv")),
                row.names = FALSE, col.names = FALSE, quote = FALSE)
  }
  object <- fit$object
  rm(fit)
  # Neighbour graphs (cells by cells, compressed sparse columns) and the
  # weighted neighbour rows, for neighbourhood purity and resolution sweeps.
  mark("export_graphs", "start")
  dir.create(file.path(output, "graphs"))
  for (name in c("RNA_nn", "RNA_snn", "ADT_nn", "ADT_snn", "wsnn")) {
    graph <- object[[name]]
    writeBin(as.integer(graph@p), file.path(output, "graphs", paste0(name, "_p.bin")), size = 4)
    writeBin(as.integer(graph@i), file.path(output, "graphs", paste0(name, "_i.bin")), size = 4)
    writeBin(as.numeric(graph@x), file.path(output, "graphs", paste0(name, "_x.bin")), size = 4)
  }
  weighted <- Indices(object[["weighted.nn"]])
  writeBin(as.integer(t(weighted)), file.path(output, "graphs", "weighted_nn_indices.bin"), size = 4)
  write_json(list(cells = length(cells), weightedNeighbors = ncol(weighted),
                  graphs = c("RNA_nn", "RNA_snn", "ADT_nn", "ADT_snn", "wsnn"),
                  layout = "Graphs: dgCMatrix slots p, i (int32, zero-based rows) and x (float32); weighted_nn_indices: int32 cells by neighbours, row-major, one-based"),
             file.path(output, "graphs", "graphs.json"), auto_unbox = TRUE, pretty = TRUE)
  mark("export_graphs", "end")
  mark("reuse_revision:WNN.clustering", "start")
  started <- proc.time()[["elapsed"]]
  object <- FindClusters(object, graph.name = "wsnn", algorithm = 4, leiden_method = "leidenbase",
                         resolution = 1.2, random.seed = settings$seed, verbose = FALSE)
  reused.seconds <- proc.time()[["elapsed"]] - started
  mark("reuse_revision:WNN.clustering", "end")
  write.table(as.integer(Idents(object)), file.path(output, "reused_revision_clusters.csv"),
              row.names = FALSE, col.names = FALSE, quote = FALSE)
  rm(object)
  invisible(gc())
  fit <- fit_workflow(raw, 1.2, "fresh_revision")
  write.table(fit$wnn, file.path(output, "fresh_revision_clusters.csv"),
              row.names = FALSE, col.names = FALSE, quote = FALSE)
  write_json(list(Seurat = as.character(packageVersion("Seurat")),
                  SeuratObject = as.character(packageVersion("SeuratObject")), R = R.version.string,
                  cells = length(cells), modalities = modalities, threads = threads,
                  rawFeatures = manifest$features, actualDimensions = dimensions,
                  settings = list(rnaNormalization = "LogNormalize", scaleFactor = 10000,
                                  variableGenes = settings$hvgCount, variableGeneMethod = "vst",
                                  adtNormalization = "CLR", adtMargin = 1, scaleFeatures = TRUE,
                                  rnaPcs = settings$rnaDimensions, adtPcsMaximum = settings$adtDimensions,
                                  kNn = settings$neighborCount, knnRange = 200, seed = settings$seed,
                                  initialResolution = settings$resolution, revisedResolution = 1.2),
                  inputSetupSeconds = setup.seconds, initialStages = initial,
                  initialSeconds = sum(unlist(initial)), initialWallSeconds = initial.wall,
                  freshRevisionStages = fit$timings, freshRevisionWallSeconds = fit$wallSeconds,
                  freshRevisedSeconds = sum(unlist(fit$timings)), reusedRevisionSeconds = reused.seconds),
              file.path(output, "workflow.json"), auto_unbox = TRUE, digits = 16, pretty = TRUE)
  quit(save = "no")
}

embeddings <- lapply(modalities, function(assay) {
  as.matrix(read.csv(gzfile(file.path(input, paste0(assay, "_embedding.csv.gz"))),
                     header = FALSE))
})
names(embeddings) <- modalities
stopifnot(all(vapply(embeddings, nrow, 0L) == length(cells)))

if (mode == "matched") {
  normalized <- lapply(embeddings, Seurat:::L2Norm)
  indices <- lapply(modalities, function(assay) {
    as.matrix(read.csv(gzfile(file.path(input, paste0(assay, "_indices.csv.gz"))),
                       header = FALSE)) + 1L
  })
  n <- length(cells)
  m <- length(modalities)
  k <- settings$neighborCount
  result.weights <- matrix(NA_real_, n, m)
  result.neighbors <- matrix(NA_integer_, n, k)
  result.affinities <- matrix(NA_real_, n, k)
  kernel <- function(distance, nearest, bandwidth) {
    shifted <- pmax(distance - nearest, 0)
    tolerance <- 8 * .Machine$double.eps * nearest
    if (bandwidth <= tolerance) return(as.numeric(shifted <= tolerance))
    exp(-shifted / bandwidth)
  }
  for (cell in seq_len(n)) {
    pool <- sort(unique(unlist(lapply(indices, function(x) x[cell, ]))))
    distances <- lapply(seq_len(m), function(a) {
      sqrt(rowSums(sweep(normalized[[a]][pool, , drop = FALSE], 2,
                         normalized[[a]][cell, ], "-")^2))
    })
    positions <- lapply(indices, function(x) match(x[cell, ], pool))
    nearest <- vapply(seq_len(m), function(a) min(distances[[a]][positions[[a]]]), 0.0)
    bandwidth <- vapply(seq_len(m), function(a) {
      max(distances[[a]][positions[[a]]]) - nearest[[a]]
    }, 0.0)
    prediction <- function(target, source) {
      estimate <- colMeans(normalized[[target]][indices[[source]][cell, ], , drop = FALSE])
      distance <- sqrt(sum((normalized[[target]][cell, ] - estimate)^2))
      kernel(distance, nearest[[target]], bandwidth[[target]])
    }
    within <- vapply(seq_len(m), function(a) prediction(a, a), 0.0)
    scores <- matrix(-Inf, m, m)
    for (a in seq_len(m)) {
      for (b in seq_len(m)) {
        if (a != b) scores[a, b] <- min(200, max(0, within[[a]] / (prediction(a, b) + 1e-4)))
      }
    }
    strength <- rowSums(exp(scores - max(scores)))
    weight <- strength / sum(strength)
    affinities <- Reduce(`+`, lapply(seq_len(m), function(a) {
      weight[[a]] * kernel(distances[[a]], nearest[[a]], bandwidth[[a]])
    }))
    selected <- order(-affinities, pool)[seq_len(k)]
    result.neighbors[cell, ] <- pool[selected]
    result.affinities[cell, ] <- affinities[selected]
    result.weights[cell, ] <- weight
  }
  write.table(result.weights, file.path(output, "matched_weights.csv"), sep = ",",
              row.names = FALSE, col.names = FALSE)
  write.table(result.affinities, file.path(output, "matched_affinities.csv"), sep = ",",
              row.names = FALSE, col.names = FALSE)
  write.table(result.neighbors, file.path(output, "matched_neighbors.csv"), sep = ",",
              row.names = FALSE, col.names = FALSE)
  quit(save = "no")
}

breadth <- as.integer(mode)
stopifnot(breadth %in% settings$seuratCandidateBreadths)
counts <- load_counts()
object <- make_object(counts)
for (a in seq_along(modalities)) {
  assay <- modalities[[a]]
  values <- embeddings[[a]]
  rownames(values) <- cells
  key <- paste0("dim", a, "_")
  colnames(values) <- paste0(key, seq_len(ncol(values)))
  object[[paste0("reduction", a)]] <- CreateDimReducObject(embeddings = values,
                                                         key = key, assay = assay)
}
reductions <- as.list(paste0("reduction", seq_along(modalities)))
dimensions <- lapply(embeddings, function(x) seq_len(ncol(x)))
weight.names <- paste0(modalities, ".weight")
set.seed(settings$seed)
started <- proc.time()[["elapsed"]]
object <- FindMultiModalNeighbors(object, reduction.list = reductions, dims.list = dimensions,
                                  k.nn = settings$neighborCount, knn.range = breadth,
                                  l2.norm = settings$l2Normalize,
                                  modality.weight.name = weight.names, verbose = FALSE)
neighbor.seconds <- proc.time()[["elapsed"]] - started
started <- proc.time()[["elapsed"]]
object <- FindClusters(object, graph.name = "wsnn", algorithm = 4,
                       resolution = settings$resolution, random.seed = settings$seed,
                       leiden_method = "leidenbase", verbose = FALSE)
cluster.seconds <- proc.time()[["elapsed"]] - started
write.table(Indices(object[["weighted.nn"]]), file.path(output, paste0("neighbors_", breadth, ".csv")),
            sep = ",", row.names = FALSE, col.names = FALSE)
write.table(object[[]][, weight.names, drop = FALSE], file.path(output, paste0("weights_", breadth, ".csv")),
            sep = ",", row.names = FALSE, col.names = FALSE)
write.table(as.integer(Idents(object)), file.path(output, paste0("clusters_", breadth, ".csv")),
            sep = ",", row.names = FALSE, col.names = FALSE)
write_json(list(Seurat = as.character(packageVersion("Seurat")),
                SeuratObject = as.character(packageVersion("SeuratObject")),
                R = R.version.string, kNn = settings$neighborCount, knnRange = breadth,
                neighborSeconds = neighbor.seconds, clusteringSeconds = cluster.seconds,
                cells = length(cells), modalities = modalities, threads = threads),
            file.path(output, paste0("measurements_", breadth, ".json")),
            auto_unbox = TRUE, digits = 16, pretty = TRUE)
