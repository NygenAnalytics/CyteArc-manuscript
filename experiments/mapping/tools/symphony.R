args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages(library(symphony))
suppressPackageStartupMessages(library(jsonlite))
stopifnot(as.character(packageVersion("symphony")) == "0.1.3")
input <- fromJSON(gzfile(args[[1]]), simplifyMatrix = TRUE)
expression <- t(input$query)
rownames(expression) <- input$featureIds
colnames(expression) <- paste0("query_", seq_len(ncol(expression)))
metadata <- data.frame(study = input$queryBatch, row.names = colnames(expression))
reference <- list(
    vargenes = data.frame(symbol = input$featureIds, mean = input$featureMeans,
                         stddev = input$featureScales),
    loadings = input$loadings,
    centroids = t(input$centroids),
    cache = list(matrix(input$clusterMass, ncol = 1),
                 input$correctedCentroids * input$clusterMass)
)
started <- proc.time()[["elapsed"]]
default_mapping <- mapQuery(expression, metadata, reference, vars = "study",
                           verbose = FALSE, do_normalize = FALSE, do_umap = FALSE,
                           sigma = input$sigma)
scaled <- scaleDataWithStats(expression, input$featureMeans, input$featureScales,
                            margin = 1, thresh = Inf)
projected <- t(input$loadings) %*% scaled
assignments <- symphony:::soft_cluster(reference$centroids,
    symphony:::cosine_normalize_cpp(projected, 2), input$sigma)
corrected <- symphony:::moe_correct_ref(as.matrix(projected), as.matrix(default_mapping$Xq),
    as.matrix(assignments), as.matrix(reference$cache[[1]]), as.matrix(reference$cache[[2]]))
errors <- list(
    projectedMaxAbs = max(abs(t(projected) - input$projected)),
    assignmentsMaxAbs = max(abs(t(assignments) - input$assignments)),
    correctedMaxAbs = max(abs(t(corrected) - input$corrected))
)
passed <- all(vapply(names(errors), function(key) errors[[key]] <= input$tolerances[[key]], logical(1)))
output <- list(package = "symphony", version = as.character(packageVersion("symphony")),
               queryCells = ncol(expression), features = nrow(expression),
               referenceClusters = ncol(reference$centroids),
               elapsedSeconds = proc.time()[["elapsed"]] - started,
               errors = errors, tolerances = input$tolerances, passed = passed,
               comparison = "Shared unclipped reference PCA scaling followed by the compiled R Symphony soft-clustering and query-correction functions",
               scalingThreshold = "Inf",
               defaultMapQuery = list(scalingThreshold = 10,
                   projectedMaxAbsDifference = max(abs(t(default_mapping$Zq_pca) - input$projected)),
                   correctedMaxAbsDifference = max(abs(t(default_mapping$Z) - input$corrected))))
write_json(output, args[[2]], auto_unbox = TRUE, pretty = TRUE, digits = NA)
if (!passed) message("Symphony parity exceeds the fixed tolerances; the measured errors are retained")
