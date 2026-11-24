# Harmony2 in R harmony on the inputs that harmony_seed_stability.py --r-reference writes: one assignment step on
# an injected state, and full fits for each arm and seed. Only RunHarmony, harmony_options and the documented
# members of the returned model object (R, Y, O, E, max_iter_kmeans, getZcorr, cluster_cpp) are used.
#
# Usage: Rscript harmony2_reference.R INPUT.json OUTPUT_DIRECTORY
# Matrices pass through little-endian float64 files with one row per cell, or per cluster for centroids.
args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages({
    library(harmony)
    library(jsonlite)
})
input <- fromJSON(args[[1]], simplifyVector = TRUE, simplifyMatrix = TRUE)
output <- args[[2]]
dir.create(output, showWarnings = FALSE, recursive = TRUE)
settings <- input$settings

read_rows <- function(path, rows, cols) {
    matrix(readBin(path, what = "double", n = rows * cols, size = 8, endian = "little"),
           nrow = rows, ncol = cols, byrow = TRUE)
}
write_rows <- function(values, name) {
    writeBin(as.vector(t(values)), file.path(output, name), size = 8, endian = "little")
    name
}
options_for <- function(arm, block_size, max_iter_cluster) {
    off <- isTRUE(arm$disableStopping)
    harmony_options(
        alpha = settings$alpha, tau = settings$tau, block.size = block_size,
        max.iter.cluster = max_iter_cluster,
        epsilon.cluster = if (off) -Inf else arm$epsilonCluster,
        epsilon.harmony = if (off) -Inf else arm$epsilonHarmony,
        batch.prop.cutoff = settings$batchPropCutoff
    )
}
# A member the model object may not expose in this release; NULL when it does not.
optional_member <- function(object, name) {
    value <- tryCatch(do.call("$", list(object, name)), error = function(error) NULL)
    if (is.null(value)) NULL else as.numeric(value)
}

# One clustering pass over a single block that holds every cell: removing the block from O and E leaves the
# injected counts, so the pass assigns each cell against the injected O and E.
assignment_step <- function(name, values, batch, step, theta) {
    set.seed(step$seed)
    object <- RunHarmony(values, data.frame(batch = batch), "batch", theta = theta,
                         sigma = settings$sigma, lambda = NULL, nclust = step$clusters, max_iter = 0,
                         ncores = 1, return_object = TRUE, verbose = FALSE,
                         .options = options_for(list(disableStopping = TRUE), 1, 1))
    first_assignments <- object$R
    first_centroids <- object$Y
    object$E <- object$E + step$expectedInjection
    object$O <- object$O + step$observedInjection
    object$max_iter_kmeans <- 1
    object$cluster_cpp()
    prefix <- sprintf("step_%s_theta%g", name, theta)
    list(
        theta = theta,
        clusters = ncol(first_centroids),
        dims = nrow(first_centroids),
        firstAssignments = write_rows(t(first_assignments), paste0(prefix, "_first_assignments.f64")),
        firstCentroids = write_rows(t(first_centroids), paste0(prefix, "_first_centroids.f64")),
        assignments = write_rows(t(object$R), paste0(prefix, "_assignments.f64")),
        centroids = write_rows(t(object$Y), paste0(prefix, "_centroids.f64"))
    )
}

full_fit <- function(name, values, batch, arm_name, arm, seed) {
    set.seed(seed)
    started <- proc.time()[["elapsed"]]
    object <- RunHarmony(values, data.frame(batch = batch), "batch", theta = arm$theta,
                         sigma = settings$sigma, lambda = NULL, nclust = settings$nclust,
                         max_iter = arm$maxIterHarmony, early_stop = TRUE, ncores = 1,
                         return_object = TRUE, verbose = FALSE,
                         .options = options_for(arm, settings$blockSize, arm$maxIterCluster))
    seconds <- proc.time()[["elapsed"]] - started
    list(
        file = write_rows(t(object$getZcorr()),
                          sprintf("fit_%s_%s_%d.f64", name, arm_name, as.integer(seed))),
        seconds = seconds,
        objectiveHarmony = optional_member(object, "objective_harmony"),
        kmeansRounds = optional_member(object, "kmeans_rounds")
    )
}

result <- list(
    package = "harmony",
    version = as.character(packageVersion("harmony")),
    r = R.version.string,
    blas = tryCatch(sessionInfo()$BLAS, error = function(error) NA_character_),
    lapack = tryCatch(La_library(), error = function(error) NA_character_),
    datasets = list()
)
for (name in names(input$datasets)) {
    dataset <- input$datasets[[name]]
    if (!identical(sort(unique(dataset$batches)), dataset$levels)) {
        stop("The batch levels differ from the order of the injected O and E columns")
    }
    values <- read_rows(dataset$coordinates, dataset$cells, dataset$dims)
    batch <- factor(dataset$batches, levels = dataset$levels)
    step <- dataset$step
    rows <- values[step$rows, , drop = FALSE]
    steps <- lapply(step$thetas, function(theta) {
        assignment_step(name, rows, batch[step$rows], step, theta)
    })
    fits <- list()
    for (arm_name in names(input$arms)) {
        arm_fits <- lapply(input$seeds, function(seed) {
            full_fit(name, values, batch, arm_name, input$arms[[arm_name]], seed)
        })
        names(arm_fits) <- as.character(input$seeds)
        fits[[arm_name]] <- arm_fits
    }
    result$datasets[[name]] <- list(steps = steps, fits = fits)
}
write_json(result, file.path(output, "summary.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA,
           null = "null")
