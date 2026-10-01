#!/usr/bin/env Rscript
# Install the fetch dependencies into a project-local library. Requires R >= 4.1.
if (getRversion() < "4.1") stop("R >= 4.1 is required")
lib <- Sys.getenv("R_LIBS_USER", ".r-library")
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(normalizePath(lib), .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"), timeout = 300)
source("r/fetch.R")
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes", lib = lib)
if (!requireNamespace("remotes", quietly = TRUE)) {
  stop("could not install remotes; check HTTPS access to cloud.r-project.org before retrying")
}
if (!requireNamespace("fitzRoy", quietly = TRUE) ||
    as.character(packageVersion("fitzRoy")) != PINNED_FITZROY) {
  remotes::install_version("fitzRoy", version = PINNED_FITZROY, lib = lib,
                           dependencies = NA, upgrade = "never")
}
for (pkg in c("fitzRoy", "nanoparquet", "jsonlite")) {
  if (!requireNamespace(pkg, quietly = TRUE)) stop("missing dependency: ", pkg)
}
stopifnot(as.character(packageVersion("fitzRoy")) == PINNED_FITZROY)
# Record the complete installed dependency versions for the fetch manifest.
versions <- as.data.frame(installed.packages()[, c("Package", "Version")])
write.csv(versions, file.path(lib, "installed-versions.csv"), row.names = FALSE)
cat("R fetch dependencies ready\n")
