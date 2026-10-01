#!/usr/bin/env Rscript
# Restore the complete fetch dependency graph from renv.lock into a local library.
if (getRversion() < "4.1") stop("R >= 4.1 is required")
lib <- Sys.getenv("R_LIBS_USER", ".r-library")
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(normalizePath(lib), .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"), timeout = 300)
source("r/fetch.R")
if (!file.exists("renv.lock")) stop("renv.lock is required; dependency versions must not drift")
# Pin the bootstrap as well: do not install the latest renv from CRAN.
renv_version <- "1.3.0"
present <- installed.packages(lib.loc = lib)
if (!"renv" %in% rownames(present) || present["renv", "Version"] != renv_version) {
  tarball <- tempfile(fileext = ".tar.gz")
  filename <- sprintf("renv_%s.tar.gz", renv_version)
  urls <- c(paste0("https://cloud.r-project.org/src/contrib/", filename),
            paste0("https://cloud.r-project.org/src/contrib/Archive/renv/", filename))
  downloaded <- FALSE
  for (url in urls) {
    status <- tryCatch(download.file(url, tarball, mode = "wb", quiet = TRUE),
                       error = function(e) 1L)
    if (status == 0L) { downloaded <- TRUE; break }
  }
  if (!downloaded) stop("could not download pinned renv from CRAN")
  install.packages(tarball, repos = NULL, type = "source", lib = lib)
  unlink(tarball)
}
if (!requireNamespace("renv", quietly = TRUE) ||
    as.character(packageVersion("renv")) != renv_version) stop("pinned renv bootstrap failed")
lock <- renv::lockfile_read("renv.lock")
stopifnot(lock$Packages$renv$Version == renv_version,
          lock$Packages$fitzRoy$Version == PINNED_FITZROY)
renv::restore(project = getwd(), library = lib, lockfile = "renv.lock", prompt = FALSE)
installed <- installed.packages(lib.loc = lib)
for (pkg in names(lock$Packages)) {
  if (!pkg %in% rownames(installed) || installed[pkg, "Version"] != lock$Packages[[pkg]]$Version) {
    stop("locked dependency was not restored: ", pkg)
  }
}
for (pkg in c("fitzRoy", "nanoparquet", "jsonlite")) {
  if (!requireNamespace(pkg, quietly = TRUE)) stop("missing dependency: ", pkg)
}
stopifnot(as.character(packageVersion("fitzRoy")) == PINNED_FITZROY)
# Record the complete installed dependency versions for the fetch manifest.
versions <- as.data.frame(installed.packages()[, c("Package", "Version")])
write.csv(versions, file.path(lib, "installed-versions.csv"), row.names = FALSE)
cat("R fetch dependencies ready (", length(lock$Packages), " locked packages)\n", sep = "")
