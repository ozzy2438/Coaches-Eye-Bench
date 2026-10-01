#!/usr/bin/env Rscript
# Fetch public AFL data with fitzRoy and write ONE parquet per season and source.
#
#   Rscript r/fetch.R --seasons 2012:2025 [--out data/raw] [--force]
#
# - AFL Tables player-match stats  : fitzRoy::fetch_player_stats_afltables (home-and-away rows only)
# - AFLCA coaches' votes            : fitzRoy:::scrape_coaches_votes, one request per round, finals = FALSE
#   (the exported fetch_coaches_votes() skips round 24 of 24-round seasons and drops the finals flag,
#    so the internal scraper is called directly; fitzRoy is pinned to avoid silent API drift)
# - Politeness: contact User-Agent, 1.5 s between AFLCA requests, everything cached on disk.
# - The locked test season (read from protocol.md) is only fetched once git tag `protocol-frozen` exists.

PINNED_FITZROY <- "1.8.0"

STATS_COLUMNS <- c(
  "Season", "Round", "Date", "Home.team", "Away.team", "Home.score", "Away.score", "Playing.for",
  "First.name", "Surname", "ID", "Kicks", "Handballs", "Disposals", "Marks", "Contested.Marks",
  "Contested.Possessions", "Uncontested.Possessions", "Goals", "Behinds", "Goal.Assists",
  "Marks.Inside.50", "Inside.50s", "Tackles", "Hit.Outs", "Clearances", "Rebounds", "Clangers",
  "Frees.For", "Frees.Against", "One.Percenters", "Bounces", "Time.on.Ground", "Brownlow.Votes"
)

parse_seasons <- function(x) {
  if (is.null(x) || !nzchar(x)) stop("--seasons is required, e.g. 2012:2025 or 2026")
  if (grepl(":", x, fixed = TRUE)) {
    p <- as.integer(strsplit(x, ":", fixed = TRUE)[[1]])
    if (length(p) != 2 || anyNA(p) || p[1] > p[2]) stop("bad --seasons range: ", x)
    seq(p[1], p[2])
  } else {
    s <- as.integer(strsplit(x, ",", fixed = TRUE)[[1]])
    if (anyNA(s)) stop("bad --seasons list: ", x)
    s
  }
}

check_columns <- function(df, needed = STATS_COLUMNS) {
  missing <- setdiff(needed, names(df))
  if (length(missing) > 0) stop("AFL Tables table lacks columns: ", paste(missing, collapse = ", "))
  invisible(TRUE)
}

home_and_away_only <- function(df) df[grepl("^[0-9]+$", as.character(df$Round)), , drop = FALSE]

read_test_season <- function(protocol = "protocol.md") {
  if (!file.exists(protocol)) return(NA_integer_)
  m <- regmatches(readLines(protocol, warn = FALSE), regexpr("^test_season = [0-9]+", readLines(protocol, warn = FALSE)))
  if (length(m) == 0) return(NA_integer_)
  as.integer(sub("^test_season = ", "", m[1]))
}

tag_exists <- function(tag = "protocol-frozen") {
  suppressWarnings(system2("git", c("rev-parse", "-q", "--verify", paste0("refs/tags/", tag)),
                           stdout = FALSE, stderr = FALSE)) == 0L
}

main <- function(args = commandArgs(trailingOnly = TRUE)) {
  get_arg <- function(flag, default = NULL) {
    i <- match(flag, args)
    if (is.na(i) || i == length(args)) default else args[i + 1]
  }
  seasons <- parse_seasons(get_arg("--seasons"))
  out <- get_arg("--out", "data/raw")
  force <- "--force" %in% args
  dir.create(out, recursive = TRUE, showWarnings = FALSE)

  contact <- Sys.getenv("CEB_CONTACT_EMAIL", "")
  if (!nzchar(contact)) stop("set CEB_CONTACT_EMAIL (used in the User-Agent of every request)")

  test_season <- read_test_season()
  if (!is.na(test_season) && test_season %in% seasons && !tag_exists()) {
    stop("season ", test_season, " is the locked test season: tag `protocol-frozen` must exist first")
  }

  suppressPackageStartupMessages({
    library(fitzRoy)
    library(nanoparquet)
  })
  v <- as.character(utils::packageVersion("fitzRoy"))
  if (v != PINNED_FITZROY && !nzchar(Sys.getenv("CEB_ALLOW_FITZROY_MISMATCH"))) {
    stop("fitzRoy ", v, " installed, ", PINNED_FITZROY, " pinned (set CEB_ALLOW_FITZROY_MISMATCH=1 to override and record it)")
  }
  options(fitzRoy.user_agent = sprintf("coaches-eye-bench/0.1 (contact: %s)", contact))

  stats_path <- function(s) file.path(out, sprintf("afltables_player_stats_%d.parquet", s))
  votes_path <- function(s) file.path(out, sprintf("aflca_coaches_votes_%d.parquet", s))

  # ---- box scores: one call for all uncached seasons, split by season -------------------------
  todo <- Filter(function(s) force || !file.exists(stats_path(s)), seasons)
  if (length(todo) > 0) {
    message("AFL Tables: fetching seasons ", paste(range(todo), collapse = "-"))
    st <- fitzRoy::fetch_player_stats_afltables(season = todo)
    check_columns(st)
    st <- home_and_away_only(st[, STATS_COLUMNS])
    st$Round <- as.character(st$Round)
    st$Date <- as.character(st$Date)
    for (s in todo) {
      d <- st[st$Season == s, , drop = FALSE]
      if (nrow(d) == 0) stop("AFL Tables returned no home-and-away rows for ", s)
      nanoparquet::write_parquet(d, stats_path(s))
      message("  ", s, ": ", nrow(d), " player-match rows")
    }
  }

  # ---- AFLCA coaches' votes: one request per home-and-away round ------------------------------
  missing_rounds <- list()
  for (s in seasons) {
    if (!force && file.exists(votes_path(s))) next
    sp <- nanoparquet::read_parquet(stats_path(s))
    max_round <- max(as.integer(sp$Round))
    parts <- list()
    miss <- integer(0)
    for (r in seq_len(max_round)) {
      res <- try(fitzRoy:::scrape_coaches_votes(season = s, round_number = r, comp = "AFLM", finals = FALSE),
                 silent = TRUE)
      if (inherits(res, "try-error") || !is.data.frame(res) || nrow(res) == 0) {
        miss <- c(miss, r)
      } else {
        parts[[length(parts) + 1]] <- res
      }
      Sys.sleep(1.5)
    }
    if (length(parts) == 0) stop("no AFLCA votes retrieved for ", s)
    v <- do.call(rbind, parts)
    v$Season <- as.integer(v$Season)
    v$Round <- as.integer(v$Round)
    v$Coaches.Votes <- as.character(v$Coaches.Votes)
    nanoparquet::write_parquet(v, votes_path(s))
    missing_rounds[[as.character(s)]] <- miss
    message("AFLCA ", s, ": ", nrow(v), " vote rows; rounds without data: ",
            if (length(miss)) paste(miss, collapse = ",") else "none")
  }

  meta_path <- file.path(out, "fetch_meta.json")
  old <- if (file.exists(meta_path)) jsonlite::read_json(meta_path) else list()
  old$fitzRoy_version <- v
  old$r_version <- R.version.string
  old$last_fetch_utc <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  old$seasons_last_call <- seasons
  old$missing_vote_rounds <- modifyList(if (is.null(old$missing_vote_rounds)) list() else old$missing_vote_rounds,
                                        missing_rounds)
  jsonlite::write_json(old, meta_path, auto_unbox = TRUE, pretty = TRUE)
  invisible(TRUE)
}

if (sys.nframe() == 0L) main()
