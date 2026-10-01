#!/usr/bin/env Rscript
# Fetch public AFL data with fitzRoy and write ONE parquet per season and source.
#
#   Rscript r/fetch.R --seasons 2012:2025 [--out data/raw] [--force] [--allow-missing-rounds]
#
# - AFL Tables player-match stats  : season-scoped pages, parsed by pinned fitzRoy helpers
# - AFLCA coaches' votes            : fitzRoy:::scrape_coaches_votes, one request per round, finals = FALSE
#   (the exported fetch_coaches_votes() skips round 24 of 24-round seasons and drops the finals flag,
#    so the internal scraper is called directly; fitzRoy is pinned to avoid silent API drift)
# - Politeness: contact User-Agent, 1.5 s between AFLCA requests, everything cached on disk.
# - The locked test season requires an unchanged tagged protocol, frozen choices and no prior test receipt.

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

season_rounds <- function(df) sort(unique(as.integer(home_and_away_only(df)$Round)))

# The exported fetch_player_stats_afltables() loads an ALL-SEASONS parquet before
# filtering. Never use it here: even a development fetch would open the test data.
fetch_stats_season <- function(season, cache, force = FALSE) {
  dir.create(cache, recursive = TRUE, showWarnings = FALSE)
  index <- file.path(cache, "match_urls.rds")
  if (force || !file.exists(index)) {
    urls <- fitzRoy:::get_afltables_urls(sprintf("%d-01-01", season),
                                        sprintf("%d-12-31", season))
    if (!length(urls)) stop("no AFL Tables match URLs for ", season)
    saveRDS(urls, index)
  }
  urls <- readRDS(index)
  pages <- character(length(urls))
  for (i in seq_along(urls)) {
    pages[i] <- file.path(cache, basename(urls[i]))
    if (force || !file.exists(pages[i])) {
      tmp <- paste0(pages[i], ".part")
      tryCatch({
        utils::download.file(urls[i], tmp, quiet = TRUE, mode = "wb")
        if (!file.rename(tmp, pages[i])) stop("could not cache ", urls[i])
      }, finally = { unlink(tmp); Sys.sleep(1.5) })
    }
  }
  stats <- fitzRoy:::scrape_afltables_match(pages)
  # `dictionary_afltables` is LazyData in fitzRoy's data/, not in its namespace: `:::` cannot see it.
  stats <- fitzRoy:::check_and_convert(stats, fitzRoy::dictionary_afltables)
  check_columns(stats)
  if (anyNA(stats$Season) || any(stats$Season != season)) stop("unexpected season in AFL Tables response")
  home_and_away_only(stats)
}

# Rounds that returned no AFLCA data stop the season unless explicitly allowed; allowed gaps are
# recorded in fetch_meta.json and the affected matches are then excluded by the Python join
# (no_votes_for_match), where the protocol's >= 99% valid-match floor decides.
check_missing_rounds <- function(season, miss, allow = FALSE) {
  if (length(miss) && !allow) {
    stop("AFLCA rounds missing for ", season, ": ", paste(miss, collapse = ","),
         "; successful rounds cached; rerun to retry, or pass --allow-missing-rounds to record the gap")
  }
  as.integer(miss)
}

read_test_season <- function(protocol = "protocol.md") {
  if (!file.exists(protocol)) return(NA_integer_)
  m <- regmatches(readLines(protocol, warn = FALSE), regexpr("^test_season = [0-9]+", readLines(protocol, warn = FALSE)))
  if (length(m) == 0) return(NA_integer_)
  as.integer(sub("^test_season = ", "", m[1]))
}

main <- function(args = commandArgs(trailingOnly = TRUE)) {
  get_arg <- function(flag, default = NULL) {
    i <- match(flag, args)
    if (is.na(i) || i == length(args)) default else args[i + 1]
  }
  seasons <- parse_seasons(get_arg("--seasons"))
  out <- get_arg("--out", "data/raw")
  force <- "--force" %in% args
  allow_missing <- "--allow-missing-rounds" %in% args
  dir.create(out, recursive = TRUE, showWarnings = FALSE)

  contact <- Sys.getenv("CEB_CONTACT_EMAIL", "")
  if (!nzchar(contact)) stop("set CEB_CONTACT_EMAIL (used in the User-Agent of every request)")

  test_season <- read_test_season()
  if (is.na(test_season)) stop("cannot read the locked season from protocol.md")
  if (any(seasons > test_season)) stop("seasons beyond the protocol are not allowed")
  if (test_season %in% seasons) {
    status <- system2("uv", c("run", "--frozen", "ceb", "guard", "--test"))
    if (status != 0L) stop("test-season guard failed; no test data fetched")
  }

  suppressPackageStartupMessages({
    library(fitzRoy)
    library(nanoparquet)
  })
  fitzroy_version <- as.character(utils::packageVersion("fitzRoy"))
  if (fitzroy_version != PINNED_FITZROY) {
    stop("fitzRoy ", fitzroy_version, " installed; run `make setup-r` for ", PINNED_FITZROY)
  }
  user_agent <- sprintf("coaches-eye-bench/0.1 (contact: %s)", contact)
  options(fitzRoy.user_agent = user_agent, HTTPUserAgent = user_agent)

  stats_path <- function(s) file.path(out, sprintf("afltables_player_stats_%d.parquet", s))
  votes_path <- function(s) file.path(out, sprintf("aflca_coaches_votes_%d.parquet", s))

  # ---- box scores: fetch only explicitly requested seasons -------------------------
  todo <- Filter(function(s) force || !file.exists(stats_path(s)), seasons)
  if (length(todo) > 0) {
    message("AFL Tables: fetching seasons ", paste(range(todo), collapse = "-"))
    for (s in todo) {
      d <- fetch_stats_season(s, file.path(out, "pages", as.character(s)), force)
      d <- d[, STATS_COLUMNS]
      d$Round <- as.character(d$Round)
      d$Date <- as.character(d$Date)
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
    rounds <- season_rounds(sp)
    round_cache <- file.path(out, "votes_by_round", as.character(s))
    dir.create(round_cache, recursive = TRUE, showWarnings = FALSE)
    parts <- list()
    miss <- integer(0)
    for (r in rounds) {
      cached <- file.path(round_cache, sprintf("round_%02d.rds", r))
      if (!force && file.exists(cached)) {
        res <- readRDS(cached)
      } else {
        res <- tryCatch(
          fitzRoy:::scrape_coaches_votes(season = s, round_number = r, comp = "AFLM", finals = FALSE),
          finally = Sys.sleep(1.5))
        if (is.data.frame(res) && nrow(res) > 0) saveRDS(res, cached)
      }
      if (inherits(res, "try-error") || !is.data.frame(res) || nrow(res) == 0) {
        miss <- c(miss, r)
      } else {
        parts[[length(parts) + 1]] <- res
      }
    }
    if (length(parts) == 0) stop("no AFLCA votes retrieved for ", s)
    miss <- check_missing_rounds(s, miss, allow_missing)
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
  old$fitzRoy_version <- fitzroy_version
  old$r_packages <- as.list(setNames(as.character(utils::installed.packages()[, "Version"]),
                                    utils::installed.packages()[, "Package"]))
  old$r_version <- R.version.string
  old$last_fetch_utc <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  old$seasons_last_call <- seasons
  old$missing_vote_rounds <- modifyList(if (is.null(old$missing_vote_rounds)) list() else old$missing_vote_rounds,
                                        missing_rounds)
  jsonlite::write_json(old, meta_path, auto_unbox = TRUE, pretty = TRUE)
  invisible(TRUE)
}

if (sys.nframe() == 0L) main()
