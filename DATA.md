# Data

## Sources

| Source | What we take | Access | Redistributed here? |
|---|---|---|---|
| AFL Tables (afltables.com) | player-match box scores, final scores, Brownlow votes (home-and-away) | `fitzRoy::fetch_player_stats_afltables` | **No** (terms not verified; `make data` re-fetches) |
| AFLCA Champion Player of the Year leaderboard (aflcoaches.com.au) | coaches' votes per player per match (5-4-3-2-1 from each of two coaches) | `fitzRoy:::scrape_coaches_votes`, one request per round | **No** |
| Squiggle API (api.squiggle.com.au) | final scores, used only to cross-check AFL Tables (`ceb squiggle-check`) | REST, User-Agent with contact, cached | **No** |

Footywire (also reachable through fitzRoy) is not used: one box-score source keeps the join auditable.

**Shipped in the repo:** code (MIT), the data manifest (`results/data_manifest.json`: source, fetch time, row counts, SHA-256 per file), and derived/aggregated outputs (metrics, per-match metric vectors, coefficient tables, figures). Per-player votes and stats are not committed; the Match Review Card demo uses fictional players.

Please cite the sources (AFL Tables, AFLCA, Squiggle, and the fitzRoy package, MIT) when you reuse results.

## Politeness

Every request carries `coaches-eye-bench/0.1 (contact: $CEB_CONTACT_EMAIL)`; the fetch refuses to run without it. AFLCA requests are spaced 1.5 s; everything is cached on disk and a season already on disk is never re-fetched without `--force`.

## fitzRoy pinning and two quirks (verified in the 1.8.0 source)

* The exported `fetch_coaches_votes()` filters out round 24 (`Round > 23 & !Finals`), which is a real home-and-away round in 24-round seasons, and its output has no finals flag. `r/fetch.R` therefore calls the internal scraper per round with `finals = FALSE`, up to the last round present in the box scores.
* `fetch_player_stats_afltables()` reads a cached parquet from the fitzRoy_data repo and scrapes only newer matches; `fetch_meta.json` records fitzRoy/R versions and the fetch time.

## Join (first-class step)

Votes and box scores come from different sites. Inside each match only:

1. **crosswalk** `(season, round, home, away)`; fallback `(season, home, away)` when that pair is unique (round labels disagree); duplicate pages served under a wrong round are dropped and counted;
2. **alias** — curated `src/ceb/resources/name_aliases.csv` rewrites a votes-side key;
3. **exact** — accent/punctuation-free full-name key (handles `Last, First` and `Name (Club)`);
4. **initial** — first initial + last token, accepted only if unique on both sides among the leftovers.

Every vote row ends matched (with its tier) or in `data/processed/unmatched_votes.csv` with a reason (`no_match`, `ambiguous_name`, `ambiguous_initial`, `duplicate_vote_row`, `match_not_in_stats`). A match enters modelling only if its votes total 30 **and** none of its vote rows is unmatched; others go to `excluded_matches.csv`. Matches with no AFLCA data are never treated as zero-vote matches. The pipeline **stops** if the row or vote-mass match rate is below 99% overall or in any season; the remedy is a curated alias, not a looser rule. `join_report.json` lists rates, tiers and exclusions per season.

## Files

`make data` writes `data/raw/*.parquet` (+ `fetch_meta.json`), `data/manifest.json`, `data/ceb.duckdb`, `data/processed/{player_match,join_report,qa_report,sql_invariants}…`. All of `data/` is git-ignored.
