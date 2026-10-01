# Data

## Sources

| Source | What we take | Access | Redistributed here? |
|---|---|---|---|
| AFL Tables (afltables.com) | player-match box scores, final scores, Brownlow votes (home-and-away) | season-scoped pages parsed with pinned fitzRoy helpers | **No** (terms not verified; `make data` re-fetches) |
| AFLCA Champion Player of the Year leaderboard (aflcoaches.com.au) | coaches' votes per player per match (5-4-3-2-1 from each of two coaches) | `fitzRoy:::scrape_coaches_votes`, one request per round | **No** |
| Squiggle API (api.squiggle.com.au) | final scores, used only to cross-check AFL Tables (`ceb squiggle-check`) | REST, User-Agent with contact, cached | **No** |

Footywire (also reachable through fitzRoy) is not used: one box-score source keeps the join auditable.

**Shipped in the repo:** code (MIT), the data manifest (`results/data_manifest.json`: source, fetch time, row counts, SHA-256 per file), and derived/aggregated outputs (metrics, per-match metric vectors, coefficient tables, figures). Per-player votes and stats are not committed; the Match Review Card demo uses fictional players.

Please cite the sources (AFL Tables, AFLCA, Squiggle, and the fitzRoy package, MIT) when you reuse results.

## Politeness

Requests identify the project using `coaches-eye-bench/0.1 (contact: $CEB_CONTACT_EMAIL)` through R's HTTPUserAgent and fitzRoy's user-agent option; the fetch refuses to run without it. AFL Tables match downloads and AFLCA requests are spaced 1.5 s. Match HTML and successful vote rounds are cached; a complete season is never re-fetched without `--force`. Failed rounds are not saved as a complete season. The pinned parser also reads fitzRoy's public player-ID mapping on GitHub; those IDs are for joins only.

## fitzRoy pinning and two quirks (verified in the 1.8.0 source)

* The exported `fetch_coaches_votes()` filters out round 24 (`Round > 23 & !Finals`), which is a real home-and-away round in 24-round seasons, and its output has no finals flag. `r/fetch.R` therefore calls the internal scraper per round with `finals = FALSE`, up to the last round present in the box scores.
* `fetch_player_stats_afltables()` reads an all-seasons parquet **before** filtering. The project therefore uses `get_afltables_urls()` with explicit season dates, caches only those match pages, and parses them with `scrape_afltables_match()`. No all-seasons statistics download is used. `fetch_meta.json` records fitzRoy/R/package versions and the fetch time.
* Vote rounds are the actual numeric rounds in that season's box scores, including Round 0 and Round 24. The source's Opening Round URL mapping still needs live verification in the pilot.

## Setup and pilot

Run `make setup && make setup-r`, set `CEB_CONTACT_EMAIL`, then `make data-pilot`.
The pilot checks 2024–2025 without changing the study's time split or model choices.
Its outputs are local under `data/pilot` and `results/pilot`; the raw cache is shared with `make data`.
Required HTTPS hosts: `cloud.r-project.org` (R dependencies), `afltables.com`,
`aflcoaches.com.au` / `www.aflcoaches.com.au`, and `github.com` / `raw.githubusercontent.com`
(fitzRoy source and player-ID mapping). `api.squiggle.com.au` is optional for score cross-checks.
GitHub release downloads may redirect to `release-assets.githubusercontent.com`.

`make data-test`, direct test-season R fetching and `ceb build --test` all require
an unchanged tagged protocol, study choices in `results/val/frozen_choices.json`,
and no previous test receipt. The pilot cannot satisfy this requirement.

## Join (first-class step)

Votes and box scores come from different sites. Inside each match only:

1. **crosswalk** `(season, round, home, away)`; fallback `(season, home, away)` when that pair is unique (round labels disagree); duplicate pages served under a wrong round are dropped and counted;
2. **alias** — curated `src/ceb/resources/name_aliases.csv` rewrites a votes-side key;
3. **exact** — accent/punctuation-free full-name key (handles `Last, First` and `Name (Club)`);
4. **initial** — first initial + last token, accepted only if unique on both sides among the leftovers.

Every vote row ends matched (with its tier) or in `data/processed/unmatched_votes.csv` with a reason (`no_match`, `ambiguous_name`, `ambiguous_initial`, `duplicate_vote_row`, `match_not_in_stats`). A match enters modelling only if its votes total 30 **and** none of its vote rows is unmatched; others go to `excluded_matches.csv`. Matches with no AFLCA data are never treated as zero-vote matches. The pipeline **stops** if the row or vote-mass match rate is below 99% overall or in any season; the remedy is a curated alias, not a looser rule. `join_report.json` lists rates, tiers and exclusions per season.

## Files

`make data` writes `data/raw/*.parquet` (+ `fetch_meta.json`), `data/manifest.json`, `data/ceb.duckdb`, `data/processed/{player_match,join_report,qa_report,sql_invariants}…`. All of `data/` is git-ignored.
