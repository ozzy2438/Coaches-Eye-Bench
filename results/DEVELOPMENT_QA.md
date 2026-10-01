# Development data quality (2012–2025)

Generated 2026-10-01T09:57:24Z from the completed `make data` audit files; command exit 0.

Effective training start: 2012. Notes: none. Included: 2752 matches / 123122 player-matches. All unchanged protocol gates pass. No 2026 data or held-out outputs exist.

| Season | Vote rows joined | Vote mass joined | Included / source matches | Included % | Valid-vote % | Processed feature min % | Raw feature min % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2012 | 1325/1325 (100.0000%) | 5940/5940 (100.0000%) | 198/198 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| 2013 | 1356/1356 (100.0000%) | 5940/5940 (100.0000%) | 198/198 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| 2014 | 1315/1315 (100.0000%) | 5940/5940 (100.0000%) | 198/198 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| 2015 | 1319/1319 (100.0000%) | 5910/5910 (100.0000%) | 197/197 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| 2016 | 1296/1296 (100.0000%) | 5940/5940 (100.0000%) | 198/198 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| 2017 | 1311/1311 (100.0000%) | 5940/5940 (100.0000%) | 198/198 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| 2018 | 1305/1311 (99.5423%) | 5910/5940 (99.4949%) | 197/198 | 99.4949 | 99.4949 | 100.0000 | 100.0000 |
| 2019 | 1300/1300 (100.0000%) | 5940/5940 (100.0000%) | 198/198 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| 2020 | 1025/1025 (100.0000%) | 4590/4590 (100.0000%) | 153/153 | 100.0000 | 100.0000 | 100.0000 | 96.7320 |
| 2021 | 1306/1306 (100.0000%) | 5940/5940 (100.0000%) | 198/198 | 100.0000 | 100.0000 | 100.0000 | 99.4949 |
| 2022 | 1303/1303 (100.0000%) | 5940/5940 (100.0000%) | 198/198 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| 2023 | 1361/1361 (100.0000%) | 6210/6210 (100.0000%) | 207/207 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| 2024 | 1342/1342 (100.0000%) | 6210/6210 (100.0000%) | 207/207 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| 2025 | 1343/1343 (100.0000%) | 6210/6210 (100.0000%) | 207/207 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |

The join floor is 99% on rows and vote mass, overall and separately for each season. The mechanical start rule requires valid-vote share ≥99% and feature coverage ≥95%. Valid-vote share counts votes present and totalling 30; included share also reflects name-match exclusions. These denominators are deliberately reported separately. Integer-stat missing values are filled with zero by the existing ingestion rule; raw coverage above makes that visible.

Crosswalk audit: 23 vote rows matched through an exact season/round and unique reversed fixture; 12 rows use the existing unique ordered season-pair round fallback; 0 duplicate-page rows dropped.

The first complete data attempt stopped at the 2023 gate: 1338/1361 rows (98.3101%) and 6090/6210 vote mass (98.0676%). Four 2023 R5 neutral-site fixtures have opposite home/away orders between the sources. A documented exact same-season/round reversal step, requiring a unique unordered fixture, repairs those identities. Player-name tiers, features, thresholds and the tagged protocol are unchanged. A reversed pair with a wrong round is still rejected. The retry uses all cached source files. [AFLCA 2023 R5 source](https://aflcoaches.com.au/awards/the-aflca-champion-player-of-the-year-award/leaderboard/2023/20240105).

## Reviewed pilot exclusions

The original pilot excluded 2024 R4 and 2025 R24, Western Bulldogs–West Coast. Both clubs field a Bailey Williams (stats IDs 12441 for Western Bulldogs and 12836 for West Coast). AFLCA awarded the Bulldogs player 7 and 2 votes respectively, but the pilot did not recognize `(WB)`, so the conservative same-name rule rejected the entire match. Explicit club-code mapping repairs this under the existing exact-name/club-hint rule; both study validation seasons now retain 207/207 matches. The original pilot data, reports and model choices were retained.

| Original pilot season | Row join % | Vote-mass join % | Included matches | Valid-vote % |
|---|---:|---:|---:|---:|
| 2024 | 99.9255 | 99.8873 | 206/207 (99.5169%) | 100 |
| 2025 | 99.9255 | 99.9678 | 206/207 (99.5169%) | 100 |

## Remaining source exclusions

| match_id                      |   season |   round | reason             | detail   |
|:------------------------------|---------:|--------:|:-------------------|:---------|
| 2018-R16-essendon-collingwood |     2018 |      16 | no_votes_for_match | —        |

2018 R16 Collingwood–Essendon: the AFLCA page has two Collingwood club labels/logos. Its six vote rows (30 votes) identify a nonexistent Collingwood–Collingwood fixture; the box-score match therefore has no correctly identified vote table. The existing rules exclude it and the 99% gates still pass. No fixture correction was guessed. [AFLCA source](https://aflcoaches.com.au/awards/the-aflca-champion-player-of-the-year-award/leaderboard/2018/20180116).

## Reproducibility and isolation

Protocol SHA-256: `e910f956435a4455f84bded4d99f3f3ad1ca6e369ea127045423d5862b0d40af` (unchanged). Pilot choice SHA-256 is unchanged. All four raw 2024–2025 file hashes match the pilot manifest, confirming reuse of the original cache. All 49 R package versions match `renv.lock`; an independent empty-library restore exited 0 under R 4.5.0. No new test files were added.

Python QA passed; independent DuckDB checks found zero wrong vote totals, duplicate player-matches or negative count-stat rows. The detailed JSON includes each season’s QA and raw missing-feature coverage. Quality gates and the tagged protocol were not changed.
