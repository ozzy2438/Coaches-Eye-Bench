# Protocol — Coaches' Eye Bench

Status: **frozen by git tag `protocol-frozen`**. This file is never edited after the tag (see §12).
Written 2026-10-01, before any implementation and before any 2026 data was loaded.

Authorship: the research question, ground truth, split design, model families and primary metric
come from the project brief (human). Every other number below (marked **[default]**) is a default
chosen by the implementing agent so the protocol is complete; they are frozen with everything else
and listed in `DECISIONS.md` so the owner can review them *before* the test season is first loaded.

## 1. Question

When AFL coaches judge who mattered most in a match, how much of that judgement can a transparent
model built from public box-score statistics reproduce, where does it agree, and where do the
statistics systematically miss what coaches see?

Ground truth: AFL Coaches Association (AFLCA) Champion Player of the Year votes. After each
home-and-away match both senior coaches give 5-4-3-2-1 votes, so a player receives 0–10 votes and a
match distributes exactly 30 votes.

## 2. Scope, unit, group

- Competition AFLM, **home-and-away rounds only** (finals carry no AFLCA match votes in this design).
- Unit: player-match (every player with a box-score row in an included match; votes = 0 if no vote row).
- Group: match (both teams ranked together).
- A match is **included** iff AFLCA votes exist for it and sum to exactly 30. Other matches are
  excluded and listed in `data/processed/excluded_matches.csv`; the exclusion count per season is reported.
  Matches without AFLCA data are never treated as zero-vote matches.
- Name matching between sources is deterministic, tiered and audited (see `DATA.md`). Required match
  rate on AFLCA vote rows is >= 99% (and on vote mass); below that the pipeline stops and reports.

## 3. Splits (by time, never random)

| Split | Seasons |
|---|---|
| Train | 2012–2023 |
| Validation | 2024–2025 |
| **Test (locked)** | **2026** |

- **Start-year rule (mechanical, decided now).** The effective first training season is the smallest
  `s` in 2012..2020 such that for every season from `s` to 2025 (a) >= 99% of AFL Tables home-and-away
  matches have a valid vote total (30) and (b) every feature in §4 has >= 95% non-missing values among
  player-match rows. It uses only 2012–2025 data. If `s > 2012` the reason is recorded. If no `s <= 2020`
  qualifies, stop and report (no silent change).
- The 2026 season is not fetched, loaded or inspected until `protocol-frozen` exists; it is evaluated
  **once** (§9). Seasons outside the splits are not used.

## 4. Features

Available to every model that uses features (M1, M2); B0/B1 use only the named columns.

- Box score (per match, as counts): kicks, handballs, marks, contested_marks, contested_possessions,
  uncontested_possessions, goals, behinds, goal_assists, marks_inside_50, inside_50s, tackles,
  hit_outs, clearances, rebounds, clangers, frees_for, frees_against, one_percenters, bounces,
  time_on_ground (% of match played).
- Match context: `won` (1 win, 0.5 draw, 0 loss for the player's team) and `margin_10` (the team's
  signed final margin in points divided by 10).
- Missing `time_on_ground`: filled with the median of the fitting data (no indicator).
- **Views.** `raw`: the 21 stats + 2 context columns. `match_z`: each of the 21 stats standardised
  across all players in the same match (mean/SD within the match; SD = 0 gives 0); context unchanged.
  `global`: the 21 stats standardised with mean/SD of the fitting data; context unchanged.
- **Excluded, always:** player identity or ID, Brownlow votes, any post-match award, anything derived
  from coaches' votes, age, career games (reputation proxies), team identity, venue, round, season.
  Team/ID/name are used only to join, to define roles (§8) and for display.
- No time-on-ground rate normalisation: coaches vote on total contribution, so time on ground enters
  as a feature rather than as a divisor **[default]**.

## 5. Models

| Id | Model | Definition |
|---|---|---|
| B0 | Disposals | score = kicks + handballs |
| B1 | Fantasy points | score = 3·kicks + 2·handballs + 3·marks + 4·tackles + 1·hit_outs + 1·frees_for − 3·frees_against + 6·goals + 1·behinds (AFL Fantasy Classic weights) |
| M1 | Interpretable | L2-penalised proportional-odds (cumulative logit) model; outcome = vote band {0}, {1–3}, {4–6}, {7–10} **[default]**; objective = mean negative log-likelihood + (α/2)·‖β‖², thresholds unpenalised; ranking score = linear predictor η (monotone in expected votes); P(any votes) = 1 − P(band 0). View ∈ {global, match_z}; α ∈ grid below. |
| M2 | Complex | LightGBM LambdaRank grouped by match, label = votes (0–10) with linear `label_gain` matching the metric; features = raw view ∪ match_z view (superset of M1's information). Calibration companion for ECE: LightGBM binary classifier on P(votes > 0) with the same leaves/min-leaf and its own tree count. |

**Selection (validation only).** M1: grid over α × view, choose max mean validation NDCG@5; ties go to
the larger α. M2: grid over `num_leaves` × `min_data_in_leaf`; for each, boosting rounds
∈ {50, 100, …, 1000} are scored on validation with the §6 metric and the best (leaves, min-leaf, rounds)
is chosen; the binary companion picks its rounds by validation log-loss over the same round grid.
Models are fitted on Train only for validation results. **For the test run, M1 and M2 are refitted on
Train + Validation with the frozen choices (rounds not rescaled).** The frozen choices are written to
`results/val/frozen_choices.json` before the test season is touched. Validation figures for M1/M2 are
mildly optimistic because they were selected on the same seasons; the test run is the unbiased figure.

Seeds are fixed (below); the Python environment is pinned by `uv.lock`; the R side by the fitzRoy
version recorded in the data manifest.

## 6. Metrics (per match, then averaged over matches; every match has weight 1)

Gains are the coaches' votes (linear). Let `k = 5`.

- **NDCG@5 (primary).** DCG@5 = Σ gain_i / log2(i+1) over the top 5 by model score, ideal DCG from
  sorting by votes. **Ties in model score are handled tie-aware** (a tie block shares the mean gain of the
  positions it spans), never by arbitrary order — essential because disposals tie often.
- **Top-1 hit.** Share of the model's tied top-scored players holding the match's maximum vote count.
- **Recall@5 of vote-receivers.** (expected number of receivers inside the top 5, tie-aware) /
  min(5, number of receivers) **[default]**, so 1.0 is attainable.
- **Spearman over vote-receivers.** Spearman correlation between model score and votes among players with
  >= 1 vote, per match; matches where it is undefined (< 3 receivers, or constant votes or scores) are
  dropped and their number reported.
- **Calibration.** Expected calibration error of P(votes > 0) for M1 and M2 (companion), 10 equal-width
  bins, pooled over the evaluation seasons.

## 7. Uncertainty and comparisons

- Match-level bootstrap, B = 2000 resamples, 95% percentile intervals, for every metric.
- Paired comparisons on per-match NDCG@5: **M1 vs B0, M1 vs B1, M2 vs M1**. Two-sided sign-flip permutation
  test (20,000 flips) on per-match differences, Holm-adjusted over these three tests (family-wise alpha 0.05);
  95% paired-bootstrap interval of the absolute difference; relative gain = mean(A)/mean(B) − 1 with its
  paired-bootstrap interval. The same comparisons are reported for the secondary metrics, each metric as its
  own Holm family of three and labelled secondary. B1 vs B0 is reported descriptively, without a test.
- **Decision rules (smallest effect of interest δ = 0.01 NDCG@5) [default].**
  - "M2 justifies losing interpretability" only if the Holm-adjusted p < 0.05 **and** the lower bound of the
    interval for M2 − M1 exceeds δ. Otherwise the finding is "not justified" (including "better, but by less than δ").
  - "M1 ≈ B1" is flagged if the whole interval for M1 − B1 lies inside (−δ, +δ).
  - "M1 beats B0 / B1" is claimed only if Holm-adjusted p < 0.05 and the interval for the difference excludes 0.
- All of the above are reported whichever way they fall.

## 8. Q2 — blind spots (validation seasons only, never test)

- Residual: with M1 (selected spec, fitted on Train) compute expected votes E[V|x] per player-match and
  rescale within each match so expected votes sum to 30, matching the fixed vote budget. residual =
  actual votes − rescaled expected votes (positive = the box score under-rates what coaches rewarded).
  Residuals sum to zero within a match. Uncertainty by match-level bootstrap (B = 2000).
- Lenses: role proxy, match closeness, team.
  - Role proxy (descriptive heuristic from box scores, not validated against true positions): per
    player-season from per-game averages (>= 6 games, else `unclassified`), in priority order — Ruck:
    hit_outs >= 8; Midfielder: clearances >= 3.5; Forward: goals >= 1.0 or marks_inside_50 >= 1.5;
    Defender: rebounds >= 2.5 or one_percenters >= 4.0; otherwise Other. Thresholds were set from domain
    knowledge and are not tuned.
  - Closeness: |margin| <= 12 close, 13–39 medium, >= 40 blowout.
  - Team: 18 teams, Holm-adjusted over teams.
- **Hypothesis H2a (to test, not assume):** mean residual of role Defender is > 0. Supported only if the
  95% interval lies above 0. All other lens results are exploratory and labelled so.
- **Coaches vs umpires.** Brownlow votes (3-2-1) are a second target, never a feature. (i) Agreement:
  NDCG@5 against coaches' votes of ranking players by Brownlow votes, versus M1, on validation matches.
  (ii) The selected M1 spec is fitted on coaches' vote bands and on Brownlow bands ({0},{1},{2},{3}) using
  Train + Validation, with a scaler shared by both; coefficients are compared with bootstrap intervals
  (B = 500); a difference is reported only if the two intervals do not overlap.

## 9. Test run (CP4)

One run, with the frozen choices, producing the same table as §6–§7 on 2026 plus the same three headline
numbers. `make eval-test` refuses unless the `protocol-frozen` tag exists, `protocol.md` is unchanged since
the tag, `results/val/frozen_choices.json` exists, and no test result has been recorded. It writes a
`test_run.json` receipt (commit, protocol hash, data manifest hash, frozen-choices hash). Re-running is
refused; any later analysis of 2026 is written under `results/post_hoc/` and labelled "post-hoc".
If 2026 has < 95% valid matches the run stops and reports.

## 10. Q3 — trends (uses seasons <= 2025 only)

- Blocks of 3 consecutive seasons from the effective start year; the remainder forms the last block (a
  remainder of one season is merged into the previous block).
- Fit the selected M1 spec per block (scaler frozen on all pre-test data so coefficients are comparable),
  match-level bootstrap B = 500 per block, 95% percentile intervals.
- A change is reported only if at least two blocks' intervals do not overlap ("clearly separate"); the
  first-vs-last difference additionally gets a Holm-adjusted bootstrap p over all features. Pre-named
  features of interest: contested_possessions, tackles, clearances, marks_inside_50, goals.

## 11. Reporting rules

Report negative and surprising results as plainly as positive ones. Validation numbers are never reported
as test numbers. Any analysis not specified here is labelled post-hoc. No claim of club or AFL
affiliation. The tool is a post-match evaluation lens, not a predictor.

## 12. Freeze and deviation policy

`protocol.md` is edited only before the first load of 2026 data. A re-freeze (moving the tag) must be
logged in `FREEZE_LOG.md` with the reason before any test data is loaded. After that, departures are
recorded in `DEVIATIONS.md` (what, why, effect) and never by editing this file; `make eval-test` checks that
`protocol.md` matches the tag.

## 13. Parameters (machine-readable; the code reads this block and nothing else)

```toml protocol-params
seed = 20261001

[splits]
train_start = 2012
train_end = 2023
val_seasons = [2024, 2025]
test_season = 2026
latest_start_allowed = 2020

[coverage]
min_valid_match_share = 0.99
min_feature_coverage = 0.95
min_test_valid_match_share = 0.95
min_join_rate = 0.99

[votes]
total_per_match = 30
bands = [[0, 0], [1, 3], [4, 6], [7, 10]]
brownlow_bands = [[0, 0], [1, 1], [2, 2], [3, 3]]

[features]
stats = [
  "kicks", "handballs", "marks", "contested_marks", "contested_possessions",
  "uncontested_possessions", "goals", "behinds", "goal_assists", "marks_inside_50",
  "inside_50s", "tackles", "hit_outs", "clearances", "rebounds", "clangers",
  "frees_for", "frees_against", "one_percenters", "bounces", "time_on_ground",
]
context = ["won", "margin_10"]

[fantasy]
kicks = 3.0
handballs = 2.0
marks = 3.0
tackles = 4.0
hit_outs = 1.0
frees_for = 1.0
frees_against = -3.0
goals = 6.0
behinds = 1.0

[metrics]
k = 5
ece_bins = 10
min_receivers_spearman = 3

[inference]
bootstrap_B = 2000
permutation_n = 20000
alpha = 0.05
sesoi_ndcg5 = 0.01

[m1]
alpha_grid = [0.0001, 0.001, 0.01, 0.1]
view_grid = ["global", "match_z"]

[m2]
num_leaves_grid = [7, 15, 31]
min_data_in_leaf_grid = [50, 200]
rounds_grid = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
learning_rate = 0.05
feature_fraction = 0.8

[q2]
close_margin_max = 12
blowout_margin_min = 40
min_games_for_role = 6
bootstrap_B = 2000
brownlow_bootstrap_B = 500

[roles]
ruck_hit_outs = 8.0
midfielder_clearances = 3.5
forward_goals = 1.0
forward_marks_inside_50 = 1.5
defender_rebounds = 2.5
defender_one_percenters = 4.0

[trends]
block_width = 3
bootstrap_B = 500
last_season_included = 2025
highlight = ["contested_possessions", "tackles", "clearances", "marks_inside_50", "goals"]
```
