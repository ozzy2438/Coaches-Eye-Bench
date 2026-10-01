# Decisions and judgement calls

Written as they were made. "Owner review" marks agent-chosen defaults frozen in `protocol.md`; the owner can change them **before the test season is first loaded** by editing `protocol.md`, moving the `protocol-frozen` tag and logging it in `FREEZE_LOG.md`.

## A. Blocker: the live data sources were unreachable from the build sandbox

The agent's sandbox egress policy returned 403 for `afltables.com`, `www.footywire.com`, `api.squiggle.com.au`, `cran.r-project.org`, `cloud.r-project.org`, `www.aflcoaches.com.au`, `api.afl.com.au`. Only GitHub raw/API, PyPI and the Ubuntu archive were reachable. The agent did not route around the policy (no third-party scraping proxies). Consequences:

* Everything that does not need live data was built and verified (see README).
* The real-data steps (CP1–CP4: `make data`, `make train`, `make eval-val`, `make eval-test`) have **not been run**; no real number exists in this repository.
* `r/fetch.R` was written against the fitzRoy 1.8.0 *source* (read-only clone) and parse-checked with R 4.3.3, its pure helpers are tested, but it has **never executed against the live sites**. Treat the first `make data` as the integration test; if the AFLCA page layout or `scrape_coaches_votes` signature differ, the symptom will be a stop at the join-rate or QA gate, not silent bad data.

**Repository side effects of the same sandbox:** pushing the branch worked; pushing the `protocol-frozen` tag returned 403 and the GitHub App had no `workflows` scope, so the CI workflow is shipped as `ci/github-actions-ci.yml` (with `make install-ci`) and the tag must be re-created and pushed by the owner (`make restore-tag`). CI has therefore not run on GitHub; the same lint/test/smoke commands were run locally, including from a clean clone with `uv sync --frozen`.

## B. Things found in the sources that shaped the design

1. `fitzRoy::fetch_coaches_votes()` drops round 24 of 24-round seasons (`Round > 23 & !Finals`) and returns no finals flag → `r/fetch.R` calls `scrape_coaches_votes(finals = FALSE)` per round, up to the last box-score round.
2. AFLCA vote tables carry no team per player → join inside the match on names; club hints used when present; ambiguous names exclude the match rather than guess.
3. The AFLCA vote total per match is 30 (checked in fitzRoy's own `calculate_coaches_vote_possibilities`), ~7 receivers per match (fitzRoy test: 64 rows for a 9-match round).

## C. Agent-chosen defaults frozen in the protocol (owner review)

| Choice | Value | Why |
|---|---|---|
| Vote bands for M1 | {0}, {1–3}, {4–6}, {7–10} | ≈84% of player-matches are 0 (≈7 receivers of 44, inferred from fitzRoy's own test fixture, not measured here); finer bands are sparse at the top; four bands keep thresholds estimable |
| Smallest effect of interest δ | 0.01 NDCG@5 | about one point on a 0–1 scale; below it a coach would not notice |
| Recall@5 | divided by min(5, receivers) | the literal definition caps at 5/receivers < 1 |
| Ties | tie-aware NDCG / Recall / Top-1 | disposals tie constantly; arbitrary row order would reward or punish baselines at random |
| M1 grid | α ∈ {1e-4,1e-3,1e-2,1e-1} × view ∈ {global, match_z} | small enough not to over-tune on two validation seasons |
| M2 grid | leaves {7,15,31} × min-leaf {50,200} × rounds 50…1000 | rounds scored with the protocol metric, not LightGBM's tie-arbitrary one |
| Selection tie-breaks | larger α (M1); fewer rounds, then fewer leaves (M2) | prefer the simpler model on exact ties |
| Features | no age, career games, team, venue | reputation/identity proxies; `uncontested_possessions` kept (≠ disposals − contested possessions) |
| TOG | a feature (median-filled), not a divisor | coaches reward total contribution, a divisor would reward cameos |
| Role thresholds | hit-outs ≥ 8, clearances ≥ 3.5, goals ≥ 1 or marks-i50 ≥ 1.5, rebounds ≥ 2.5 or 1%ers ≥ 4 | domain knowledge, not tuned; they are descriptive only |
| Closeness | ≤ 12 close, ≥ 40 blowout | two goals / seven goals |
| Q2 residual | expected votes rescaled to 30 per match | votes are a zero-sum budget; residuals sum to 0 per match |
| Trend blocks | width 3, remainder merged if single | five blocks over 2012–2025 |
| M2 threads | 1 | bit-for-bit reproducible across machines; the full grid is expected to take tens of minutes (extrapolated from synthetic timing, not measured on real data) |
| Refit for test | Train + Validation, frozen choices, rounds not rescaled | uses all pre-test information |

## D. Implementation clarifications (protocol silent; not deviations)

* **Bootstrap p-values** (trend first-vs-last, team residuals, role residuals) use `z = estimate / bootstrap SE`. The percentile tail share cannot go below 1/(B+1), which after Holm over 23 features made even an overwhelming synthetic effect "non-significant" (p_holm 0.38 at B = 60; floor ≈ 0.046 at B = 500). CIs remain percentile intervals.
* **Join strictness:** the ≥ 99% floor is applied to the total *and* each season, on rows *and* vote mass. A match with any unmatched vote row is excluded (its labels would be wrong), not modelled.
* **Start-year rule** uses valid-vote share = 1 − (no-votes + wrong-total) / matches; unmatched-row exclusions are policed by the join-rate gate instead.
* **Dev vs test separation** goes beyond the protocol text: the dev manifest and `make data` never open 2026 files, `make data-test` and `r/fetch.R` check the tag, and a single-run receipt is written.
* **Smoke** is synthetic and uses three tiny seasons, not the brief's "1 season": the split logic needs train/val/test seasons, and CI must not hit the live sites on every commit. `make smoke-real` covers real data once fetched.
* **Squiggle** is used only to cross-check final scores (`ceb squiggle-check`). The contact e-mail comes from `CEB_CONTACT_EMAIL`; nothing personal is hard-coded or committed.
* **Raw data is not redistributed** — the sources' terms were not verified from the sandbox.
* AFLW replication (stretch) was not attempted: it needs another stats source and the unverified AFLCA 2022 double-season handling.

## E. Post-hoc additions (labelled as such in outputs)

* **M2-residual lens for Q2.** The protocol's M1 residual conflates "stats miss it" with "M1 is linear". With non-linearity planted, M1 residuals pile on midfielders and push other roles negative. Residuals against LightGBM expected votes (`q2_posthoc_m2_*.csv`) are reported next to the protocol analysis, never instead of it.
* **H2a sensitivity** (`scripts/h2a_sensitivity.py`, synthetic, committed as `results/synthetic_h2a_sensitivity.json`):

| Scenario | M1 residual (protocol) | M2 residual (post-hoc) |
|---|---|---|
| no planted effects | −0.017 [−0.039, +0.007] | −0.018 [−0.040, +0.006] |
| defender premium 0.8 | +0.033 [−0.000, +0.067] | +0.025 [−0.012, +0.063] |
| premium 2.0 + non-linearity | +0.005 [−0.025, +0.034] | +0.007 [−0.021, +0.035] |

H2a was "supported" in none of them. Read a real-data "not supported" as *this lens cannot tell*, not as *no blind spot*.

## F. What to review before the first `make data-test`

Everything in table C and the Q2 lens caveat in E. Independent position labels would improve role validity but cannot by themselves prove missing defensive work.

## G. Readiness review before any real-data run (2026-10-01)

The owner requested implementation of the readiness review without new test files.
The original protocol and its numerical choices remain unchanged; no 2026 data was fetched.

* **Test isolation:** fitzRoy 1.8.0 source inspection showed the exported statistics fetch loads an all-seasons parquet before filtering. Replaced that route with explicitly dated season pages parsed by pinned fitzRoy helpers. This enforces the original protocol rather than changing the design. Live parsing remains an integration check, not a claimed success.
* **Fetch corrections:** include numeric Round 0; keep the fitzRoy version separate from vote tables; retain successful round/page caches after a failure; require the full test guard for direct R fetch and CLI test builds.
* **Pilot:** `make data-pilot` limits ingestion to 2024–2025. It shares raw caches but isolates processed data, manifests, reduced smoke outputs and choices under `data/pilot` / `results/pilot`. The final study still trains on the protocol's training seasons. Quality-gate failures retain audit CSVs.
* **Metric naming:** display `recall5` as Normalized Recall@5, explicitly defining the denominator `min(5, receivers)`. Calculation unchanged.
* **Decision threshold:** retain δ = 0.01 as a research convention. The earlier claim that a coach would not notice a smaller difference was unsupported. Do not equate the statistical rule with demonstrated usability or adoption value. A lower confidence bound below δ is insufficient evidence that the gain exceeds δ; it does not imply the estimated gain is below δ.
* **Q2 interpretation:** retain the pre-specified H2a calculation, but do not interpret residuals as proof of unrecorded work or non-support as proof of no blind spot. Other role comparisons remain exploratory. Signed mean residuals grouped by match closeness are structurally zero after the per-match sum-to-30 normalization; do not interpret that lens as an accuracy comparison. A different estimand would require a documented protocol amendment before test access.
* **R reproducibility:** `make setup-r` installs the required fitzRoy version locally; the manifest records every installed R package version. A complete transitive R lockfile is still pending successful dependency installation. Python uses `uv sync --frozen`.
* **Environment evidence:** the existing 66 Python tests passed before these changes. R 4.5.0 was installed locally from Debian's signed package index; base-R helper checks passed. CRAN installation is blocked by the cloud proxy. The original `protocol-frozen` tag was successfully published to GitHub at `4bf7582` during this review.
