# Freeze log

Append-only. A re-freeze (moving the `protocol-frozen` tag) is allowed only before the test season is first loaded, and must be logged here with the reason.

| Date | Tag commit | Reason |
|---|---|---|
| 2026-10-01 | `4bf7582` | Initial freeze: protocol written before any implementation; no 2026 data in the repository. |

## Note on the pushed tag

The tag was created in the build session on `4bf7582` but **could not be pushed** (the sandbox's git proxy returned HTTP 403 for tag refs). Commit `4bf7582` is the first commit after `main` on branch `claude/coaches-eye-bench-build` and the only one that has ever touched `protocol.md`. On any clone run `make restore-tag` (verifies `protocol.md` is unchanged since that commit, then re-creates the annotated tag), then `git push origin protocol-frozen`. `make eval-test` refuses until the tag exists.
