# Freeze log

Append-only. A re-freeze (moving the `protocol-frozen` tag) is allowed only before the test season is first loaded, and must be logged here with the reason.

| Date | Tag commit | Reason |
|---|---|---|
| 2026-10-01 | `4bf7582` | Initial freeze: protocol written before any implementation; no 2026 data in the repository. |

## Note on the pushed tag

The original build session could not push the tag. On 2026-10-01 the readiness review restored and successfully published it on the same original commit `4bf7582`. The protocol was not changed or re-frozen. Fetch it on an existing clone with `git fetch origin tag protocol-frozen`; a fresh full clone receives it automatically. The restore helper remains available for historical/offline clones.
