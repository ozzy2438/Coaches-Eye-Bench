#!/usr/bin/env bash
# Re-create the `protocol-frozen` tag on the commit that first added protocol.md, but only if protocol.md
# is byte-identical at HEAD. (The tag could not be pushed from the build sandbox: its git proxy answered 403.)
set -euo pipefail
if git rev-parse -q --verify refs/tags/protocol-frozen >/dev/null; then
  echo "tag protocol-frozen already exists at $(git rev-parse --short protocol-frozen^{commit})"; exit 0
fi
c=$(git log --diff-filter=A --format=%H -- protocol.md | tail -1)
[ -n "$c" ] || { echo "protocol.md was never committed"; exit 1; }
git diff --quiet "$c" HEAD -- protocol.md || { echo "protocol.md changed since $c: do not re-create the tag; see FREEZE_LOG.md"; exit 1; }
git tag -a protocol-frozen "$c" -m "Protocol frozen (re-created on the original freezing commit $c)"
echo "created protocol-frozen at $(git rev-parse --short "$c"); push it with: git push origin protocol-frozen"
