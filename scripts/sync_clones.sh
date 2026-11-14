#!/usr/bin/env bash
# Send this branch's new commits to a server clone via a git bundle.
# Usage (Git Bash, from the repo root):
#   SERVER_NAME=datalab-king1 ROOT_DIR=/data/scfeng/LCG-pmpo-gaussian-h10-emacritic bash scripts/sync_clones.sh
set -euo pipefail

SERVER_NAME=${SERVER_NAME:?set SERVER_NAME, e.g. kingplus0 or datalab-king1}
ROOT_DIR=${ROOT_DIR:?set ROOT_DIR, e.g. /home/bigking/scfeng/LCG or /data/scfeng/LCG}
BUNDLE_REMOTE="$(dirname "$ROOT_DIR")/laptop_sync.bundle"   # next to the clone, not inside it

BRANCH=$(git branch --show-current)
SERVER_BRANCH=$(ssh "$SERVER_NAME" "git -C '$ROOT_DIR' branch --show-current")
[ "$BRANCH" = "$SERVER_BRANCH" ] || { echo "laptop on '$BRANCH', server on '$SERVER_BRANCH'"; exit 1; }

BASE=$(ssh "$SERVER_NAME" "git -C '$ROOT_DIR' rev-parse HEAD")
if [ "$(git rev-parse HEAD)" = "$BASE" ]; then echo "already in sync at $BASE"; exit 0; fi

git bundle create sync.bundle "$BRANCH" "^$BASE"
scp sync.bundle "$SERVER_NAME:$BUNDLE_REMOTE"
ssh "$SERVER_NAME" "git -C '$ROOT_DIR' pull --ff-only '$BUNDLE_REMOTE' '$BRANCH' && rm '$BUNDLE_REMOTE'"
rm sync.bundle

echo "laptop: $(git rev-parse HEAD)"
echo "server: $(ssh "$SERVER_NAME" "git -C '$ROOT_DIR' rev-parse HEAD")"
