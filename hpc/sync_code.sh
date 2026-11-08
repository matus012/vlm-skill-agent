#!/bin/bash
# Ship the committed HEAD of this repo to PERUN as a clean clone at that commit (hpc/PERUN_SIM.md section 1). On the local machine,
# from the repo root:   hpc/sync_code.sh [DEST]     (DEST default $PROJ/repo; dev: $PROJ/dev/<who>/repo)
# Refuses a dirty tree. The .git dir is rsynced to DEST.git (objects, refs, tags; no GitHub access from PERUN), then
# DEST is cloned from it once and afterwards fetched (branches + all tags, e.g. phase1-world-frozen-v2), checked out
# detached at HEAD's commit, and `git status --porcelain` must be empty there. Prints the commit.
set -euo pipefail
LOGIN=$LOGIN_NODE
PROJ=${PROJ:-$HOME/vlm_agent_project}  # your cluster project dir
SSH=(ssh -o BatchMode=yes -o LogLevel=ERROR "$LOGIN")
TOP=$(git rev-parse --show-toplevel)
DEST=${1:-$PROJ/repo}
[[ $DEST == "$PROJ"/* ]] || { echo "DEST must be under $PROJ"; exit 2; }
[ -z "$(git -C "$TOP" status --porcelain)" ] || { echo "refusing: $TOP has uncommitted changes"; git -C "$TOP" status --short; exit 1; }
SHA=$(git -C "$TOP" rev-parse HEAD)
"${SSH[@]}" "mkdir -p $(dirname "$DEST")"
rsync -a --delete -e "ssh -o BatchMode=yes -o LogLevel=ERROR" "$TOP/.git/" "$LOGIN:$DEST.git/"
"${SSH[@]}" bash -s -- "$DEST" "$SHA" <<'REMOTE'
set -euo pipefail
DEST=$1 SHA=$2
if [ -d "$DEST/.git" ]; then
    [ -z "$(git -C "$DEST" status --porcelain)" ] || { echo "refusing: $DEST was edited on PERUN"; git -C "$DEST" status --short; exit 1; }
else
    git clone -q "$DEST.git" "$DEST"
fi
cd "$DEST"
git remote set-url origin "$DEST.git"
git fetch -q --force --tags origin '+refs/heads/*:refs/remotes/origin/*'
git -c advice.detachedHead=false checkout -q --detach "$SHA"
[ "$(git rev-parse HEAD)" = "$SHA" ] && [ -z "$(git status --porcelain)" ] || { echo "checkout of $SHA not clean"; exit 1; }
echo "PERUN $DEST at $(git log -1 --format='%H %cI %s') ($(git tag | wc -l) tags)"
REMOTE
echo "synced commit $SHA"
