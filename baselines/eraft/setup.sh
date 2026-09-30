#!/usr/bin/env bash
# E-RAFT (uzh-rpg/E-RAFT, MIT) pinned + the official DSEC checkpoint, then the uv env.
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=c58ce0524ea0ebfa9849991caafb547f44fe9bfd
mkdir -p third_party weights/trained
if [ ! -d third_party/E-RAFT/.git ]; then
  git clone https://github.com/uzh-rpg/E-RAFT.git third_party/E-RAFT
fi
git -C third_party/E-RAFT fetch -q origin && git -C third_party/E-RAFT checkout -q "$COMMIT"
if [ ! -f weights/dsec.tar ]; then
  curl -fL -o weights/dsec.tar.tmp https://download.ifi.uzh.ch/rpg/ERAFT/checkpoints/dsec.tar
  mv weights/dsec.tar.tmp weights/dsec.tar
fi
uv sync
