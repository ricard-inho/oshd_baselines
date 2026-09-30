#!/usr/bin/env bash
# E2VID baseline: clone upstream (GPL-3.0, not vendored) at a pinned commit + download weights.
set -euo pipefail
cd "$(dirname "$0")"
E2VID_COMMIT=d0a7c005f460f2422f2a4bf605f70820ea7a1e5f   # uzh-rpg/rpg_e2vid master, 2019-12-29
mkdir -p third_party weights
if [ ! -d third_party/rpg_e2vid/.git ]; then
  git clone https://github.com/uzh-rpg/rpg_e2vid.git third_party/rpg_e2vid
fi
git -C third_party/rpg_e2vid fetch -q origin || true
git -C third_party/rpg_e2vid checkout -q "$E2VID_COMMIT"
W=weights/E2VID_lightweight.pth.tar
if [ ! -s "$W" ]; then
  # official URL from the upstream README (http://rpg.ifi.uzh.ch/data/E2VID/models/...)
  curl -fL -o "$W" https://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar \
    || curl -fL -o "$W" http://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar
fi
echo "sha256: $(sha256sum "$W")"
uv sync
