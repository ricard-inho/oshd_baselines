#!/usr/bin/env bash
# MegaPose baseline: clone happypose at a pinned commit, download the RGB MegaPose weights, build the env.
set -euo pipefail
cd "$(dirname "$0")"
HAPPYPOSE_COMMIT=6278e12b9ada314a5def8041d9f33c649fe82f40
mkdir -p third_party weights/megapose-models
if [ ! -d third_party/happypose/.git ]; then
  git clone https://github.com/agimus-project/happypose third_party/happypose
fi
git -C third_party/happypose fetch -q origin "$HAPPYPOSE_COMMIT" 2>/dev/null || true
git -C third_party/happypose checkout -q "$HAPPYPOSE_COMMIT"

# Final checkpoints only (the per-epoch snapshots are skipped); ~170 MB
BASE=https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/megapose-models
for run in coarse-rgb-906902141 refiner-rgb-653307694; do
  mkdir -p "weights/megapose-models/$run"
  for f in checkpoint.pth.tar config.yaml; do
    [ -s "weights/megapose-models/$run/$f" ] || curl -fL --retry 3 -o "weights/megapose-models/$run/$f" "$BASE/$run/$f"
  done
done
sha256sum -c - <<'SUMS'
30fa2ba97d1bb1b27f6443b6671f557baebf3dff93fcc8605a6c7924ead759a4  weights/megapose-models/coarse-rgb-906902141/checkpoint.pth.tar
52dd473fc719488e2f45a98aac2f83960ebe1534b2845cb292c0a287ba70adb7  weights/megapose-models/refiner-rgb-653307694/checkpoint.pth.tar
SUMS

uv sync   # torch 2.5.1+cu124 wheels; builds the small cosypose C++ extension with the system g++
