#!/usr/bin/env bash
# SPNv2-on-events baseline setup: upstream SPNv2 at a pinned commit, uv env, ImageNet EfficientNet-B3 weights.
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=debee80c43f226a22dce857b0a136de62e9e939d   # tpark94/spnv2 master, 2024-09-22 (same as baselines/spnv2)
mkdir -p third_party weights/torch/hub/checkpoints
if [ ! -d third_party/spnv2/.git ]; then
  git clone https://github.com/tpark94/spnv2 third_party/spnv2
fi
git -C third_party/spnv2 fetch -q origin
git -C third_party/spnv2 checkout -q "$COMMIT"
uv sync
# torchvision ImageNet weights used by upstream's efficientnet_b3(pretrained=True) (BSD-3, via timm)
W=weights/torch/hub/checkpoints/efficientnet_b3_rwightman-cf984f9c.pth
[ -f "$W" ] || curl -L -o "$W" https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth
echo "spnv2_events setup done"
