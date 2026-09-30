#!/usr/bin/env bash
# Voxel-grid keypoint network (ours): no upstream repo. Fetches the ImageNet ResNet-18
# initialisation (torchvision, BSD-3-Clause) into weights/ and syncs the uv env.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p weights/trained
W=weights/resnet18-f37072fd.pth
if [ ! -f "$W" ]; then
  curl -fL -o "$W.tmp" https://download.pytorch.org/models/resnet18-f37072fd.pth && mv "$W.tmp" "$W"
fi
echo "f37072fd  $(sha256sum "$W" | cut -c1-8)" | awk '$1!=$2{print "WARNING: checksum prefix mismatch"; exit 1}'
uv sync
