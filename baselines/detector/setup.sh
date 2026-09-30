#!/usr/bin/env bash
# Mask R-CNN detector: no upstream clone needed (torchvision), only the COCO init weights.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p weights
uv sync
# torchvision MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 (BSD-3 code; checks the hash in the file name)
uv run python -c "from detector.model import coco_init_weights; print(coco_init_weights())"
