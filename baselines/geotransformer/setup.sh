#!/usr/bin/env bash
# GeoTransformer baseline: uv env (torch 2.4.1+cu121), upstream code pinned, CPU C++ extension
# built with torch's JIT loader (no nvcc needed), ModelNet release weights.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
GEOTR_COMMIT=e7a135af4c318ff3b8d7f6c963df094d7e4ea540   # qinzheng93/GeoTransformer main, 2026-09
mkdir -p third_party weights
if [ ! -d third_party/GeoTransformer ]; then
  git clone https://github.com/qinzheng93/GeoTransformer third_party/GeoTransformer
fi
git -C third_party/GeoTransformer checkout -q "$GEOTR_COMMIT"
# the crop lives in ../lidar_registration (pooh-lidar); it needs no TEASER++ here
uv sync -p 3.12 --inexact
PATH="$HERE/.venv/bin:$PATH" .venv/bin/python build_ext.py
W=weights/geotransformer-modelnet.pth.tar
[ -s "$W" ] || curl -L -o "$W" \
  https://github.com/qinzheng93/GeoTransformer/releases/download/1.0.0/geotransformer-modelnet.pth.tar
.venv/bin/python -c "import torch, sys; sys.path.insert(0, 'third_party/GeoTransformer'); import geotransformer.ext; print('geotransformer.ext OK, cuda', torch.cuda.is_available())"
