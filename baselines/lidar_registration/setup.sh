#!/usr/bin/env bash
# FPFH + TEASER++ + ICP baseline: uv env + TEASER++ python bindings built from source.
# No pip wheel of teaserpp_python exists on PyPI, so we build it (CPU only, ~3 min) against
# Eigen 3.4.0 headers downloaded into third_party/. No weights (training-free).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
TEASER_COMMIT=52a9c52ee7d4c838c5e8a75458c33178be5bfb70   # MIT-SPARK/TEASER-plusplus master, 2026-09
EIGEN_VER=3.4.0
mkdir -p third_party
uv sync -p 3.12 --inexact   # --inexact: keep the source-built teaserpp_python

if [ ! -d third_party/TEASER-plusplus ]; then
  git clone https://github.com/MIT-SPARK/TEASER-plusplus third_party/TEASER-plusplus
fi
git -C third_party/TEASER-plusplus fetch -q origin "$TEASER_COMMIT" || true
git -C third_party/TEASER-plusplus checkout -q "$TEASER_COMMIT"

if [ ! -d third_party/eigen-install ]; then
  curl -L -o third_party/eigen.tar.gz \
    "https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VER}/eigen-${EIGEN_VER}.tar.gz"
  tar -xzf third_party/eigen.tar.gz -C third_party
  cmake -S "third_party/eigen-${EIGEN_VER}" -B "third_party/eigen-build" \
        -DCMAKE_INSTALL_PREFIX="$HERE/third_party/eigen-install" -DBUILD_TESTING=OFF >/dev/null
  cmake --install third_party/eigen-build >/dev/null
fi

if ! .venv/bin/python -c "import teaserpp_python" 2>/dev/null; then
  CMAKE_PREFIX_PATH="$HERE/third_party/eigen-install" CMAKE_BUILD_PARALLEL_LEVEL=8 \
  SKBUILD_CMAKE_DEFINE="ENABLE_DIAGNOSTIC_PRINT=OFF;BUILD_DOC=OFF" \
    uv pip install --python .venv/bin/python ./third_party/TEASER-plusplus
fi
.venv/bin/python -c "import teaserpp_python, open3d; print('teaserpp_python OK, open3d', open3d.__version__)"
