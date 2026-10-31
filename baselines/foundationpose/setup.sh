#!/usr/bin/env bash
# FoundationPose baseline setup: upstream code (pinned), weights, Docker image.
#   bash baselines/foundationpose/setup.sh            # all three steps
#   SKIP_BUILD=1 bash baselines/foundationpose/setup.sh
#   FP_CUDA_ARCH=8.6 FP_IMAGE=oshd-foundationpose:a1b694b-sm86 bash baselines/foundationpose/setup.sh   # RTX 3090
#   (FP_CUDA_ARCH: compute capability of the GPU, `nvidia-smi --query-gpu=compute_cap --format=csv`)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
COMMIT=a1b694b83e633c2cb6115b9063d940a687759392   # NVlabs/FoundationPose main, 2026-04-29
IMAGE=${FP_IMAGE:-oshd-foundationpose:a1b694b}

# 1. upstream code
mkdir -p "$HERE/third_party" "$HERE/weights"
if [ ! -d "$HERE/third_party/FoundationPose/.git" ]; then
  git clone https://github.com/NVlabs/FoundationPose "$HERE/third_party/FoundationPose"
fi
git -C "$HERE/third_party/FoundationPose" fetch -q origin "$COMMIT" 2>/dev/null || true
git -C "$HERE/third_party/FoundationPose" checkout -q "$COMMIT"

# 2. weights (Google Drive folder from the upstream readme): refiner + scorer
#    https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i
#    If gdown is blocked (quota / "Access denied"), download the two folders
#    2023-10-28-18-33-37/ and 2024-01-11-20-02-45/ by hand into weights/.
need() { [ -s "$HERE/weights/$1/model_best.pth" ] && [ -s "$HERE/weights/$1/config.yml" ]; }
if ! need 2023-10-28-18-33-37 || ! need 2024-01-11-20-02-45; then
  (cd "$HERE/weights" && uvx --from gdown gdown --folder \
      https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i -O .)
fi
need 2023-10-28-18-33-37 && need 2024-01-11-20-02-45 || { echo "weights missing, see above" >&2; exit 1; }
ls -la "$HERE"/weights/*/

# 3. Docker image (compiles pytorch3d, nvdiffrast and mycpp for FP_CUDA_ARCH, default sm_89; ~20-40 min)
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  docker build --network=host --build-arg TORCH_CUDA_ARCH_LIST="${FP_CUDA_ARCH:-8.9}" -t "$IMAGE" \
      -f "$HERE/docker/Dockerfile" "$HERE"
fi
