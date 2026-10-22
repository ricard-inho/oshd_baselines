# FoundationPose (NVlabs) inference image for the POOH-D baselines.
#
# Build context: baselines/foundationpose (setup.sh must have cloned third_party/FoundationPose).
#   docker build -t pooh-foundationpose -f docker/Dockerfile .
#
# Modern stack following upstream's conda instructions (commit a1b694b, python 3.11, numpy 2,
# torch cu124) instead of the legacy wenbowen123/foundationpose image (python 3.8), because
# pooh-bench / pooh-dataset need python >= 3.10. CUDA extensions are compiled here, for sm_89
# (RTX 4090) only; override TORCH_CUDA_ARCH_LIST for other GPUs.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_CUDA_ARCH_LIST="8.9"
ARG PYTORCH3D_REF=v0.7.9
ARG NVDIFFRAST_REF=v0.4.0
ARG POOH_DATASET_VERSION=0.5.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential cmake ninja-build pkg-config \
        libeigen3-dev libboost-system-dev libboost-program-options-dev pybind11-dev \
        libgl1 libglib2.0-0 libgomp1 libegl1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH \
    UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300
RUN uv venv -p 3.11 /opt/venv

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} FORCE_CUDA=1 CUDA_HOME=/usr/local/cuda MAX_JOBS=8
RUN uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 \
    && uv pip install setuptools wheel ninja "numpy>=2,<2.3" pybind11 cmake

# GPU extensions (compiled against the torch above)
RUN uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@${PYTORCH3D_REF}"
RUN uv pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git@${NVDIFFRAST_REF}"

# FoundationPose python deps (upstream requirements.txt) + pooh-bench deps. pooh-dataset comes from
# TestPyPI without its dependencies (they come from PyPI: no other package is taken from TestPyPI).
COPY third_party/FoundationPose/requirements.txt /tmp/fp_requirements.txt
RUN uv pip install -r /tmp/fp_requirements.txt \
    && uv pip install --no-deps --index-url https://test.pypi.org/simple/ "pooh-dataset==${POOH_DATASET_VERSION}" \
    && uv pip install "numpy>=1.23" "huggingface_hub>=0.23" "pillow>=9.0" "trimesh>=4.0" \
        "pandas>=2.0" "pyarrow>=14" "pyyaml>=6.0" "tqdm>=4.60"

# Upstream code at the pinned commit + its C++ extension (mycpp). Weights are mounted at runtime
# into /opt/FoundationPose/weights (docker/run.sh).
COPY third_party/FoundationPose /opt/FoundationPose
RUN cd /opt/FoundationPose/mycpp && rm -rf build && mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=/opt/venv/bin/python \
        -DPYTHON_EXECUTABLE=/opt/venv/bin/python -Dpybind11_DIR=$(python -c "import pybind11;print(pybind11.get_cmake_dir())") \
    && make -j8 && ls /opt/FoundationPose/mycpp/build/mycpp*.so

# open3d's pybind module links libusb (kept as a separate layer so the compiled layers stay cached)
RUN apt-get update && apt-get install -y --no-install-recommends libusb-1.0-0 libx11-6 \
    && rm -rf /var/lib/apt/lists/* \
    && python -c "import open3d, cv2, trimesh, pytorch3d, nvdiffrast.torch, warp, kornia, pooh_dataset; print('imports ok')"

# warp caches compiled kernels; keep them in a writable place for arbitrary --user uids
ENV WARP_CACHE_PATH=/tmp/warp_cache XDG_CACHE_HOME=/tmp/.cache HOME=/tmp \
    PYTHONPATH=/opt/FoundationPose PYTHONUNBUFFERED=1
WORKDIR /opt/FoundationPose
