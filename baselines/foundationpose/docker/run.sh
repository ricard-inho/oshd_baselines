#!/usr/bin/env bash
# Run a command in the FoundationPose image with the repo, the dataset, results/ and weights mounted.
#
#   baselines/foundationpose/docker/run.sh python -m fp_oshd.predict --setting ZS --split test --domain real
#   baselines/foundationpose/docker/run.sh            # interactive shell
#
# Honours OSH_DATASET_ROOT / OSHD_BENCH_RESULTS / OSHD_DATA_DIR / OSHD_BENCHMARK like oshd-bench. Symlinks in the
# dataset root (the dev dataset links into ~/.cache/oshd) are resolved and mounted too.
# GPU jobs take the shared lock work/gpu.lock (set FP_NO_LOCK=1 to skip, e.g. for a shell).
# OSH_DATASET_SRC=<osh-dataset checkout> mounts it and puts its src/ first on PYTHONPATH, so a
# local, unreleased oshd is used instead of the one installed in the image.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO=$(cd "$HERE/../.." && pwd)
IMAGE=${FP_IMAGE:-oshd-foundationpose:a1b694b}
RESULTS=${OSHD_BENCH_RESULTS:-$REPO/results}
CACHE=${OSHD_DATA_DIR:-$HOME/.cache/oshd}
mkdir -p "$RESULTS" "$REPO/work"

declare -A MOUNTS=()
add() { [ -e "$1" ] && MOUNTS["$(realpath "$1")"]=1 || true; }
add "$REPO"; add "$RESULTS"; add "$CACHE"
if [ -n "${OSH_DATASET_ROOT:-}" ]; then
  add "$OSH_DATASET_ROOT"
  MOUNTS["$OSH_DATASET_ROOT"]=1   # the unresolved path too (it is what the code opens)
  for e in "$OSH_DATASET_ROOT"/* "$OSH_DATASET_ROOT"/.cache; do
    [ -L "$e" ] && add "$(dirname "$(realpath "$e")")"
  done
fi
ARGS=()
for m in "${!MOUNTS[@]}"; do ARGS+=(-v "$m:$m"); done
ARGS+=(-v "$HERE/weights:/opt/FoundationPose/weights:ro")
PYPATH="/opt/FoundationPose:$REPO/bench/src:$HERE"
if [ -n "${OSH_DATASET_SRC:-}" ]; then
  SRC=$(realpath "$OSH_DATASET_SRC"); PYPATH="$SRC/src:$PYPATH"
  ARGS+=(-v "$SRC:$SRC:ro")
fi
ENVS=(-e "OSHD_BENCH_RESULTS=$RESULTS" -e "OSHD_DATA_DIR=$CACHE" -e "PYTHONPATH=$PYPATH")
[ -n "${OSH_DATASET_ROOT:-}" ] && ENVS+=(-e "OSH_DATASET_ROOT=$OSH_DATASET_ROOT")
[ -n "${OSHD_BENCHMARK:-}" ] && ENVS+=(-e "OSHD_BENCHMARK=$OSHD_BENCHMARK")
TTY=(); [ -t 0 ] && [ -t 1 ] && TTY=(-it)
[ $# -eq 0 ] && set -- bash

CMD=(docker run --rm "${TTY[@]}" --gpus all --ipc=host --hostname "$(hostname)" --user "$(id -u):$(id -g)"
     "${ARGS[@]}" "${ENVS[@]}" -w "$HERE" "$IMAGE" "$@")
if [ "${FP_NO_LOCK:-0}" = 1 ]; then exec "${CMD[@]}"; fi
exec flock "$REPO/work/gpu.lock" "${CMD[@]}"
