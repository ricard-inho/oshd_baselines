#!/usr/bin/env bash
# Policy replay has no upstream code or downloadable weights: the frozen policy is the exported
# rsl_rl actor from the training run (path in configs/replay.yaml). This only builds the env
# and checks that the policy file is there.
set -euo pipefail
cd "$(dirname "$0")"
uv sync
ONNX=$(uv run python -c "import yaml;print(yaml.safe_load(open('configs/replay.yaml'))['policy']['onnx'])")
test -f "$ONNX" || { echo "missing policy: $ONNX (edit policy.onnx in configs/replay.yaml)"; exit 1; }
echo "policy: $ONNX"
