#!/usr/bin/env bash
### Host: run closed_loop.py in the SRB container (this repo at /root/oshd_baselines, the
### pipeline at /root/oshd, SRB at /root/ws), after installing the shipped PPO policy.
###   baselines/docking_policies/run_closed_loop.sh results/policies/docking_test_v3/scenarios.json \
###       --out results/policies/docking_test_v3/teacher --workers 3
### Paths are relative to the repo root. Env: SRB_DIR, PIPELINE_DIR (default: siblings of this repo).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SRB_DIR="$(cd "${SRB_DIR:-${REPO}/../space_robotics_bench}" && pwd)"
PIPE="$(cd "${PIPELINE_DIR:-${REPO}/../osh_dataset_pipeline}" && pwd)"
CMD="cd /root/oshd && sim/install_policies.sh && cd /root/oshd_baselines && python baselines/docking_policies/closed_loop.py $(printf '%q ' "$@")"
exec env SRB_DOCKER_IMAGE="${SRB_DOCKER_IMAGE:-space_robotics_bench:6.1.0}" \
    SRB_DOCKER_PROFILE="${SRB_DOCKER_PROFILE:-simulator}" \
    "${SRB_DIR}/.docker/run.bash" -v "${PIPE}:/root/oshd:rw" -v "${REPO}:/root/oshd_baselines:rw" \
    -e "HOST_UID=$(id -u):$(id -g)" bash -c "${CMD}"
