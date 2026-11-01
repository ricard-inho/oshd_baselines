# lidar_registration: training-free target crop + FPFH + TEASER++ + ICP

This baseline fills Table I, LiDAR row, method `fpfh_teaser_icp`, setting **ZS** only, because it is training-free. The modality and sensor are both `lidar` (Livox Mid-360). The output pose is `T_lidar_model` in the CubeSat_U1 CAD frame.

The package `oshd_lidar` also holds the **shared target crop** that `baselines/geotransformer` uses (as the path dependency `oshd-lidar`).

| | |
|---|---|
| Upstream | [MIT-SPARK/TEASER-plusplus](https://github.com/MIT-SPARK/TEASER-plusplus) @ `52a9c52ee7d4c838c5e8a75458c33178be5bfb70`, MIT licence. FPFH and ICP come from Open3D (MIT) via pip. |
| Weights | none (training-free) |
| Env | uv project (Python 3.12). PyPI has no `teaserpp_python` wheel, so `setup.sh` builds it from source (CPU, about 2 min) against Eigen 3.4.0 headers that it downloads into `third_party/`. |
| ZS / FT | ZS = training-free, evaluated on real test. There is nothing to fine-tune, so there is no FT. |

## Commands

```bash
cd baselines/lidar_registration && ./setup.sh && cd ../..
# (dev) export OSH_DATASET_ROOT=$PWD/work/dev_dataset OSHD_BENCH_RESULTS=$PWD/work/dev_results OSHD_BENCHMARK=none
uv run --project baselines/lidar_registration lidar-reg-predict --setting ZS --domain real --split test
uv run --project baselines/lidar_registration lidar-reg-predict --setting ZS --domain synthetic --split test
uv run --project bench oshd-bench eval
# diagnostics (these read GT, so they are never part of a result)
uv run --project baselines/lidar_registration lidar-reg-predict --crop oracle     # -> setting ZS-oracle
uv run --project baselines/lidar_registration lidar-stats --domain real --split test  # points on the cube per scan
uv run --project baselines/lidar_registration lidar-viz --setting ZS --n 8          # work/debug/lidar/*.png
```

`uv run` keeps the source-built TEASER++. A plain `uv sync` would remove it, because it is not in the lock file. Use `uv sync --inexact`, or rerun `setup.sh`.

## Pipeline (`src/oshd_lidar/`)

1. **Crop** (`crop.py`, `TargetCropper`). It is training-free and never reads GT. `runtime_ms` includes it.
   * **Gate** on range (0.15–2.0 m) and on height along gravity (−0.25 to +0.12 m relative to the lidar). Gravity is the mean accelerometer direction of the Mid-360's IMU over the trajectory, so it comes from sensor data only. The height band is the known lab geometry: chaser and cube both float on the same flat table.
   * **Remove large planes** with iterative RANSAC (2 cm threshold). Inliers are removed only where they form a connected patch larger than 25 cm.
   * **Cluster** with DBSCAN (eps 3.5 cm). Oversized clusters are re-split at eps 2 cm.
   * **Gate on cube size**: at least 6 points, PCA extents of at most 20 cm, second extent at least 2 cm.
   * **Score** each cluster with a hand-set cube-likeness score (point count and the three extents, where a visible cube is about 12 × 9 × 5 cm). A causal tracking prior adds a Gaussian (σ = 15 cm) on the distance to the previous scan's pick in the same trajectory, in the lidar frame. `--no-track` disables it.
   * **Grow** the chosen cluster by 2 cm back into the gated cloud.

   Why no background model? The cube is static in the world, so a per-trajectory world-frame background model would absorb the cube itself.
2. **Model cloud.** 20k area-uniform CAD samples (`oshd_bench.flow_gt.surface_points`), downsampled on a 1 cm voxel grid, with normals.
3. **Correspondences.** FPFH (5 cm radius). Each scan point takes its 5 feature-space nearest neighbours on the model, capped at the 400 best matches.
4. **TEASER++.** Noise bound 2 cm, GNC-TLS rotation, heuristic max clique (`PMC_HEU`). On a 10 cm object almost every pair of correspondences is consistent at 2 cm. The inlier graph is then dense, and the exact max clique took up to 20 s per scan.
5. **ICP** (point-to-plane, 3 cm) from the TEASER++ result and 6 centroid-based starts. The winner is the solution whose cube centre lies *behind* the observed surface (a visibility check), then the one with the best fitness. The cube symmetry needs no special handling, because the metrics are symmetry-aware.

`--crop oracle` crops a 12 cm sphere around the GT centre. It is for diagnosis only and is written under the setting `ZS-oracle`, which Table I never reads.

## Feasibility: points on the cube (dev data, fitted dev extrinsic)

The lidar extrinsic in `work/dev_dataset` is a placeholder, and it is about 180° wrong. Even so, the cube *is* visible in the scans.

`lidar-extrinsic` fits a dev-only `T_body_lidar` from the cube itself. It uses a Kabsch fit of the cube-cluster centroids to the mocap cube track, then a robust point-to-box refinement. The median point-to-box distance is 0.64 cm. It also makes the world-frame scan-to-scan consistency 2.7 cm instead of 14 cm.

The fit is written into a copy of the dev dataset, `work/lidar_dev/dataset`, with GT in `work/lidar_dev/results`. The real dataset is never touched. The real dataset has no lidar extrinsic yet, so `oshd-bench gt` skips the lidar there.

Points within 2 cm of the GT cube surface per scan (`lidar-stats`; scans are about 17k points):

| split | scans | median | p10 | p90 | ≥ 30 pts |
|---|---|---|---|---|---|
| real test (009) | 323 | 70 | 31 | 164 | 91 % |
| real train (007) | 362 | 50 | 30 | 70 | 90 % |
| "synthetic" train (008) | 341 | 57 | 20 | 81 | 85 % |

By range (real test): 109 points at ≤ 0.4 m, 68 at 0.4–0.5 m, 40 at 0.5–0.75 m, 45 at 0.75–1 m, 23 at 1–1.5 m. Typically 3 faces are hit.

Registration is **feasible but coarse**. The Mid-360's range noise (about 2 cm) is 20 % of the cube size.

## Dev results (fitted extrinsic, real test 009, 323 scans)

| | crop hit (≤ 8 cm) | e_t mean | centre error median | e_q | ADD(-S) AUC | rate |
|---|---|---|---|---|---|---|
| ZS (auto crop) | 92 % | 4.8 cm | 2.3 cm | 23° | 0.63 | 33 Hz |
| ZS-oracle | – | – | 3.4 cm | – | – | – |

These are dev numbers on a dev extrinsic. They show the pipeline works; they are not paper results. The oracle crop is worse in the median because the 12 cm sphere also catches clutter just above the cube.

## Simulated docking: LiDAR-to-CAD ICP, single scan (paper Table I, LiDAR row)

`oshd_lidar.sim_icp` (`lidar-sim-icp`), training-free, CPU only.

| | |
|---|---|
| Split | `benchmark_v2`, `sim_zerog/docking/test`: 162 trajectories |
| Samples | every 5th scan (2 Hz of 10 Hz) with ≥ 30 returns on the target: 12 932 samples (`oshd-bench sim-prepare --task docking --sensor lidar --stride 5` → `results/sim/sim_zerog_docking_test_v2_lidar/`) |
| Segmentation | the simulator's labels of the returns (a return is the target's if it lies on its surface): **oracle**, like FoundationPose's semantic mask. `target_points` in `samples.parquet` |
| Method | 96 rotation hypotheses (uniform on SO(3)), scan centroid at the CAD's box centre; point-to-plane ICP with a Tukey kernel at 15%, 5%, 2% of the CAD diameter; the best fitness (returns within 2% of the diameter) wins |
| Lidar | the sim scan re-expressed about the sensor origin: oshd ≥ 1.1 (or the local `osh-dataset` checkout) reads benchmark_v2's scans corrected; see `osh_dataset_pipeline/sim/fix_lidar_origin.py` |

```bash
cd ~/osh_dataset/oshd_baselines
baselines/foundationpose/.venv/bin/oshd download --benchmark v2 --domain sim_zerog --task docking --split test -m lidar -m mocap
uv run --project bench oshd-bench sim-prepare --task docking --sensor lidar --stride 5      # committed; ~15 min
cd baselines/lidar_registration
uv run lidar-sim-icp --setting ORACLE_SEG --workers 6 --resume                              # ~2.5 h on 6 cores
cd ../.. && uv run --project bench oshd-bench sim-eval --run-dir results/sim/sim_zerog_docking_test_v2_lidar \
    --pred baselines/lidar_registration/results/sim_zerog_docking_test_v2_lidar/predictions_ORACLE_SEG.parquet
```

Until oshd 1.1.0 is on PyPI, add `--with-editable ../osh-dataset` (`../../../osh-dataset` from the
baseline folder) to the `uv run` commands, so the lidar scans are read about the sensor.

### Result (2026-11-01)

**Paper Table I row** (means, rotation up to the measured symmetries): e_t = 8.12 cm,
e_q = 19.7°, S = 0.568; every sample gets a pose. Medians 0.38 cm, 0.38°.

| range | samples | e_t med (cm) | e_t mean (cm) | e_q sym med (°) | e_q sym mean (°) | S sym mean |
|---|---:|---:|---:|---:|---:|---:|
| < 0.3 m | 1562 | 1.97 | 15.07 | 0.96 | 34.9 | 1.136 |
| 0.3-0.5 m | 4893 | 0.61 | 12.77 | 0.74 | 29.5 | 0.887 |
| 0.5-1 m | 3742 | 0.30 | 3.94 | 0.27 | 10.2 | 0.236 |
| 1-2 m | 1981 | 0.28 | 1.57 | 0.27 | 6.9 | 0.132 |
| > 2 m | 754 | 0.32 | 1.54 | 0.35 | 5.9 | 0.108 |

* The simulated lidar has no noise, so a correct registration is sub-centimetre and sub-degree.
* **Close range fails for the same reason as FoundationPose's single frame:** below 0.5 m the
  scan sees one face of the target, and the hypothesis search often settles on a wrong one.
* Registration takes a median of 4.2 s per scan on one CPU core (96 hypotheses x 3 ICP levels).
