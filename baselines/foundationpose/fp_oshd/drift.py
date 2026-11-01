"""Drift of ``fp_oshd.predict_track``: pose error against the time since the last registration.

Host side, after ``oshd-bench sim-eval`` has written ``per_sample_<setting>.parquet``:

    .venv/bin/python -m fp_oshd.drift --run-dir ../../results/sim/sim_zerog_inspection_test_v2 \\
        --pred results/sim_zerog_inspection_test_v2/predictions_TRACK_ORACLE_INIT.parquet

A track starts at a registered frame (finite ``score``) and lasts until the next one; the time of
a sample is ``t_ns`` minus the registration's. Writes ``DRIFT_<setting>.csv`` / ``.md`` next to the
predictions: per time bin, the samples, coverage (of the GT samples in that bin), mean / median
e_t, e_q (up to the measured symmetries) and SPEED, and the fraction within 5 cm / 5°. Samples
before the first registration of a segment have no time since registration; they appear in the
coverage of the overall table only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BINS_S = [0, 1, 2, 5, 10, 20, 30, 60]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fp_oshd.drift", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--pred", required=True)
    a = ap.parse_args(argv)
    pred_p = Path(a.pred)
    stem = pred_p.stem.removeprefix("predictions_")
    pred = pd.read_parquet(pred_p)
    per = pd.read_parquet(pred_p.parent / f"per_sample_{stem}.parquet")

    pred = pred.sort_values(["trajectory", "t_ns"])
    reg_t = pred["t_ns"].where(np.isfinite(pred["score"].astype(float)))
    pred["t_reg"] = reg_t.groupby(pred["trajectory"]).ffill()
    df = per.merge(pred[["trajectory", "t_ns", "t_reg"]], on=["trajectory", "t_ns"], how="left")
    df["t_since_reg_s"] = (df["t_ns"] - df["t_reg"]) / 1e9
    tracked = df[df["t_since_reg_s"].notna()].copy()
    tracked["bin"] = pd.cut(tracked["t_since_reg_s"], BINS_S + [np.inf], right=False)

    rows = []
    for b, g in tracked.groupby("bin", observed=True):
        rows.append({"t_since_reg_s": str(b), "samples": len(g),
                     "e_t_cm_mean": 100 * g["e_t"].mean(), "e_t_cm_median": 100 * g["e_t"].median(),
                     "e_q_sym_deg_mean": g["e_q_sym"].mean(), "e_q_sym_deg_median": g["e_q_sym"].median(),
                     "speed_sym_mean": g["speed_sym"].mean(),
                     "within_5cm_5deg_sym": float(((g["e_t"] < 0.05) & (g["e_q_sym"] < 5)).mean())})
    t = pd.DataFrame(rows)
    t.to_csv(pred_p.parent / f"DRIFT_{stem}.csv", index=False)
    n_reg = int(np.isfinite(pred["score"].astype(float)).sum())
    lines = [f"# Drift of {stem}: error vs time since registration", "",
             f"{len(tracked)} tracked samples in {n_reg} tracks (one registration each); "
             f"{len(df) - len(tracked)} GT samples before a track started (no prediction).", "",
             "| t since registration (s) | samples | e_t mean (cm) | e_t med (cm) | e_q sym mean (°) | "
             "e_q sym med (°) | S sym mean | 5cm/5° sym |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in t.itertuples():
        lines.append(f"| {r.t_since_reg_s} | {r.samples} | {r.e_t_cm_mean:.2f} | {r.e_t_cm_median:.2f} | "
                     f"{r.e_q_sym_deg_mean:.2f} | {r.e_q_sym_deg_median:.2f} | {r.speed_sym_mean:.3f} | "
                     f"{r.within_5cm_5deg_sym:.3f} |")
    (pred_p.parent / f"DRIFT_{stem}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
