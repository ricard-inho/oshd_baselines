"""Pick the 32 validation trajectories (held out of training) among the satellite_mockup runs of
the train list; writes ``configs/val_trajectories.txt``, one per line (run once, committed).

    uv run --project baselines/spnv2_events python baselines/spnv2_events/scripts/make_val_split.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
N_VAL = 32

gt = pd.read_parquet(ROOT / "results" / "sim" / "sim_zerog_docking_train_v3" / "gt.parquet")
names = np.sort(gt.loc[gt["object"] == "satellite_mockup", "trajectory"].unique())
val = sorted(np.random.default_rng(0).choice(names, N_VAL, replace=False).tolist())
out = ROOT / "baselines" / "spnv2_events" / "configs" / "val_trajectories.txt"
out.write_text("".join(f"{t}\n" for t in val))
print(f"{out}: {len(val)} of {len(names)} trajectories")
