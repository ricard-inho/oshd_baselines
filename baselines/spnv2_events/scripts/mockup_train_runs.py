"""Print the satellite_mockup runs of the benchmark's sim_zerog/docking train split, one per line
(what `oshd download -t ...` needs; needs the dataset metadata: run any `oshd download` first)."""

import json

from oshd import OshDataset
from oshd.hub import default_root

ds = OshDataset(benchmark="v3")
targets = json.loads((default_root() / "models" / "sim" / "targets.json").read_text())
for t in ds.split("train", "sim_zerog", task="docking"):
    if targets.get(t, {}).get("model") == "satellite_mockup":
        print(t)
