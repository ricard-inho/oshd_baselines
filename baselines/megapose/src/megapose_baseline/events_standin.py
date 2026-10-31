"""Stand-in for the E2VID frames (smoke tests only): event-count images of the sample windows,
written in the E2VID layout ``<out>/<domain>_<split>/<trajectory>/<t_ns>.png`` + an index parquet
(``<out>/<domain>_<split>.parquet``: trajectory, t_ns, path, runtime_ms).

    megapose-events-standin --out ../../work/megapose_e2vid_standin --limit 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .common import load_samples


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="megapose-events-standin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", default="real")
    ap.add_argument("--split", default="test")
    ap.add_argument("--window-ms", type=float, default=50.0)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)

    import oshd_bench
    from oshd_bench.sensors import events_window
    from oshd_dataset.events import event_histogram

    ds = oshd_bench.open_dataset()
    s = load_samples(oshd_bench.results_dir(), "events", a.domain, a.split)
    s = s.drop_duplicates(["trajectory", "t_ns"])
    if a.limit:
        s = s.iloc[:a.limit]
    root = Path(a.out)
    rows = []
    for r in tqdm(s.itertuples(index=False), total=len(s), desc="event images"):
        t0 = time.perf_counter()
        intr = ds[r.trajectory].calibration.intrinsics("events")
        h = event_histogram(events_window(ds[r.trajectory], int(r.t_ns), a.window_ms), intr.height, intr.width)
        c = h.sum(0)
        img = (255 * np.clip(c / max(np.percentile(c[c > 0], 99) if (c > 0).any() else 1, 1), 0, 1)).astype(np.uint8)
        p = root / f"{a.domain}_{a.split}" / r.trajectory / f"{int(r.t_ns)}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(p)
        rows.append({"trajectory": r.trajectory, "t_ns": int(r.t_ns), "path": str(p.relative_to(root)),
                     "runtime_ms": (time.perf_counter() - t0) * 1000})
    pd.DataFrame(rows).to_parquet(root / f"{a.domain}_{a.split}.parquet", index=False)
    print(f"{len(rows)} frames -> {root / f'{a.domain}_{a.split}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
