"""MegaPose is not re-trained for OSHD (training-free, CAD-based).

ZS / FT differ only in the 2D detector that initialises MegaPose (baselines/detector):
ZS = Mask R-CNN trained on the synthetic train split, FT = additionally fine-tuned on real train.
This entry point exists for the baseline contract and only checks the inputs are in place.
"""

from __future__ import annotations

import argparse


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="megapose-train", description=__doc__)
    ap.add_argument("--setting", default="ZS")
    a = ap.parse_args(argv)
    from .common import WEIGHTS_DIR

    print(__doc__)
    for d in ("coarse-rgb-906902141", "refiner-rgb-653307694"):
        p = WEIGHTS_DIR / "megapose-models" / d / "checkpoint.pth.tar"
        print(f"{'ok' if p.exists() else 'MISSING (run setup.sh)'}: {p}")
    print(f"setting {a.setting}: nothing to train here; train the detector in baselines/detector.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
