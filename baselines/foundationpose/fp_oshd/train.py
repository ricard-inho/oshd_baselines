"""There is nothing to train for FoundationPose itself.

FoundationPose's public release has no training code and its refiner / scorer are used
as released. ZS / FT for the rows ``foundationpose`` and ``foundationpose_icp`` are defined
by the Mask R-CNN that supplies the object mask (``baselines/detector``):

* ZS: detector trained on the synthetic train split,
* FT: that detector fine-tuned on the real train split.

This entry point only checks that the detections those settings need exist.
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fp_oshd.train", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", default=None, choices=["ZS", "FT"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--domain", default="real")
    a = ap.parse_args(argv)
    from oshd_bench import detection_path, results_dir

    print(__doc__)
    missing = 0
    for s in [a.setting] if a.setting else ["ZS", "FT"]:
        p = detection_path(results_dir(), "realsense_color", "maskrcnn", s, a.domain, a.split)
        print(f"{s}: {p} {'ok' if p.exists() else 'MISSING (run baselines/detector predict)'}")
        missing += not p.exists()
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
