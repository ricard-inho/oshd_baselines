"""Run the detector on every sample of a GT sample list and write a detections file.

    results/detections/<sensor>/maskrcnn/<setting>/<domain>_<split>.parquet

Boxes (xyxy) and masks are in full-resolution pixels of the raw (distorted) sensor image;
for events that is the 1280x720 E2VID frame. The top ``--topk`` detections above
``--min-score`` are written per sample (default: the single best one); samples without a
detection get no row. ``runtime_ms`` covers the detector only (resize, network, mask
pasting / upsampling), not image decoding and not the E2VID reconstruction, whose cost is
in results/e2vid/<domain>_<split>/index.parquet. The ``T`` column is never read.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pooh_bench import DetectionWriter, detection_path, results_dir
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .common import DETECTOR, SENSOR_CFG, SENSORS, _DS, load_checkpoint, load_image, sample_list


class InferSet(Dataset):
    def __init__(self, sensor, domain, split, limit):
        self.sensor, self.domain, self.split = sensor, domain, split
        df = sample_list(sensor, domain, split, with_pose=False, limit=limit)
        self.items = df[["trajectory", "index", "t_ns"]].drop_duplicates(["trajectory", "t_ns"]).to_dict("records")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r = self.items[i]
        img = load_image(_DS.get(), self.sensor, self.domain, self.split, r["trajectory"], r["index"], r["t_ns"])
        return r, img


def overlay(img: np.ndarray, box, mask, score, path: Path):
    im = img.astype(np.float32)
    if mask is not None:
        im[mask] = 0.5 * im[mask] + 0.5 * np.array([255, 0, 0])
    pil = Image.fromarray(im.astype(np.uint8))
    d = ImageDraw.Draw(pil)
    if box is not None:
        lw = max(2, img.shape[1] // 400)
        d.rectangle(list(map(float, box)), outline=(0, 255, 0), width=lw)
        d.text((box[0], max(0, box[1] - 12)), f"{score:.2f}", fill=(0, 255, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(path)


@torch.no_grad()
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sensor", required=True, choices=SENSORS)
    ap.add_argument("--setting", required=True, choices=["ZS", "FT"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--domain", default="real")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--checkpoint", default=None, help="default weights/<sensor>/<setting>.pt")
    ap.add_argument("--min-score", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--debug-dir", default=None, help="save overlays of the first --debug-n samples here")
    ap.add_argument("--debug-n", type=int, default=4)
    a = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(a.sensor, a.setting, device, a.checkpoint)
    classes = ck["classes"]
    size_in = SENSOR_CFG[a.sensor]["input_size"]
    data = InferSet(a.sensor, a.domain, a.split, a.limit)
    dl = DataLoader(data, batch_size=None, shuffle=False, num_workers=a.workers)
    out = detection_path(results_dir(), a.sensor, DETECTOR, a.setting, a.domain, a.split)
    w = DetectionWriter(out, {"detector": "Mask R-CNN R50-FPN v2 (torchvision)", "setting": a.setting,
                              "checkpoint_meta": ck.get("meta"), "input_size": size_in,
                              "min_score": a.min_score, "topk": a.topk,
                              "runtime_ms": "detector only: resize + network + mask upsampling (no decode, no E2VID)"})
    # warm-up (cudnn autotune, lazy init) so the first sample's runtime is representative
    if len(data):
        h0, w0 = (size_in[1], size_in[0]) if size_in else data[0][1].shape[:2]
        model([torch.zeros(3, h0, w0, device=device)])
    n_det, dbg = 0, 0
    for r, img in tqdm(dl, total=len(data), desc=f"{a.sensor} {a.setting} {a.domain}_{a.split}"):
        img = np.asarray(img)
        H, W = img.shape[:2]
        if device.type == "cuda":
            torch.cuda.synchronize()
        tic = time.perf_counter()
        x = torch.from_numpy(img).to(device).permute(2, 0, 1).float().div_(255)
        if size_in and (W, H) != tuple(size_in):
            x = F.interpolate(x[None], size=(size_in[1], size_in[0]), mode="bilinear", align_corners=False,
                              antialias=True)[0]
        o = model([x])[0]
        keep = (o["scores"] >= a.min_score).nonzero().flatten()[: a.topk]
        dets = []
        for k in keep.tolist():
            box = o["boxes"][k].clone()
            m = o["masks"][k:k + 1]  # (1, 1, h, w) probabilities at network-input size
            if m.shape[-2:] != (H, W):
                box[[0, 2]] *= W / m.shape[-1]
                box[[1, 3]] *= H / m.shape[-2]
                m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
            dets.append((box.cpu().numpy(), (m[0, 0] > 0.5).cpu().numpy(), float(o["scores"][k]),
                         classes[int(o["labels"][k]) - 1]))
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - tic) * 1e3
        for box, mask, score, name in dets:
            w.add(r["trajectory"], a.sensor, int(r["t_ns"]), name, box, score, mask, dt)
            n_det += 1
        if a.debug_dir and dbg < a.debug_n:
            b = dets[0] if dets else (None, None, 0.0, None)
            overlay(img, b[0], b[1], b[2],
                    Path(a.debug_dir) / f"det_{a.sensor}_{a.setting}_{a.domain}_{a.split}_{r['trajectory']}_{r['t_ns']}.png")
            dbg += 1
    p = w.save()
    print(f"{p}: {n_det} detections on {len(data)} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
