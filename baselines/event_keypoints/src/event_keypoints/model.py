"""Two ResNet-18 "SimpleBaseline" nets (Xiao et al. 2018): a ResNet-18 encoder + 3 deconvs.

* ``Detector``  (stage A): centre heatmap (CenterNet focal loss) + log box side, on the
  whole 1/2-resolution voxel grid; output stride 4 (96x160 for a 384x640 input).
* ``KeypointNet`` (stage B): 9 keypoint heatmaps (8 cube corners + centre) on a 256x256 crop,
  output 64x64.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torchvision

from .common import NUM_BINS

WEIGHTS = Path(__file__).resolve().parents[2] / "weights" / "resnet18-f37072fd.pth"


def _resnet18(in_ch: int, pretrained: bool) -> nn.Module:
    net = torchvision.models.resnet18(weights=None)
    if pretrained and WEIGHTS.exists():
        net.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    elif pretrained:
        print(f"[event_keypoints] {WEIGHTS} missing (run setup.sh): ResNet-18 from scratch")
    w = net.conv1.weight.data  # (64, 3, 7, 7) -> (64, in_ch, 7, 7), same response scale
    conv = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
    conv.weight.data = w.mean(1, keepdim=True).repeat(1, in_ch, 1, 1) * (3.0 / in_ch)
    net.conv1 = conv
    return nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3,
                         net.layer4)


def _deconvs(c_in: int = 512, c: int = 256, n: int = 3) -> nn.Sequential:
    layers = []
    for _ in range(n):
        layers += [nn.ConvTranspose2d(c_in, c, 4, 2, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
        c_in = c
    return nn.Sequential(*layers)


def _head(c: int, out: int, bias: float = 0.0) -> nn.Sequential:
    h = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(c, out, 1))
    nn.init.constant_(h[-1].bias, bias)
    return h


class Detector(nn.Module):
    def __init__(self, in_ch: int = NUM_BINS, pretrained: bool = True):
        super().__init__()
        self.enc = _resnet18(in_ch, pretrained)
        self.up = _deconvs(512, 128)
        self.heat = _head(128, 1, bias=-2.19)  # CenterNet prior
        self.size = _head(128, 1, bias=5.0)    # log(side px), ~150 px

    def forward(self, x):
        f = self.up(self.enc(x))
        return self.heat(f), self.size(f)


class KeypointNet(nn.Module):
    def __init__(self, in_ch: int = NUM_BINS, n_kp: int = 9, pretrained: bool = True):
        super().__init__()
        self.enc = _resnet18(in_ch, pretrained)
        self.up = _deconvs(512, 256)
        self.out = nn.Conv2d(256, n_kp, 1)

    def forward(self, x):
        return self.out(self.up(self.enc(x)))


def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 2, beta: float = 4) -> torch.Tensor:
    """CenterNet penalty-reduced focal loss; ``target`` Gaussians with peak exactly 1."""
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = target.ge(0.999).float()
    neg = 1 - pos
    pl = torch.log(p) * (1 - p) ** alpha * pos
    nl = torch.log(1 - p) * p**alpha * (1 - target) ** beta * neg
    n = pos.sum().clamp(min=1)
    return -(pl.sum() + nl.sum()) / n
