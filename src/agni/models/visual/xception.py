"""Standalone Xception architecture for deepfake visual detection.

Attribution & Licensing:
- Original Architecture: Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions", CVPR 2017.
- FaceForensics++ Adaptation: Rossler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images", ICCV 2019.
- DeepfakeBench Benchmark: Yan et al., "DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection", NeurIPS 2023 D&B.
  Checkpoint and reference code: https://github.com/SCLBD/DeepfakeBench (CC BY-NC 4.0 / MIT).

This standalone implementation removes heavy external framework dependencies (tensorboard, lmdb, albumentations)
while strictly preserving exact state_dict tensor compatibility with the official DeepfakeBench Xception weights.
"""

from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    """Depthwise separable 2D convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    """Xception residual block with depthwise separable convolutions."""

    def __init__(self, in_filters: int, out_filters: int, reps: int, strides: int = 1, start_with_relu: bool = True, grow_first: bool = True):
        super().__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for _ in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class Xception(nn.Module):
    """Xception network configured for 2-class deepfake detection."""

    def __init__(self, num_classes: int = 2, inc: int = 3):
        super().__init__()
        self.num_classes = num_classes

        # Entry flow
        self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # Middle flow (blocks 4 to 11)
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.last_linear = nn.Linear(2048, self.num_classes)

        # Retained for strict state_dict parameter matching with DeepfakeBench checkpoint
        self.adjust_channel = nn.Sequential(
            nn.Conv2d(2048, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
        )

    def features(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.conv1(inp)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)

        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        x = self.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        out = self.last_linear(x)
        return out

    def forward(self, inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(inp)
        out = self.classifier(feat)
        return out, feat


def load_xception_detector(checkpoint_path: str | Path, num_classes: int = 2) -> Xception:
    """Instantiates Xception and loads weights with zero missing/unexpected key tolerance.
    
    Args:
        checkpoint_path: Path to official xception_best.pth weights file.
        num_classes: Number of classification targets (2: 0=real, 1=fake).
        
    Returns:
        Xception model in evaluation mode.
    """
    path = Path(checkpoint_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Xception checkpoint not found at: {path}")

    model = Xception(num_classes=num_classes)
    raw_ckpt = torch.load(path, map_location="cpu")

    # DeepfakeBench weights are saved with 'backbone.' prefix in state_dict
    state_dict = {}
    for k, v in raw_ckpt.items():
        clean_key = k.replace("backbone.", "").replace("module.", "")
        state_dict[clean_key] = v

    load_result = model.load_state_dict(state_dict, strict=True)
    if len(load_result.missing_keys) > 0 or len(load_result.unexpected_keys) > 0:
        raise RuntimeError(
            f"State-dict mismatch loading {path.name}: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )

    model.eval()
    return model
