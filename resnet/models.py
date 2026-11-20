"""Shared ResNet-18 through layer3 and the five policy variants."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet.text import FrozenTextStandIn, LANG_DIM

K_TOKENS = 16
DENSE_SIDE = 8
IMAGE_SIZE = 128
VISUAL_DIM = 256
PROPRIO_DIM = 9
ACTION_DIM = 7
N_CAMERAS = 2


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        hidden = F.relu(self.bn1(self.conv1(x)))
        hidden = self.bn2(self.conv2(hidden))
        return F.relu(hidden + identity)


def _layer(in_channels: int, out_channels: int, stride: int, blocks: int = 2) -> nn.Sequential:
    layers = [BasicBlock(in_channels, out_channels, stride)]
    for _ in range(blocks - 1):
        layers.append(BasicBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)


class ResNet18Layer3(nn.Module):
    """ResNet-18 through layer3. A 128x128 image becomes an 8x8x256 map."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = _layer(64, 64, stride=1)
        self.layer2 = _layer(64, 128, stride=2)
        self.layer3 = _layer(128, 256, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)


class CoordinatePredictor(nn.Module):
    """Predict K coordinates in [-1, 1]. FiLM is optional and fixed per run."""

    def __init__(self, visual_dim: int, lang_dim: int, k: int) -> None:
        super().__init__()
        self.k = k
        self.gamma = nn.Linear(lang_dim, visual_dim)
        self.beta = nn.Linear(lang_dim, visual_dim)
        self.mlp = nn.Sequential(
            nn.Linear(visual_dim, visual_dim),
            nn.ReLU(),
            nn.Linear(visual_dim, k * 2),
        )

    def forward(self, pooled: torch.Tensor, language: torch.Tensor, use_film: bool) -> torch.Tensor:
        context = pooled
        if use_film:
            context = self.gamma(language) * pooled + self.beta(language)
        raw = self.mlp(context).view(pooled.shape[0], self.k, 2)
        return 2.0 * torch.sigmoid(raw) - 1.0


class CoordinateInjection(nn.Module):
    def __init__(self, visual_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, visual_dim),
            nn.ReLU(),
            nn.Linear(visual_dim, visual_dim),
        )

    def forward(self, sampled: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return sampled + self.mlp(coords)


class ActionHead(nn.Module):
    def __init__(self, visual_dim: int, proprio_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(visual_dim + proprio_dim, visual_dim),
            nn.ReLU(),
            nn.Linear(visual_dim, action_dim),
        )

    def forward(self, summary: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((summary, proprio), dim=-1))


def bilinear_sample(feature_map: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Sample K locations. coords are (B, K, 2) in [-1, 1] as (x, y)."""
    grid = coords.unsqueeze(2)
    sampled = F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(-1).permute(0, 2, 1)


def uniform_coords(batch: int, k: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    side = int(math.isqrt(k))
    if side * side != k:
        raise ValueError(f"K={k} must be a square for the equal-budget uniform grid")
    axis = torch.linspace(-1.0, 1.0, side, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.stack((xx, yy), dim=-1).reshape(1, k, 2)
    return coords.expand(batch, -1, -1).contiguous()


def behavior_cloning_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction, target)


def load_imagenet(policy) -> tuple[int, int]:
    from torchvision.models import ResNet18_Weights, resnet18

    reference = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    source = reference.state_dict()
    destination = policy.vision.state_dict()
    copied = 0
    with torch.no_grad():
        for key, value in destination.items():
            if key in source and tuple(source[key].shape) == tuple(value.shape):
                value.copy_(source[key])
                copied += 1
    policy.vision.load_state_dict(destination)
    return copied, len(destination)


class GridSamplerPolicy(nn.Module):
    """Five variants on one ResNet-18 and one 2-layer transformer.

    vision_bc prepends a learned CLS token and uses no language.
    dense_language keeps the full 8x8 map and prepends one language token.
    vanilla_grids predicts coordinates without FiLM.
    grids_film applies FiLM inside the coordinate head.
    uniform_ablation uses a fixed 4x4 grid.
    The action reads encoded[:, 0].
    """

    VARIANTS = (
        "vision_bc",
        "dense_language",
        "vanilla_grids",
        "grids_film",
        "uniform_ablation",
    )

    def __init__(self) -> None:
        super().__init__()
        self.vision = ResNet18Layer3()
        self.text = FrozenTextStandIn()
        self.coord_vanilla = CoordinatePredictor(VISUAL_DIM, LANG_DIM, K_TOKENS)
        self.coord_film = CoordinatePredictor(VISUAL_DIM, LANG_DIM, K_TOKENS)
        self.inject = CoordinateInjection(VISUAL_DIM)
        self.lang_proj = nn.Linear(LANG_DIM, VISUAL_DIM)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, VISUAL_DIM))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=VISUAL_DIM,
                nhead=4,
                dim_feedforward=512,
                dropout=0.0,
                batch_first=True,
            ),
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.heads = nn.ModuleDict(
            {name: ActionHead(VISUAL_DIM, PROPRIO_DIM, ACTION_DIM) for name in self.VARIANTS}
        )

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        batch, cameras, channels, height, width = images.shape
        flat = images.reshape(batch * cameras, channels, height, width)
        features = self.vision(flat)
        return features.reshape(batch, cameras, VISUAL_DIM, DENSE_SIDE, DENSE_SIDE)

    def _camera_coords(
        self,
        feature_map: torch.Tensor,
        language: torch.Tensor,
        variant: str,
    ) -> torch.Tensor:
        if variant == "uniform_ablation":
            return uniform_coords(feature_map.shape[0], K_TOKENS, feature_map.device, feature_map.dtype)
        pooled = feature_map.mean(dim=(-2, -1))
        if variant == "vanilla_grids":
            return self.coord_vanilla(pooled, language, use_film=False)
        if variant == "grids_film":
            return self.coord_film(pooled, language, use_film=True)
        raise KeyError(variant)

    def _tokens(
        self,
        features: torch.Tensor,
        language: torch.Tensor,
        variant: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch = features.shape[0]
        if variant in ("vision_bc", "dense_language"):
            tokens = features.permute(0, 1, 3, 4, 2).reshape(batch, N_CAMERAS * DENSE_SIDE * DENSE_SIDE, VISUAL_DIM)
            return tokens, None

        sampled = []
        coord_sets = []
        for camera in range(N_CAMERAS):
            feature_map = features[:, camera]
            coords = self._camera_coords(feature_map, language, variant)
            sampled.append(self.inject(bilinear_sample(feature_map, coords), coords))
            coord_sets.append(coords)
        tokens = torch.cat(sampled, dim=1)
        return tokens, torch.stack(coord_sets, dim=1)

    def forward(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        token_ids: torch.Tensor,
        variant: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Size]:
        if variant not in self.VARIANTS:
            raise KeyError(variant)
        features = self.encode_images(images)
        language = self.text(token_ids)
        tokens, coords = self._tokens(features, language, variant)
        if variant == "vision_bc":
            prefix = self.cls_token.expand(tokens.shape[0], -1, -1)
        else:
            prefix = self.lang_proj(language).unsqueeze(1)
        sequence = torch.cat((prefix, tokens), dim=1)
        summary = self.encoder(sequence)[:, 0]
        return self.heads[variant](summary, proprio), coords, sequence.shape
