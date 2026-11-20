"""Behavior-cloning loops for the five ResNet-18 variants."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from resnet.data import IMAGENET_MEAN, IMAGENET_STD, FrameDataset, prepare_batch, proprio_stats
from resnet.models import GridSamplerPolicy, behavior_cloning_loss, bilinear_sample, load_imagenet
from resnet.text import attach_minilm, encode_task, use_goal_suffix


def run_epoch(
    policy: GridSamplerPolicy,
    loader: DataLoader,
    device: torch.device,
    variant: str,
    image_mean: torch.Tensor,
    image_std: torch.Tensor,
    stats: dict,
    optimizer: torch.optim.Optimizer | None,
    clip_norm: float = 1.0,
) -> float:
    training = optimizer is not None
    policy.train(training)
    total = 0.0
    count = 0
    for batch in loader:
        images, proprio, actions, token_ids = prepare_batch(batch, device, image_mean, image_std, augment=training)
        proprio = (proprio - torch.as_tensor(stats["mean"], device=device)) / torch.as_tensor(stats["std"], device=device)
        raw, _coords, _shape = policy(images, proprio, token_ids, variant)
        prediction = torch.tanh(raw)
        loss = behavior_cloning_loss(prediction, actions)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_norm)
            optimizer.step()
        total += float(loss.detach()) * actions.shape[0]
        count += actions.shape[0]
    return total / max(count, 1)


def image_stats(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 1, 3, 1, 1)
    return mean, std


def load_policy(path: Path, device: torch.device) -> tuple[GridSamplerPolicy, dict, str]:
    blob = torch.load(path, map_location=device, weights_only=False)
    variant = blob["variant"]
    policy = GridSamplerPolicy().to(device)
    policy.load_state_dict(blob["state_dict"])
    if variant != "vision_bc":
        use_goal_suffix(policy)
    policy.eval()
    stats = {"mean": blob["proprio_mean"], "std": blob["proprio_std"]}
    return policy, stats, variant


def _fit(
    variant: str,
    splits: dict,
    device: torch.device,
    epochs: int,
    path: Path,
    text_mode: str,
    train_text: bool,
) -> dict:
    stats = proprio_stats(splits["train"])
    policy = GridSamplerPolicy().to(device)
    for parameter in policy.text.parameters():
        parameter.requires_grad = train_text
    if variant != "vision_bc":
        use_goal_suffix(policy)
    copied, total = load_imagenet(policy)
    print(f"  {variant}: copied {copied}/{total} ImageNet-1K tensors into ResNet-18 through layer3", flush=True)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in policy.parameters() if parameter.requires_grad),
        lr=1e-4,
        weight_decay=1e-4,
    )
    train_loader = DataLoader(FrameDataset(splits["train"]), batch_size=32, shuffle=True)
    val_loader = DataLoader(FrameDataset(splits["val"]), batch_size=64, shuffle=False)
    image_mean, image_std = image_stats(device)
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss = run_epoch(policy, train_loader, device, variant, image_mean, image_std, stats, optimizer)
        with torch.no_grad():
            val_loss = run_epoch(policy, val_loader, device, variant, image_mean, image_std, stats, None)
        if variant != "vision_bc":
            use_goal_suffix(policy)
        print(
            f"  {variant} epoch {epoch}/{epochs}: train_mse {train_loss:.6f} "
            f"val_mse {val_loss:.6f} seconds {time.perf_counter() - started:.1f}",
            flush=True,
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "variant": variant,
            "state_dict": best_state,
            "proprio_mean": stats["mean"],
            "proprio_std": stats["std"],
            "val_loss": best_loss,
            "vision_init": "ImageNet-1K resnet18",
            "text_init": text_mode,
        },
        path,
    )
    policy.load_state_dict(best_state)
    if variant != "vision_bc":
        use_goal_suffix(policy)
    policy.eval()
    print(f"  saved {path} best_val_mse {best_loss:.6f}", flush=True)
    return {"policy": policy, "stats": stats, "val_loss": best_loss, "path": str(path)}


def train_variant(variant: str, splits: dict, device: torch.device, epochs: int, path: Path) -> Path:
    bundle = _fit(
        variant,
        splits,
        device,
        epochs,
        path,
        "trained byte stand-in, no MiniLM file",
        train_text=True,
    )
    return Path(bundle["path"])


def train_protocol_variant(variant: str, splits: dict, device: torch.device, epochs: int, path: Path) -> dict:
    text_mode = "byte stand-in" if variant == "vision_bc" else "byte stand-in goal suffix"
    return _fit(variant, splits, device, epochs, path, text_mode, train_text=False)


def language_swap(policy: GridSamplerPolicy, device: torch.device, episodes: list[dict], frozen: bool) -> dict:
    episode = episodes[0]
    images = torch.from_numpy(episode["images"][:4]).permute(0, 1, 4, 2, 3).float().div_(255.0)
    mean = torch.tensor(list(IMAGENET_MEAN), device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor(list(IMAGENET_STD), device=device).view(1, 1, 3, 1, 1)
    images = (images.to(device) - mean) / std
    stove = torch.tensor([encode_task("put the bowl on the stove")], dtype=torch.long, device=device)
    plate = torch.tensor([encode_task("put the bowl on the plate")], dtype=torch.long, device=device)
    stove = stove.expand(images.shape[0], -1)
    plate = plate.expand(images.shape[0], -1)
    policy.eval()
    with torch.no_grad():
        features = policy.encode_images(images)
        camera = features[:, 0]
        pooled = camera.mean(dim=(-2, -1))
        lang_stove = policy.text(stove)
        lang_plate = policy.text(plate)
        coords_stove = policy.coord_film(pooled, lang_stove, use_film=True)
        coords_plate = policy.coord_film(pooled, lang_plate, use_film=True)
        samples_stove = bilinear_sample(camera, coords_stove)
        samples_plate = bilinear_sample(camera, coords_plate)
    return {
        "language_cosine": float(F.cosine_similarity(lang_stove[0], lang_plate[0], dim=0)),
        "coords_max_abs": float((coords_stove - coords_plate).abs().max()),
        "samples_max_abs": float((samples_stove - samples_plate).abs().max()),
        "coords_changed": bool(not torch.allclose(coords_stove, coords_plate)),
        "samples_changed": bool(not torch.allclose(samples_stove, samples_plate)),
        "encoder_frozen": frozen,
    }


def train_film(splits: dict, device: torch.device, encoder_path: Path, out_path: Path, epochs: int = 50) -> dict:
    from resnet.text import FrozenMiniLM

    encoder = FrozenMiniLM(encoder_path, device)
    frozen = all(not parameter.requires_grad for parameter in encoder.model.parameters())
    if not frozen:
        raise RuntimeError("MiniLM parameters are trainable")
    stats = proprio_stats(splits["train"])
    policy = GridSamplerPolicy().to(device)
    attach_minilm(policy, encoder)
    copied, total = load_imagenet(policy)
    print(f"imagenet_tensors {copied}/{total}", flush=True)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in policy.parameters() if parameter.requires_grad),
        lr=1e-4,
        weight_decay=1e-4,
    )
    train_loader = DataLoader(FrameDataset(splits["train"]), batch_size=32, shuffle=True)
    val_loader = DataLoader(FrameDataset(splits["val"]), batch_size=64, shuffle=False)
    image_mean, image_std = image_stats(device)
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss = run_epoch(policy, train_loader, device, "grids_film", image_mean, image_std, stats, optimizer)
        with torch.no_grad():
            val_loss = run_epoch(policy, val_loader, device, "grids_film", image_mean, image_std, stats, None)
        attach_minilm(policy, encoder)
        print(
            f"epoch {epoch}/{epochs} train_mse {train_loss:.6f} val_mse {val_loss:.6f} "
            f"seconds {time.perf_counter() - started:.1f}",
            flush=True,
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
    policy.load_state_dict(best_state)
    attach_minilm(policy, encoder)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "variant": "grids_film",
            "state_dict": best_state,
            "proprio_mean": stats["mean"],
            "proprio_std": stats["std"],
            "val_loss": best_loss,
            "vision_init": "ImageNet-1K resnet18",
            "text_init": "frozen sentence-transformers/all-MiniLM-L6-v2",
            "text_encoder_path": str(encoder_path),
        },
        out_path,
    )
    print(f"saved {out_path} best_val_mse {best_loss:.6f}", flush=True)
    return {"policy": policy, "stats": stats, "val_loss": best_loss, "encoder_frozen": frozen}
