"""Demonstration frames, splits, and batch preparation."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from resnet.text import encode_task

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/user2/simulation-robot-learning")
DATA = SOURCE / "datasets" / "libero_goal"
BDDL = SOURCE / "third_party" / "LIBERO" / "libero" / "libero" / "bddl_files"
CACHE = ROOT / "datasets"
SOURCE_CACHE = SOURCE / "datasets" / "cp2_live"
ENCODER = SOURCE / "models" / "all-MiniLM-L6-v2"

N_USE = 40
TRAIN_N = 28
VAL_N = 6
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TASKS = (
    {
        "name": "put the bowl on the stove",
        "hdf5": DATA / "put_the_bowl_on_the_stove_demo.hdf5",
        "bddl": BDDL / "libero_goal" / "put_the_bowl_on_the_stove.bddl",
    },
    {
        "name": "put the bowl on the plate",
        "hdf5": DATA / "put_the_bowl_on_the_plate_demo.hdf5",
        "bddl": BDDL / "libero_goal" / "put_the_bowl_on_the_plate.bddl",
    },
)
VARIANTS = (
    "vision_bc",
    "dense_language",
    "vanilla_grids",
    "grids_film",
    "uniform_ablation",
)
TASK_KEY = {
    "put the bowl on the stove": "put_the_bowl_on_the_stove",
    "put the bowl on the plate": "put_the_bowl_on_the_plate",
}


def demo_ids(path: Path) -> list[int]:
    with h5py.File(path, "r") as handle:
        ids = sorted(int(key.split("_", 1)[1]) for key in handle["data"].keys() if key.startswith("demo_"))
    if len(ids) < N_USE:
        raise RuntimeError(f"{path} has {len(ids)} demos, need {N_USE}")
    return ids[:N_USE]


def cache_path(task: dict) -> Path:
    local = CACHE / f"{task['bddl'].stem}.hdf5"
    if local.exists():
        return local
    shared = SOURCE_CACHE / f"{task['bddl'].stem}.hdf5"
    if shared.exists():
        return shared
    return local


def write_cache(task: dict, episodes: list[dict]) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{task['bddl'].stem}.hdf5"
    with h5py.File(path, "w") as handle:
        handle.attrs["language"] = task["name"]
        for episode in episodes:
            group = handle.create_group(f"demo_{episode['id']}")
            group.attrs["success"] = int(episode["success"])
            group.attrs["success_step"] = episode["success_step"]
            group.create_dataset("images", data=episode["images"])
            group.create_dataset("proprio", data=episode["proprio"])
            group.create_dataset("actions", data=episode["actions"])
            group.create_dataset("init_state", data=episode["init_state"])
    return path


def read_cache(task: dict) -> list[dict]:
    path = cache_path(task)
    episodes = []
    with h5py.File(path, "r") as handle:
        language = handle.attrs["language"]
        ids = sorted(int(key.split("_", 1)[1]) for key in handle.keys())
        for demo_id in ids:
            group = handle[f"demo_{demo_id}"]
            init_state = np.asarray(group["init_state"], dtype=np.float64)
            episodes.append(
                {
                    "id": demo_id,
                    "language": language,
                    "init_state": init_state,
                    "images": np.asarray(group["images"]),
                    "proprio": np.asarray(group["proprio"], dtype=np.float32),
                    "actions": np.asarray(group["actions"], dtype=np.float32),
                    "success": bool(group.attrs["success"]),
                    "success_step": int(group.attrs["success_step"]),
                    "group": tuple(np.round(init_state, 3).tolist()),
                }
            )
    return episodes


def split_episodes(episodes: list[dict]) -> dict[str, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for episode in episodes:
        groups.setdefault(episode["group"], []).append(episode)
    ordered = [groups[key] for key in sorted(groups, key=lambda item: min(ep["id"] for ep in groups[item]))]
    buckets = {"train": [], "val": [], "test": []}
    limits = {"train": TRAIN_N, "val": VAL_N, "test": 10**9}
    for group in ordered:
        placed = False
        for name in ("train", "val", "test"):
            if name != "test" and len(buckets[name]) + len(group) > limits[name]:
                continue
            for episode in group:
                episode["split"] = name
            buckets[name].extend(group)
            placed = True
            break
        if not placed:
            buckets["test"].extend(group)
    return buckets


def ids_for(config: dict, split: str, task_name: str) -> list[int]:
    key = TASK_KEY[task_name]
    return [int(item) for item in config["splits"][split][key]]


def init_state(task: dict, demo_id: int) -> np.ndarray:
    with h5py.File(task["hdf5"], "r") as handle:
        return np.asarray(handle[f"data/demo_{demo_id}/states"][0], dtype=np.float64)


def recorded_actions(task: dict, demo_id: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(task["hdf5"], "r") as handle:
        group = handle["data"][f"demo_{demo_id}"]
        actions = np.asarray(group["actions"], dtype=np.float32)
        state = np.asarray(group["states"][0], dtype=np.float64)
    return actions, state


class FrameDataset(Dataset):
    def __init__(self, episodes: list[dict]) -> None:
        self.frames: list[tuple[dict, int]] = []
        for episode in episodes:
            for step in range(len(episode["actions"])):
                self.frames.append((episode, step))

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode, step = self.frames[index]
        return {
            "images": torch.from_numpy(episode["images"][step]),
            "proprio": torch.from_numpy(episode["proprio"][step]),
            "actions": torch.from_numpy(episode["actions"][step]),
            "token_ids": torch.tensor(encode_task(episode["language"]), dtype=torch.long),
        }


def shift_images(images: torch.Tensor, max_shift: int = 4) -> torch.Tensor:
    if max_shift <= 0:
        return images
    batch, cameras, channels, height, width = images.shape
    padded = F.pad(
        images.reshape(batch * cameras, channels, height, width),
        (max_shift, max_shift, max_shift, max_shift),
    )
    shifted = []
    span = 2 * max_shift + 1
    for index in range(batch):
        top = int(torch.randint(0, span, ()).item())
        left = int(torch.randint(0, span, ()).item())
        crop = padded[index * cameras : (index + 1) * cameras, :, top : top + height, left : left + width]
        shifted.append(crop)
    return torch.stack(shifted, dim=0)


def to_chw(images: torch.Tensor) -> torch.Tensor:
    if images.ndim == 5 and images.shape[-1] == 3:
        return images.permute(0, 1, 4, 2, 3).contiguous()
    if images.ndim == 4 and images.shape[-1] == 3:
        return images.permute(0, 3, 1, 2).contiguous()
    return images


def prepare_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    augment: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    images = to_chw(batch["images"].to(device)).float().div_(255.0)
    if augment:
        images = shift_images(images)
    images = (images - mean) / std
    return images, batch["proprio"].to(device), batch["actions"].to(device), batch["token_ids"].to(device)


def normalize_proprio(proprio: torch.Tensor, stats: dict[str, np.ndarray]) -> torch.Tensor:
    mean = torch.as_tensor(stats["mean"], dtype=proprio.dtype, device=proprio.device)
    std = torch.as_tensor(stats["std"], dtype=proprio.dtype, device=proprio.device)
    return (proprio - mean) / std


def proprio_stats(episodes: list[dict]) -> dict[str, np.ndarray]:
    stacked = np.concatenate([episode["proprio"] for episode in episodes], axis=0)
    return {
        "mean": stacked.mean(axis=0).astype(np.float32),
        "std": np.clip(stacked.std(axis=0), 1e-3, None).astype(np.float32),
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
