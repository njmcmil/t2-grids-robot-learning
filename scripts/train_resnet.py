#!/usr/bin/env python3
"""Train the ResNet-18 bowl policies in simulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resnet.data import (  # noqa: E402
    ENCODER,
    TASKS,
    VARIANTS,
    FrameDataset,
    ids_for,
    load_json,
    read_cache,
    split_episodes,
    write_cache,
)
from resnet.simulate import load_selected, replay_task  # noqa: E402
from resnet.train import train_film, train_protocol_variant, train_variant  # noqa: E402

import torch  # noqa: E402


def merged_splits():
    per_task = []
    for task in TASKS:
        episodes = read_cache(task)
        splits = split_episodes(episodes)
        per_task.append((task, splits))
        print(
            f"Split {task['name']}: " + ", ".join(f"{name} {len(rows)}" for name, rows in splits.items()),
            flush=True,
        )
    merged = {
        name: [episode for _task, splits in per_task for episode in splits[name]]
        for name in ("train", "val", "test")
    }
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("replay", "train", "protocol", "film"), default="train", nargs="?")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)

    if args.stage == "replay":
        for task in TASKS:
            episodes = replay_task(task)
            path = write_cache(task, episodes)
            wins = sum(1 for episode in episodes if episode["success"])
            print(f"  cached {path}: demonstration success {wins}/{len(episodes)}", flush=True)
        return 0

    if args.stage == "train":
        merged = merged_splits()
        print(
            f"frames train {len(FrameDataset(merged['train']))} val {len(FrameDataset(merged['val']))}",
            flush=True,
        )
        for variant in args.variants:
            train_variant(
                variant,
                merged,
                device,
                args.epochs,
                ROOT / "weights" / "resnet_heldout" / f"{variant}.pt",
            )
        return 0

    if args.stage == "protocol":
        config = load_json(ROOT / "configs" / "resnet_protocol.json")
        by_task = {}
        for task in TASKS:
            train_ids = ids_for(config, "train", task["name"])
            val_ids = ids_for(config, "validation", task["name"])
            print(f"loading {task['name']} train {len(train_ids)} val {len(val_ids)}", flush=True)
            by_task[task["name"]] = {
                "train": load_selected(task, train_ids),
                "val": load_selected(task, val_ids),
            }
        splits = {
            name: [episode for task_name in by_task for episode in by_task[task_name][name]]
            for name in ("train", "val")
        }
        for variant in args.variants:
            train_protocol_variant(
                variant,
                splits,
                device,
                args.epochs,
                ROOT / "weights" / "resnet_protocol" / f"{variant}.pt",
            )
        return 0

    merged = merged_splits()
    train_film(merged, device, ENCODER, ROOT / "weights" / "resnet_film.pt", epochs=args.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
