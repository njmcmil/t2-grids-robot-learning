#!/usr/bin/env python3
"""Score the five ResNet-18 policies on 20 shared initial states."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resnet.data import TASKS, init_state, load_json  # noqa: E402
from resnet.simulate import demonstration_video, make_env, rollout, save_autonomous  # noqa: E402
from resnet.train import load_policy  # noqa: E402

TOKEN_BUDGET = {
    "vision_bc": {"tokens_per_camera": 64, "sequence_length": 129, "language": "none (learned CLS)"},
    "dense_language": {"tokens_per_camera": 64, "sequence_length": 129, "language": "goal-word byte stand-in"},
    "vanilla_grids": {"tokens_per_camera": 16, "sequence_length": 33, "language": "goal-word byte stand-in"},
    "grids_film": {"tokens_per_camera": 16, "sequence_length": 33, "language": "goal word inside FiLM and as a token"},
    "uniform_ablation": {"tokens_per_camera": 16, "sequence_length": 33, "language": "goal-word byte stand-in; coordinates fixed"},
}
WEIGHTS = ROOT / "weights" / "resnet_heldout"
OUT = ROOT / "results" / "resnet_heldout"
POLICY_DIR = ROOT / "videos" / "policy"
REPLAY_DIR = ROOT / "videos" / "replay"


def main() -> int:
    config = load_json(ROOT / "configs" / "resnet_heldout.json")
    target = float(config["success_target"])
    seed = int(config["env_seed"])
    demo_ids = [int(item) for item in config["heldout_demo_ids"]]
    timeout = int(config["timeout_steps"])
    variants = list(config["variants"])
    print(f"success_target {target} timeout_steps {timeout} env_seed {seed}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inits = {
        task["name"]: {demo_id: init_state(task, demo_id) for demo_id in demo_ids}
        for task in TASKS
    }
    records = []
    saved_success = set()
    saved_failure = set()
    for variant in variants:
        policy, stats, loaded = load_policy(WEIGHTS / f"{variant}.pt", device)
        if loaded != variant:
            raise RuntimeError(f"{variant} weight file says {loaded}")
        budget = TOKEN_BUDGET[variant]
        for task in TASKS:
            env = make_env(task["bddl"])
            try:
                for demo_id in demo_ids:
                    want_video = (variant, task["name"]) not in saved_success or task["name"] not in saved_failure
                    result = rollout(
                        env, policy, stats, variant, task["name"],
                        inits[task["name"]][demo_id], device, seed, want_video, timeout,
                    )
                    records.append(
                        {
                            "variant": variant,
                            "task": task["name"],
                            "demo_id": demo_id,
                            "seed": seed,
                            "success": result["success"],
                            "steps": result["steps"],
                            "success_step": result["success_step"],
                            "latency_ms": result["latency_ms"],
                            "failure_mode": result["failure_mode"],
                            "tokens_per_camera": budget["tokens_per_camera"],
                            "sequence_length": budget["sequence_length"],
                            "language_features": budget["language"],
                            "timeout_steps": timeout,
                            "success_target": target,
                        }
                    )
                    print(
                        f"  {variant} | {task['name']} | demo_{demo_id} | seed {seed} | "
                        f"success={int(result['success'])} steps={result['steps']}",
                        flush=True,
                    )
                    if result["frames"] and result["success"] and (variant, task["name"]) not in saved_success:
                        path = POLICY_DIR / f"autonomous_{variant}_{task['bddl'].stem}_demo_{demo_id}_seed{seed}.mp4"
                        save_autonomous(result["frames"], task["name"], seed, demo_id, variant, True, path)
                        saved_success.add((variant, task["name"]))
                    elif result["frames"] and not result["success"] and task["name"] not in saved_failure:
                        path = POLICY_DIR / f"autonomous_failure_{variant}_{task['bddl'].stem}_demo_{demo_id}_seed{seed}.mp4"
                        save_autonomous(result["frames"], task["name"], seed, demo_id, variant, False, path)
                        saved_failure.add(task["name"])
            finally:
                env.close()
        del policy
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = []
    for variant in variants:
        for task in TASKS:
            rows = [row for row in records if row["variant"] == variant and row["task"] == task["name"]]
            wins = sum(1 for row in rows if row["success"])
            rate = wins / len(rows)
            latency = float(np.median([row["latency_ms"] for row in rows]))
            modes = {}
            for row in rows:
                modes[row["failure_mode"]] = modes.get(row["failure_mode"], 0) + 1
            cell = {
                "variant": variant,
                "task": task["name"],
                "successes": wins,
                "rollouts": len(rows),
                "success_rate": rate,
                "success_target": target,
                "target_met": rate + 1e-9 >= target,
                "median_latency_ms": latency,
                "tokens_per_camera": TOKEN_BUDGET[variant]["tokens_per_camera"],
                "sequence_length": TOKEN_BUDGET[variant]["sequence_length"],
                "seed": seed,
                "failure_modes": modes,
            }
            summary.append(cell)
            print(f"  {variant} | {task['name']}: {wins}/{len(rows)} rate {rate:.3f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps({"config": config, "cells": summary}, indent=2), encoding="utf-8")
    for task, demo_id in (
        (TASKS[0], 0),
        (TASKS[1], 0),
        (TASKS[1], 1),
    ):
        path = REPLAY_DIR / f"demonstration_replay_{task['bddl'].stem}_demo_{demo_id}_seed{seed}.mp4"
        ok = demonstration_video(task, demo_id, seed, path)
        print(f"wrote {path} success={int(ok)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
