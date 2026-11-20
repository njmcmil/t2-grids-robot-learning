#!/usr/bin/env python3
"""Score saved ResNet-18 weights on the 20 benchmark states."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resnet.data import TASKS, VARIANTS, load_json  # noqa: E402
from resnet.simulate import benchmark_rollout, make_env  # noqa: E402
from resnet.train import load_policy  # noqa: E402

WEIGHTS = ROOT / "weights" / "resnet_protocol"
OUT = ROOT / "results" / "resnet_protocol" / "summary.json"


def main() -> int:
    config = load_json(ROOT / "configs" / "resnet_protocol.json")
    if int(config["split_seed"]) != 42 or int(config["timeout_steps"]) != 300 or int(config["rollouts_per_task"]) != 20:
        raise RuntimeError("protocol file does not match the 20-state, 300-step setup")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = {task["name"]: make_env(task["bddl"]) for task in TASKS}
    records = []
    loaded = {}
    try:
        for variant in VARIANTS:
            policy, stats, name = load_policy(WEIGHTS / f"{variant}.pt", device)
            if name != variant:
                raise RuntimeError(f"{variant} weight file says {name}")
            loaded[variant] = {"policy": policy, "stats": stats, "val_loss": torch.load(WEIGHTS / f"{variant}.pt", map_location="cpu", weights_only=False).get("val_loss")}
            for case in config["cases"]:
                task_name = case["task"].replace("_", " ")
                result = benchmark_rollout(
                    envs[task_name],
                    policy,
                    stats,
                    variant,
                    task_name,
                    case["state"],
                    int(case["seed"]),
                    device,
                    int(config["timeout_steps"]),
                )
                records.append(
                    {
                        "variant": variant,
                        "task": task_name,
                        "benchmark_state_index": case["benchmark_state_index"],
                        "seed": int(case["seed"]),
                        "success": result["success"],
                        "steps": result["steps"],
                        "latency_ms": result["latency_ms"],
                    }
                )
                print(
                    f"  {variant} {task_name} benchmark_{case['benchmark_state_index']} "
                    f"seed {case['seed']} success={int(result['success'])} steps={result['steps']}",
                    flush=True,
                )
    finally:
        for env in envs.values():
            env.close()

    summary = []
    for variant in VARIANTS:
        rows = [row for row in records if row["variant"] == variant]
        wins = sum(int(row["success"]) for row in rows)
        by = {}
        for task in TASKS:
            task_rows = [row for row in rows if row["task"] == task["name"]]
            task_wins = sum(int(row["success"]) for row in task_rows)
            by[task["name"]] = {
                "successes": task_wins,
                "rollouts": len(task_rows),
                "rate": task_wins / len(task_rows),
            }
        summary.append(
            {
                "variant": variant,
                "successes": wins,
                "rollouts": len(rows),
                "success_rate": wins / len(rows),
                "by_task": by,
                "val_loss": loaded[variant]["val_loss"],
                "weights": str(WEIGHTS / f"{variant}.pt"),
            }
        )
        print(f"RATE {variant} {wins}/{len(rows)} {wins / len(rows):.4f}", flush=True)
    payload = {
        "split_seed": 42,
        "training_seed": 0,
        "timeout_steps": 300,
        "rollouts_per_task": 20,
        "success": "env.check_success",
        "architecture": "ResNet-18 through layer3, shared transformer, one action per step",
        "retrieval": False,
        "variants": summary,
        "rollouts": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
