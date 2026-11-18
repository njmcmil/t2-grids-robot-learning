from pathlib import Path
import csv, json, hashlib
import numpy as np


def summarize(root):
    root = Path(root)
    selection_path = root / "configs/final_selection.json"
    if not selection_path.exists():
        print("No frozen selection. Final results have not been produced.")
        return []
    selection = json.loads(selection_path.read_text())
    raw = (root / "configs/evaluation_protocol.json").read_bytes()
    protocol = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == selection["protocol_sha256"]

    # Build one row per model, training seed, and task.
    rows = []
    for chosen in selection["checkpoints"]:
        for task in protocol["audit"]:
            cases = [c for c in protocol["cases"] if c["task"] == task]
            records = []
            for case in cases:
                path = (
                    root
                    / "results/final"
                    / f"{chosen['variant']}_seed{chosen['training_seed']}"
                    / (case["case_id"] + ".json")
                )
                if not path.exists():
                    continue
                record = json.loads(path.read_text())
                assert record["protocol_sha256"] == digest
                assert record["checkpoint_sha256"] == chosen["sha256"]
                records.append(record)
            complete = len(records) == 20
            samples = [x for r in records for x in r["latency_samples_ms"]]
            peaks = [
                r["peak_cuda_allocated_bytes"]
                for r in records
                if r["peak_cuda_allocated_bytes"] is not None
            ]
            rows.append(
                {
                    "variant": chosen["variant"],
                    "training_seed": chosen["training_seed"],
                    "task": task,
                    "completed": len(records),
                    "expected": 20,
                    "status": "complete" if complete else "incomplete",
                    "successes": sum(r["success"] for r in records),
                    "success_rate": (
                        sum(r["success"] for r in records) / 20 if complete else None
                    ),
                    "median_latency_ms": float(np.median(samples)) if samples else None,
                    "p95_latency_ms": (
                        float(np.percentile(samples, 95)) if samples else None
                    ),
                    "peak_cuda_allocated_bytes": max(peaks) if peaks else None,
                    "tokens_per_camera": (
                        records[0]["tokens_per_camera"] if records else None
                    ),
                }
            )
    out = root / "results"
    out.mkdir(exist_ok=True)
    with (out / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    # Average the task scores for each seed, then combine the two seeds.
    aggregates = []
    for variant in sorted({r["variant"] for r in rows}):
        group = [r for r in rows if r["variant"] == variant]
        if any(r["status"] != "complete" for r in group):
            aggregates.append({"variant": variant, "status": "incomplete"})
            continue
        seed_rates = [
            np.mean([r["success_rate"] for r in group if r["training_seed"] == seed])
            for seed in (0, 1)
        ]
        mean = float(np.mean(seed_rates))
        aggregates.append(
            {
                "variant": variant,
                "status": "complete",
                "mean_success_rate": mean,
                "seed_rates": seed_rates,
                "between_seed_sample_std": float(np.std(seed_rates, ddof=1)),
                "success_target": protocol["success_target"],
                "target_met": mean >= protocol["success_target"],
                "latency_target_met": all(r["p95_latency_ms"] <= 250 for r in group),
            }
        )
    (out / "aggregate.json").write_text(json.dumps(aggregates, indent=2))
    print(json.dumps(aggregates, indent=2))
    return rows
