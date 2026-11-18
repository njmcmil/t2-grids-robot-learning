"""Evaluate action-chunk retrieval with frozen GridS or GridS-FiLM camera features."""

from pathlib import Path
import argparse
import json
import hashlib
import sys
from time import perf_counter

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.spatial_correction import SpatialCorrection
from t2.data import CAMERAS, encode, preparation
from t2.simulation import configuration_size, local_xml, model_inputs, task_definition
from t2.train import load_policy

CHUNK = 8


def select_benchmark_cases(manifest):
    original = json.loads((ROOT / "configs/evaluation_protocol.json").read_text())
    blocked = []
    sizes = {}
    for task, info in manifest["tasks"].items():
        with h5py.File(ROOT / info["file"], "r") as file:
            nq = configuration_size(file["data/demo_0"].attrs["model_file"])
            sizes[task] = nq
            for split in ("train", "validation"):
                for name in manifest["splits"][split][task]:
                    ep = file["data"][name]
                    blocked.extend(
                        (
                            ep.attrs["init_state"][1 : 1 + nq],
                            ep["states"][1, 1 : 1 + nq],
                        )
                    )
    for case in original["cases"]:
        blocked.append(np.asarray(case["state"])[1 : 1 + sizes[case["task"]]])
    rng = np.random.default_rng(2027)
    selected = {}
    for task, nq in sizes.items():
        _, _, definition, _, get_path, _, _ = task_definition(ROOT, task)
        path = (
            Path(get_path("init_states"))
            / definition.problem_folder
            / definition.init_states_file
        )
        states = np.asarray(torch.load(path, weights_only=False))
        selected[task] = {}
        for i in rng.permutation(len(states)):
            qpos = states[i, 1 : 1 + nq]
            if any(
                qpos.shape == q.shape and np.allclose(qpos, q, rtol=0, atol=1e-4)
                for q in blocked
            ):
                continue
            selected[task][f"benchmark_{i}"] = states[i].tolist()
            blocked.append(qpos)
            if len(selected[task]) == 10:
                break
        if len(selected[task]) != 10:
            raise ValueError(f"Only {len(selected[task])} eligible starts for {task}")
    return selected


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("grids", "grids_film"), default="grids")
    parser.add_argument("--seed", type=int, choices=(0, 1), default=0)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--all-validation",
        action="store_true",
        help="Run all six validation episodes for each task.",
    )
    modes.add_argument(
        "--benchmark-20",
        action="store_true",
        help="Run 10 fresh benchmark starts per task.",
    )
    modes.add_argument(
        "--paired-20",
        action="store_true",
        help="Compare on the first 10 original cases per task using their original seeds.",
    )
    modes.add_argument(
        "--remaining-validation",
        action="store_true",
        help="Evaluate the remaining four validation scenes per task.",
    )
    modes.add_argument(
        "--paired-40",
        action="store_true",
        help="Run all 20 original starts per task for this checkpoint.",
    )
    args = parser.parse_args()
    paired_mode = args.paired_20 or args.paired_40
    manifest, _, _ = preparation(ROOT)
    selection = json.loads((ROOT / "configs/final_selection.json").read_text())
    item = next(
        c
        for c in selection["checkpoints"]
        if c["variant"] == args.variant and c["training_seed"] == args.seed
    )
    path = ROOT / item["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    model, saved, device = load_policy(path)
    extractor = SpatialCorrection(model)
    output = ROOT / "results/development/chunk_retrieval"
    if (args.variant, args.seed) != ("grids", 0):
        output = output / f"{args.variant}_seed{args.seed}"
    report_name = (
        "remaining_validation.json" if args.remaining_validation else "validation.json"
    )
    if args.all_validation:
        report_name = "all_validation.json"
    output.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": item["sha256"], "chunk": CHUNK, "rollouts": []}
    benchmark_mode = args.benchmark_20 or paired_mode
    if benchmark_mode:
        if paired_mode:
            protocol_bytes = (ROOT / "configs/evaluation_protocol.json").read_bytes()
            assert (
                hashlib.sha256(protocol_bytes).hexdigest()
                == selection["protocol_sha256"]
            )
            protocol = json.loads(protocol_bytes)
            count_per_task = 20 if args.paired_40 else 10
            paired = {
                task: {
                    c["case_id"]: c
                    for c in [c for c in protocol["cases"] if c["task"] == task][
                        :count_per_task
                    ]
                }
                for task in manifest["tasks"]
            }
            assert all(len(group) == count_per_task for group in paired.values())
            cases = {
                task: {name: c["state"] for name, c in group.items()}
                for task, group in paired.items()
            }
            report_name = f"paired_{count_per_task * len(paired)}.json"
            report["protocol_sha256"] = selection["protocol_sha256"]
        else:
            cases = select_benchmark_cases(manifest)
            report_name = "benchmark_20.json"
            report["selection_seed"] = 2027
            report["rollout_seed"] = 0
        report["cases"] = {task: list(group) for task, group in cases.items()}
        if (output / report_name).exists():
            existing = json.loads((output / report_name).read_text())
            if any(existing.get(k) != v for k, v in report.items() if k != "rollouts"):
                raise ValueError("Existing follow-up report has different settings.")
            report = existing
    for task, info in manifest["tasks"].items():
        _, _, definition, bddl, get_path, Env, suite = task_definition(ROOT, task)
        assert definition.language == info["language_instruction"]
        vocab = saved["config"]["vocab"]
        instruction = encode(definition.language, vocab).to(device)
        features, phases, commands, lengths = [], [], [], []
        with h5py.File(ROOT / info["file"], "r") as file:
            for name in manifest["splits"]["train"][task]:
                ep = file["data"][name]
                actions = ep["actions"][1:]
                n = len(actions)
                images = {
                    c: torch.from_numpy(ep["obs"][c][:-1])
                    .to(device)
                    .permute(0, 3, 1, 2)
                    .float()
                    .div(255)
                    for c in CAMERAS
                }
                _, tokens = extractor.predict(images, instruction.expand(n, -1))
                features.append(tokens)
                phases.extend(np.arange(n) / n)
                commands.append(
                    actions[np.minimum(np.arange(n)[:, None] + np.arange(CHUNK), n - 1)]
                )
                lengths.extend([n] * n)
            bank = torch.cat(features)
            # Weight visual differences by variation observed in training.
            scale = bank.std(0).clamp_min(0.1)
            bank = bank / scale
            phase = torch.tensor(phases, device=device)
            actions = np.concatenate(commands)
            validation = manifest["splits"]["validation"][task]
            scenes = validation[2:] if args.remaining_validation else validation[:2]
            if args.all_validation:
                scenes = validation
            if benchmark_mode:
                scenes = cases[task]
            for name in scenes:
                if benchmark_mode and any(
                    r["task"] == task and r["episode"] == name
                    for r in report["rollouts"]
                ):
                    continue
                settings = (
                    dict(control_freq=20, horizon=300, ignore_done=False)
                    if paired_mode
                    else {}
                )
                env = Env(
                    bddl_file_name=str(bddl),
                    camera_heights=128,
                    camera_widths=128,
                    **settings,
                )
                try:
                    seed = paired[task][name]["seed"] if paired_mode else 0
                    env.seed(seed)
                    env.reset()
                    if benchmark_mode:
                        state = np.asarray(cases[task][name])
                    else:
                        ep = file["data"][name]
                        env.reset_from_xml_string(
                            local_xml(ep.attrs["model_file"], get_path, suite)
                        )
                        env.sim.reset()
                        state = ep["states"][1]
                    obs = env.set_init_state(state)
                    assert not env.check_success()
                    low, high = env.env.action_spec
                    progress = 0.0
                    query_times = []
                    for step in range(1, 301):
                        offset = (step - 1) % CHUNK
                        if offset == 0:
                            if args.paired_40:
                                if device.type == "cuda":
                                    torch.cuda.synchronize(device)
                                start = perf_counter()
                            images, text = model_inputs(
                                obs, definition.language, vocab, device
                            )
                            _, tokens = extractor.predict(images, text)
                            distance = (bank - tokens / scale).square().mean(1)
                            # Compare similarly timed parts of complete demonstrations.
                            score = distance + 4 * (phase - progress).square()
                            index = int(score.argmin())
                            sequence = actions[index]
                            progress = min(
                                1.0,
                                max(progress, float(phase[index]))
                                + CHUNK / lengths[index],
                            )
                        if args.paired_40 and offset == 0:
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                            query_times.append((perf_counter() - start) * 1000)
                        obs, _, done, _ = env.step(
                            np.clip(sequence[offset], low, high).tolist()
                        )
                        success = bool(env.check_success())
                        if success or done:
                            break
                finally:
                    env.close()
                row = dict(task=task, episode=name, success=success, steps=step)
                if paired_mode:
                    if args.paired_40:
                        row.update(
                            tokens_per_camera=model.tokens_per_camera,
                            query_ms=query_times,
                            latency_scope="Batch-one online chunk query including image preprocessing and retrieval; excludes simulator step.",
                        )
                    original_path = (
                        ROOT
                        / "results/final"
                        / f"{args.variant}_seed{args.seed}"
                        / f"{name}.json"
                    )
                    original = json.loads(original_path.read_text())
                    assert original["protocol_sha256"] == selection["protocol_sha256"]
                    assert (
                        original["checkpoint_sha256"] == item["sha256"]
                        and original["seed"] == seed
                    )
                    row.update(seed=seed, original_success=original["success"])
                report["rollouts"].append(row)
                target = output / report_name
                temp = target.with_suffix(".tmp")
                temp.write_text(json.dumps(report, indent=2))
                temp.replace(target)
                print(row, flush=True)
    extractor.hook.remove()
    if benchmark_mode:
        wins = sum(r["success"] for r in report["rollouts"])
        total = sum(len(group) for group in cases.values())
        assert len(report["rollouts"]) == total
        print(f"Overall success: {wins}/{total} ({wins / total:.0%})")


if __name__ == "__main__":
    main()
