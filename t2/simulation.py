"""Simulator integration. Import torch and torchvision before LIBERO native libraries."""

from pathlib import Path
import torch
import torchvision
import numpy as np
import h5py, json, sys, hashlib, time
import xml.etree.ElementTree as ET
from .data import CAMERAS, image_tensor, encode, preparation
from .train import load_policy


def simulator_tools(root):
    sys.path.insert(0, str(Path(root) / "LIBERO"))
    import robosuite
    import robosuite.macros as macros

    macros.IMAGE_CONVENTION = "opengl"
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    return benchmark, get_libero_path, OffScreenRenderEnv, robosuite


def task_definition(root, task_name):
    benchmark, get_path, Env, robosuite = simulator_tools(root)
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    i = next(
        i for i in range(suite.get_num_tasks()) if suite.get_task(i).name == task_name
    )
    task = suite.get_task(i)
    bddl = Path(get_path("bddl_files")) / task.problem_folder / task.bddl_file
    return suite, i, task, bddl, get_path, Env, robosuite


def configuration_size(xml):
    body = ET.fromstring(xml).find("worldbody")
    return sum(
        {"free": 7, "ball": 4, "slide": 1, "hinge": 1}[j.get("type", "hinge")]
        for j in body.findall(".//joint")
    ) + 7 * len(body.findall(".//freejoint"))


def local_xml(xml, get_path, robosuite):
    tree = ET.fromstring(xml)
    for node in tree.findall("./asset/*"):
        old = node.get("file")
        if old is None:
            continue
        if "/robosuite/" in old:
            path = (
                Path(robosuite.__file__).resolve().parent
                / old.rsplit("/robosuite/", 1)[1]
            )
        elif "/assets/" in old:
            path = Path(get_path("assets")) / old.split("/assets/", 1)[1]
        else:
            path = Path(old)
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        node.set("file", str(path))
    return ET.tostring(tree, encoding="unicode")


def development_case(root, task_name):
    root = Path(root)
    manifest, _, _ = preparation(root)
    name = manifest["splits"]["train"][task_name][0]
    with h5py.File(root / manifest["tasks"][task_name]["file"], "r") as f:
        episode = f["data"][name]
        return {
            "task": task_name,
            "seed": 0,
            "state": episode["states"][1].tolist(),
            "source_episode": name,
            "model_xml": str(episode.attrs["model_file"]),
            "kind": "development_training_scene",
            "case_id": f"{task_name}_{name}",
        }


def prepare_protocol(root, success_target, seed=2026):
    """Select common benchmark states without executing policies or viewing outcomes."""
    root = Path(root)
    manifest, _, digest = preparation(root)
    if not 0 < success_target <= 1:
        raise ValueError(
            "Choose and justify a success-rate target before final testing."
        )
    rng = np.random.default_rng(seed)
    cases = []
    audit = {}
    for task_name in manifest["tasks"]:
        suite, i, task, bddl, get_path, _, _ = task_definition(root, task_name)
        init_path = (
            Path(get_path("init_states")) / task.problem_folder / task.init_states_file
        )
        # Trusted local LIBERO initial-state file, not an arbitrary downloaded pickle.
        states = np.asarray(torch.load(init_path, weights_only=False))

        # Collect familiar robot and object positions before choosing the final starting scenes.
        blocked = []
        for source_task in manifest["tasks"]:
            with h5py.File(root / manifest["tasks"][source_task]["file"], "r") as f:
                source_nq = configuration_size(f["data/demo_0"].attrs["model_file"])
                if source_task == task_name:
                    nq = source_nq
                for split in ("train", "validation"):
                    for name in manifest["splits"][split][source_task]:
                        ep = f["data"][name]
                        blocked.extend(
                            [
                                np.asarray(ep.attrs["init_state"])[1 : 1 + source_nq],
                                ep["states"][1, 1 : 1 + source_nq],
                            ]
                        )
        selected = []
        for idx in rng.permutation(len(states)):
            state = states[int(idx)]
            config = state[1 : 1 + nq]
            if any(
                config.shape == other.shape
                and np.allclose(config, other, rtol=0, atol=1e-4)
                for other in blocked
            ):
                continue
            if any(np.allclose(config, other, rtol=0, atol=1e-4) for other in selected):
                continue
            selected.append(config)
            cases.append(
                {
                    "task": task_name,
                    "seed": seed + len(selected),
                    "state": state.tolist(),
                    "benchmark_state_index": int(idx),
                    "kind": "heldout_benchmark_state",
                    "bddl_sha256": hashlib.sha256(bddl.read_bytes()).hexdigest(),
                    "case_id": f"{task_name}_benchmark_{int(idx)}",
                }
            )
            if len(selected) == 20:
                break
        if len(selected) != 20:
            raise RuntimeError(
                f"{task_name}: only {len(selected)} eligible distinct states. Agree a new held-out protocol; do not duplicate states silently."
            )
        audit[task_name] = {
            "selected": 20,
            "excluded_reference_configurations": len(blocked),
            "initial_state_file_sha256": hashlib.sha256(
                init_path.read_bytes()
            ).hexdigest(),
        }
    protocol = {
        "split_manifest_sha256": digest,
        "selection_seed": seed,
        "timeout_steps": 300,
        "rollouts_per_task": 20,
        "success_target": success_target,
        "p95_latency_target_ms": 250,
        "audit": audit,
        "cases": cases,
        "limitation": "Initial qpos similarity audit at atol=1e-4; collection-session independence not established. No policy outcome used for selection.",
    }
    path = root / "configs/evaluation_protocol.json"
    text = json.dumps(protocol, indent=2) + "\n"
    if path.exists() and path.read_text() != text:
        raise FileExistsError(
            "Protocol already exists with different contents; preserve the original experiment."
        )
    path.write_text(text)
    return path


def model_inputs(obs, instruction, vocab, device):
    mapping = {
        "agentview_rgb": "agentview_image",
        "eye_in_hand_rgb": "robot0_eye_in_hand_image",
    }
    images = {c: image_tensor(obs[mapping[c]]).unsqueeze(0).to(device) for c in CAMERAS}
    return images, encode(instruction, vocab).unsqueeze(0).to(device)


def labelled_frame(obs, task, seed, step, success, kind):
    from PIL import Image, ImageDraw

    frame = Image.new("RGB", (512, 576), "black")
    frame.paste(
        Image.fromarray(obs["agentview_image"][::-1]).resize((512, 512)), (0, 64)
    )
    draw = ImageDraw.Draw(frame)
    draw.text((8, 4), kind, fill="white")
    draw.text((8, 20), task, fill="white")
    draw.text(
        (8, 36), f"seed={seed} step={step} simulator_success={success}", fill="white"
    )
    return np.asarray(frame)


def rollout(
    root, checkpoint, case, output_dir, timeout=300, video=True, protocol_hash=None
):
    import imageio.v2 as imageio

    root = Path(root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / (case["case_id"] + ".json")
    if result_path.exists():
        raise FileExistsError(result_path)
    model, saved, device = load_policy(checkpoint)
    if saved["config"]["smoke_test"]:
        raise ValueError("Do not evaluate a smoke-test checkpoint as a trained policy.")
    manifest, norm, digest = preparation(root)
    assert digest == saved["config"]["manifest_sha256"]
    suite, i, task, bddl, get_path, Env, robosuite = task_definition(root, case["task"])
    if "bddl_sha256" in case:
        assert hashlib.sha256(bddl.read_bytes()).hexdigest() == case["bddl_sha256"]
    mean = np.asarray(saved["normalization"]["mean"], np.float32)
    scale = np.asarray(saved["normalization"]["scale"], np.float32)
    env = None
    writer = None
    result = None
    try:
        env = Env(
            bddl_file_name=str(bddl),
            camera_heights=128,
            camera_widths=128,
            control_freq=20,
            horizon=timeout,
            ignore_done=False,
        )
        env.seed(case["seed"])
        env.reset()
        if "model_xml" in case:
            env.reset_from_xml_string(local_xml(case["model_xml"], get_path, robosuite))
            env.sim.reset()
        state = np.asarray(case["state"])
        obs = env.set_init_state(state)
        assert np.allclose(env.get_sim_state()[1:], state[1:], rtol=0, atol=1e-6)
        if env.check_success():
            raise RuntimeError(
                "Starting state is already successful; report invalid case, not policy success."
            )
        low, high = env.env.action_spec
        inputs = model_inputs(obs, task.language, saved["config"]["vocab"], device)

        def synchronize():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        # Batch-one model-only latency, no rendering, input preparation, or CPU transfer.
        with torch.inference_mode():
            for _ in range(10):
                model(*inputs)
            synchronize()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            times = []
            for _ in range(100):
                synchronize()
                start = time.perf_counter()
                model(*inputs)
                synchronize()
                times.append((time.perf_counter() - start) * 1000)
        peak = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        )
        if video:
            writer = imageio.get_writer(
                str(output_dir / (case["case_id"] + ".mp4")), fps=20
            )
            writer.append_data(
                labelled_frame(obs, case["task"], case["seed"], 0, False, case["kind"])
            )
        clipped = 0
        success = False
        termination = "timeout"
        for step in range(1, timeout + 1):
            inputs = model_inputs(obs, task.language, saved["config"]["vocab"], device)
            with torch.inference_mode():
                raw = model(*inputs)[0].cpu().numpy() * scale + mean
            if not np.isfinite(raw).all():
                raise RuntimeError("Invalid action")

            # Keep the predicted command within the controller limits before moving the robot.
            action = np.clip(raw, low, high)
            clipped += int(np.any(raw != action))
            obs, _, done, _ = env.step(action.tolist())
            success = bool(env.check_success())
            if writer:
                writer.append_data(
                    labelled_frame(
                        obs, case["task"], case["seed"], step, success, case["kind"]
                    )
                )
            if success:
                termination = "success"
                break
            if done:
                termination = "environment_done"
                break
        result = {
            "case_id": case["case_id"],
            "kind": case["kind"],
            "task": case["task"],
            "seed": case["seed"],
            "variant": saved["config"]["variant"],
            "training_seed": saved["config"]["training_seed"],
            "success": success,
            "steps": step,
            "termination": termination,
            "clipped_steps": clipped,
            "checkpoint_sha256": hashlib.sha256(
                Path(checkpoint).read_bytes()
            ).hexdigest(),
            "tokens_per_camera": model.tokens_per_camera,
            "parameters": saved["config"]["parameters"],
            "latency_median_ms": float(np.median(times)),
            "latency_p95_ms": float(np.percentile(times, 95)),
            "latency_samples_ms": times,
            "peak_cuda_allocated_bytes": peak,
            "latency_scope": "model only; batch 1; 10 warmups + 100 synchronized measurements; excludes preprocessing and simulator",
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        }
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
    if protocol_hash is not None:
        result["protocol_sha256"] = protocol_hash
    result_path.write_text(json.dumps(result, indent=2))
    print(case["task"], case["case_id"], "success=", result["success"], flush=True)
    return result
