"""Simulator setup, closed-loop rollouts, and video writing."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

from resnet.data import IMAGENET_MEAN, IMAGENET_STD, normalize_proprio, to_chw
from resnet.models import GridSamplerPolicy
from resnet.text import encode_task


def make_env(bddl: Path, size: int = 128):
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=size,
        camera_widths=size,
    )
    env.seed(0)
    return env


def proprio_from_obs(obs: dict) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
            np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1),
        ]
    ).astype(np.float32)


def images_from_obs(obs: dict) -> np.ndarray:
    views = (
        np.asarray(obs["agentview_image"], dtype=np.uint8),
        np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8),
    )
    return np.stack(views, axis=0)


def replay_task(task: dict) -> list[dict]:
    """Restore each saved state and pair it with the action taken there."""
    language = task["name"]
    env = make_env(task["bddl"])
    try:
        env.reset()
        if env.language_instruction != language:
            raise RuntimeError(
                f"{task['bddl']} language is {env.language_instruction!r}, expected {language!r}"
            )
        episodes = []
        import h5py

        with h5py.File(task["hdf5"], "r") as handle:
            from resnet.data import demo_ids

            stored = json.loads(handle["data"].attrs["problem_info"])["language_instruction"]
            if stored != language:
                raise RuntimeError(f"HDF5 language is {stored!r}, expected {language!r}")
            for demo_id in demo_ids(task["hdf5"]):
                group = handle["data"][f"demo_{demo_id}"]
                actions = np.asarray(group["actions"], dtype=np.float32)
                states = np.asarray(group["states"], dtype=np.float64)
                init_state = states[0]
                env.reset()
                frames = []
                proprios = []
                success_step = None
                for step in range(len(actions)):
                    obs = env.set_init_state(states[step])
                    frames.append(images_from_obs(obs))
                    proprios.append(proprio_from_obs(obs))
                    if success_step is None and bool(env.check_success()):
                        success_step = step
                episode = {
                    "id": demo_id,
                    "language": language,
                    "init_state": init_state,
                    "images": np.stack(frames, axis=0),
                    "proprio": np.stack(proprios, axis=0),
                    "actions": actions,
                    "success": success_step is not None,
                    "success_step": -1 if success_step is None else int(success_step),
                    "group": tuple(np.round(init_state, 3).tolist()),
                }
                episodes.append(episode)
                print(
                    f"  demonstration {language} demo_{demo_id}: "
                    f"recorded_state_success={int(episode['success'])} steps={len(frames)}",
                    flush=True,
                )
        return episodes
    finally:
        env.close()


def load_selected(task: dict, demo_ids: list[int]) -> list[dict]:
    import h5py

    env = make_env(task["bddl"])
    episodes = []
    try:
        with h5py.File(task["hdf5"], "r") as handle:
            for demo_id in demo_ids:
                group = handle["data"][f"demo_{demo_id}"]
                actions = np.asarray(group["actions"], dtype=np.float32)
                states = np.asarray(group["states"], dtype=np.float64)
                env.reset()
                frames = []
                proprios = []
                for step in range(len(actions)):
                    obs = env.set_init_state(states[step])
                    frames.append(images_from_obs(obs))
                    proprios.append(proprio_from_obs(obs))
                episodes.append(
                    {
                        "id": demo_id,
                        "language": task["name"],
                        "images": np.stack(frames),
                        "proprio": np.stack(proprios),
                        "actions": actions,
                    }
                )
                print(f"  loaded {task['name']} demo_{demo_id} steps={len(actions)}", flush=True)
    finally:
        env.close()
    return episodes


def failure_mode(start: np.ndarray, end: np.ndarray, success: bool) -> str:
    if success:
        return "success"
    moved = float(np.linalg.norm(end[:2] - start[:2]))
    if moved < 0.03:
        return "timeout_no_grasp"
    if end[2] < 0.85 or abs(float(end[0])) > 0.45:
        return "timeout_dropped_or_thrown"
    return "timeout_displaced"


def label_frame(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame).resize((512, 512), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    y = 6
    for line in lines:
        draw.rectangle((4, y, 8 + 7 * len(line), y + 14), fill=(0, 0, 0))
        draw.text((6, y), line, fill=(255, 255, 255))
        y += 16
    return np.asarray(image)


def write_mp4(frames: list[np.ndarray], path: Path, fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None
    for frame in frames:
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")


def overlay_clip(frames: list[np.ndarray], lines_for, path: Path) -> None:
    labeled = []
    for index, frame in enumerate(frames[::2]):
        labeled.append(label_frame(frame[0], lines_for(index, index == (len(frames[::2]) - 1))))
    if labeled:
        labeled[-1] = label_frame(frames[-1][0], lines_for(len(frames) - 1, True))
    write_mp4(labeled, path)


def _prepared_image(view: np.ndarray, device: torch.device, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    image = to_chw(torch.from_numpy(view).unsqueeze(0).to(device)).float().div_(255.0)
    return (image - mean) / std


def rollout(
    env,
    policy: GridSamplerPolicy,
    stats: dict,
    variant: str,
    task_name: str,
    init: np.ndarray,
    device: torch.device,
    seed: int,
    keep_frames: bool,
    timeout_steps: int = 600,
) -> dict:
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 1, 3, 1, 1)
    token_ids = torch.tensor([encode_task(task_name)], dtype=torch.long, device=device)
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(init)
    start_bowl = np.asarray(obs["akita_black_bowl_1_pos"], dtype=np.float64).copy()
    frames = []
    latencies = []
    success_step = None
    import time

    for step in range(timeout_steps):
        view = images_from_obs(obs)
        if keep_frames:
            frames.append(view)
        image = _prepared_image(view, device, mean, std)
        proprio = normalize_proprio(torch.from_numpy(proprio_from_obs(obs)).unsqueeze(0).to(device), stats)
        started = time.perf_counter()
        with torch.no_grad():
            raw, coords, shape = policy(image, proprio, token_ids, variant)
            action = torch.tanh(raw)[0].detach().cpu().numpy().astype(np.float32)
        latencies.append((time.perf_counter() - started) * 1000.0)
        obs, _reward, done, _info = env.step(action)
        if bool(env.check_success()):
            success_step = step
            if keep_frames:
                frames.append(images_from_obs(obs))
            break
        if done:
            break
    end_bowl = np.asarray(obs["akita_black_bowl_1_pos"], dtype=np.float64)
    success = success_step is not None
    return {
        "success": success,
        "steps": success_step + 1 if success else len(latencies),
        "success_step": success_step,
        "latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "failure_mode": failure_mode(start_bowl, end_bowl, success),
        "frames": frames,
        "sequence_length": int(shape[-2]) if shape is not None else 0,
        "coords": None if coords is None else coords.detach().cpu(),
    }


def save_autonomous(frames, task, seed, demo_id, variant, success, path: Path) -> None:
    def lines(_index, is_last):
        return [
            "kind: autonomous policy",
            f"task: {task}",
            f"variant: {variant}",
            f"seed: {seed}",
            f"init: demo_{demo_id}",
            f"simulator_success: {str(bool(success if is_last else False)).lower()}",
        ]

    overlay_clip(frames, lines, path)


def demonstration_video(task: dict, demo_id: int, seed: int, path: Path) -> bool:
    import h5py

    with h5py.File(task["hdf5"], "r") as handle:
        actions = np.asarray(handle[f"data/demo_{demo_id}/actions"], dtype=np.float32)
        state = np.asarray(handle[f"data/demo_{demo_id}/states"][0], dtype=np.float64)
    env = make_env(task["bddl"])
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(state)
    frames = []
    success = False
    try:
        for action in actions:
            frames.append(images_from_obs(obs))
            obs, _r, done, _i = env.step(action)
            success = bool(env.check_success())
            if success or done:
                frames.append(images_from_obs(obs))
                break
    finally:
        env.close()

    def lines(_index, is_last, task_name=task["name"], ok=success):
        return [
            "kind: demonstration replay",
            f"task: {task_name}",
            f"seed: {seed}",
            f"init: demo_{demo_id}",
            f"simulator_success: {str(bool(ok if is_last else False)).lower()}",
        ]

    overlay_clip(frames, lines, path)
    return success


def benchmark_rollout(env, policy, stats, variant, task_name, state, seed, device, timeout_steps: int) -> dict:
    import time

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 1, 3, 1, 1)
    token_ids = torch.tensor([encode_task(task_name)], dtype=torch.long, device=device)
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(np.asarray(state, dtype=np.float64))
    success = False
    steps = 0
    latencies = []
    for step in range(timeout_steps):
        image = _prepared_image(images_from_obs(obs), device, mean, std)
        proprio = normalize_proprio(torch.from_numpy(proprio_from_obs(obs)).unsqueeze(0).to(device), stats)
        started = time.perf_counter()
        with torch.no_grad():
            raw, _coords, _shape = policy(image, proprio, token_ids, variant)
            action = torch.tanh(raw)[0].detach().cpu().numpy().astype(np.float32)
        latencies.append((time.perf_counter() - started) * 1000.0)
        obs, _reward, done, _info = env.step(action)
        steps = step + 1
        if bool(env.check_success()):
            success = True
            break
        if done:
            break
    return {
        "success": success,
        "steps": steps,
        "latency_ms": float(np.mean(latencies)) if latencies else 0.0,
    }


def render_demo(task: dict, demo_id: int, size: int) -> list[np.ndarray]:
    import h5py

    env = make_env(task["bddl"], size=size)
    frames: list[np.ndarray] = []
    try:
        with h5py.File(task["hdf5"], "r") as handle:
            group = handle["data"][f"demo_{demo_id}"]
            actions = np.asarray(group["actions"], dtype=np.float32)
            init_state = np.asarray(group["states"][0], dtype=np.float64)
        env.reset()
        obs = env.set_init_state(init_state)
        for action in actions:
            frames.append(np.asarray(obs["agentview_image"], dtype=np.uint8))
            obs, _reward, done, _info = env.step(action)
            if bool(env.check_success()):
                frames.append(np.asarray(obs["agentview_image"], dtype=np.uint8))
                break
            if done:
                break
        return frames
    finally:
        env.close()


def side_by_side(left: list[np.ndarray], right: list[np.ndarray]) -> list[np.ndarray]:
    length = max(len(left), len(right))
    frames = []
    for index in range(length):
        a = left[min(index, len(left) - 1)]
        b = right[min(index, len(right) - 1)]
        frames.append(np.concatenate((a, b), axis=1))
    return frames
