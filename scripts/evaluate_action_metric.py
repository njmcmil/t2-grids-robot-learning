"""Compare five frozen encoders with a training-only action-sequence retrieval head."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from t2.data import CAMERAS, encode, preparation
from t2.models import VARIANTS
from t2.simulation import labelled_frame, local_xml, model_inputs, task_definition
from t2.train import load_policy

CHUNK = 8
TIMEOUT = 300


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


class Retriever:
    def __init__(self, model):
        self.model = model
        self.parts = []
        self.hook = model.token_projection.register_forward_hook(self.capture)

    def capture(self, module, inputs, tokens):
        if tokens.shape[1] == 256:
            # Keep the same retrieval-vector size after the dense encoder runs.
            grid = tokens.transpose(1, 2).reshape(-1, 64, 16, 16)
            tokens = F.avg_pool2d(grid, 4).flatten(2).transpose(1, 2)
        self.parts.append(tokens.flatten(1))

    def features(self, images, text):
        self.parts.clear()
        self.model(images, text)
        return torch.cat(self.parts, dim=1)

    def fit(self, manifest, vocab, device):
        features, commands, phases, lengths, instructions = [], [], [], [], []
        for task, info in manifest["tasks"].items():
            words = encode(info["language_instruction"], vocab).to(device)
            with h5py.File(ROOT / info["file"], "r") as f:
                for name in manifest["splits"]["train"][task]:
                    ep = f["data"][name]
                    actions = ep["actions"][1:]
                    n = len(actions)
                    for start in range(0, n, 16):
                        end = min(n, start + 16)
                        images = {c: torch.from_numpy(ep["obs"][c][start:end]).to(device)
                                  .permute(0, 3, 1, 2).float().div(255) for c in CAMERAS}
                        features.append(self.features(images, words.expand(end-start, -1)))
                    indices = np.minimum(np.arange(n)[:, None] + np.arange(CHUNK), n-1)
                    commands.append(actions[indices])
                    phases.extend(np.arange(n) / n)
                    lengths.extend([n] * n)
                    instructions.extend([info["language_instruction"]] * n)
        self.commands = np.concatenate(commands)
        self.phase = torch.tensor(phases, dtype=torch.float32, device=device)
        self.lengths = lengths
        self.instructions = np.array(instructions)
        x = torch.cat(features)
        self.center = x.mean(0)
        self.scale = x.std(0).clamp_min(0.1)
        self.bank = (x - self.center) / self.scale
        # Fit a supervised distance: examples with similar upcoming movements
        # should be close. All labels and feature scales come from training episodes.
        y = torch.as_tensor(self.commands.reshape(len(x), -1), device=device).float()
        y = (y - y.mean(0)) / y.std(0).clamp_min(0.1)
        design = self.bank.double()
        gram = design.T @ design
        gram.diagonal().add_(gram.diagonal().mean() * 0.1)
        self.weights = torch.linalg.solve(gram, design.T @ y.double()).float()
        self.motion = self.bank @ self.weights
        self.visual_norm = self.bank.square().mean(1)
        self.motion_norm = self.motion.square().mean(1)

    def query(self, images, text, instruction, progress):
        x = (self.features(images, text) - self.center) / self.scale
        y = x @ self.weights
        # Query-only terms in squared distances are equal for every candidate.
        visual = self.visual_norm - 2 * (self.bank @ x[0]) / x.shape[1]
        motion = self.motion_norm - 2 * (self.motion @ y[0]) / y.shape[1]
        score = visual + motion + 4 * (self.phase - progress).square()
        if self.model.variant != "vision_only":
            allowed = torch.from_numpy(self.instructions == instruction).to(score.device)
            if not allowed.any():
                raise ValueError(f"Unknown instruction: {instruction}")
            score = score.masked_fill(~allowed, torch.inf)
        i = int(score.argmin())
        next_progress = min(1., max(progress, float(self.phase[i])) + CHUNK/self.lengths[i])
        return self.commands[i], next_progress


def validation_cases(manifest):
    cases = []
    for task, names in manifest["splits"]["validation"].items():
        with h5py.File(ROOT / manifest["tasks"][task]["file"], "r") as f:
            for name in names:
                ep = f["data"][name]
                cases.append(dict(task=task, case_id=task+"_"+name, seed=0,
                                  state=ep["states"][1], xml=ep.attrs["model_file"]))
    return cases


def rollout(policy, case, vocab, device, video):
    import imageio.v2 as imageio
    task = case["task"]
    _, _, definition, bddl, get_path, Env, suite = task_definition(ROOT, task)
    if "bddl_sha256" in case:
        assert sha(bddl) == case["bddl_sha256"]
    env = Env(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128,
              control_freq=20, horizon=TIMEOUT, ignore_done=False)
    writer = None
    try:
        env.seed(case["seed"])
        env.reset()
        if "xml" in case:
            env.reset_from_xml_string(local_xml(case["xml"], get_path, suite))
            env.sim.reset()
        state = np.asarray(case["state"])
        obs = env.set_init_state(state)
        assert np.allclose(env.get_sim_state()[1:], state[1:], atol=1e-6, rtol=0)
        assert not env.check_success()
        low, high = env.env.action_spec
        progress, times = 0., []
        if video:
            writer = imageio.get_writer(str(video), fps=20)
            writer.append_data(labelled_frame(obs,task,case["seed"],0,False,"Autonomous retrieval"))
        for step in range(1, TIMEOUT + 1):
            offset = (step - 1) % CHUNK
            if offset == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = perf_counter()
                images, text = model_inputs(obs, definition.language, vocab, device)
                commands, progress = policy.query(images, text, definition.language, progress)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                times.append((perf_counter()-start)*1000)
            obs, _, done, _ = env.step(np.clip(commands[offset], low, high).tolist())
            success = bool(env.check_success())
            if writer:
                writer.append_data(labelled_frame(obs,task,case["seed"],step,success,"Autonomous retrieval"))
            if success or done:
                break
        return dict(case_id=case["case_id"], task=task, seed=case["seed"], success=success,
                    steps=step, query_ms=times, tokens_per_camera=policy.model.tokens_per_camera)
    finally:
        if writer:
            writer.close()
        env.close()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("preflight", "validation", "run-400"):
        mode.add_argument("--"+name, action="store_true")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int, choices=[0,1])
    args = parser.parse_args()
    if args.run_400 and (args.variant is not None or args.seed is not None):
        parser.error("--run-400 evaluates all five variants and both seeds")
    manifest, _, digest = preparation(ROOT)
    selection = json.loads((ROOT/"configs/final_selection.json").read_text())
    protocol_path = ROOT/"configs/evaluation_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    assert sha(protocol_path) == selection["protocol_sha256"]
    assert protocol["split_manifest_sha256"] == digest and protocol["timeout_steps"] == TIMEOUT
    cases = protocol["cases"] if args.run_400 else validation_cases(manifest)
    if args.run_400:
        assert len({c["case_id"] for c in cases}) == 40
        assert Counter(c["task"] for c in cases) == {task:20 for task in manifest["tasks"]}
    pairs = selection["checkpoints"]
    assert len(pairs) == 10 and {(i["variant"], i["training_seed"]) for i in pairs} == {(v,s) for v in VARIANTS for s in (0,1)}
    folder = ROOT/"results/action_metric_v1"/("paired_400" if args.run_400 else "validation")
    summary = []
    for item in pairs:
        variant, seed = item["variant"], item["training_seed"]
        if args.variant and args.variant != variant or args.seed is not None and args.seed != seed:
            continue
        path = ROOT/item["path"]
        assert sha(path) == item["sha256"]
        model, saved, device = load_policy(path)
        assert saved["config"]["manifest_sha256"] == digest
        vocab = saved["config"]["vocab"]
        retriever = Retriever(model)
        retriever.fit(manifest, vocab, device)
        if args.preflight:
            for task, info in manifest["tasks"].items():
                with h5py.File(ROOT/info["file"], "r") as f:
                    ep = f["data"][manifest["splits"]["validation"][task][0]]
                    imgs = {c:torch.from_numpy(ep["obs"][c][0]).permute(2,0,1)[None].to(device).float()/255 for c in CAMERAS}
                    words = encode(info["language_instruction"],vocab)[None].to(device)
                    chunk, _ = retriever.query(imgs,words,info["language_instruction"],0.)
                    assert chunk.shape == (8,7) and np.isfinite(chunk).all()
            print(variant, seed, "both tasks passed", flush=True)
            retriever.hook.remove()
            continue
        dest = folder/f"{variant}_seed{seed}"
        dest.mkdir(parents=True, exist_ok=True)
        identity = dict(checkpoint=item["sha256"], protocol=sha(protocol_path), implementation=sha(Path(__file__)), manifest=digest)
        records = []
        for case in cases:
            output = dest/(case["case_id"]+".json")
            if output.exists():
                r = json.loads(output.read_text())
                assert r["identity"] == identity and r["case_id"] == case["case_id"]
            else:
                r = rollout(retriever, case, vocab, device, output.with_suffix(".mp4") if args.run_400 else None)
                r.update(identity=identity,variant=variant,training_seed=seed)
                save(output, r)
            records.append(r)
            print(variant, seed, r["case_id"], r["success"], flush=True)
        for task in manifest["tasks"]:
            group = [r for r in records if r["task"] == task]
            times = [t for r in group for t in r["query_ms"]]
            wins = sum(r["success"] for r in group)
            summary.append(dict(variant=variant,seed=seed,task=task,trials=len(group),successes=wins,
                                rate=wins/len(group),query_median_ms=float(np.median(times)),
                                query_p95_ms=float(np.percentile(times,95)),tokens_per_camera=model.tokens_per_camera))
        retriever.hook.remove()
    if args.run_400:
        assert sum(r["trials"] for r in summary) == 400
        save(folder/"summary.json",summary)


if __name__ == "__main__":
    main()

