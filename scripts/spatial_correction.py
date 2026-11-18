"""Fit a spatial correction to a frozen GridS model and compare on validation scenes."""
from pathlib import Path
import hashlib
import json
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from t2.data import CAMERAS, encode, preparation
from t2.simulation import local_xml, model_inputs, task_definition
from t2.train import load_policy


class SpatialCorrection:
    """Use the original camera tokens before attention averages them."""
    def __init__(self, model):
        self.model = model
        self.tokens = []
        self.hook = model.token_projection.register_forward_hook(self.capture)

    def capture(self, module, inputs, output):
        self.tokens.append(output.flatten(1))

    def predict(self, images, text):
        self.tokens.clear()
        base = self.model(images, text)
        return base, torch.cat(self.tokens, dim=1)


@torch.inference_mode()
def main():
    manifest, norm, _ = preparation(ROOT)
    selection = json.loads((ROOT / 'configs/final_selection.json').read_text())
    checkpoint = next(c for c in selection['checkpoints']
                      if c['variant'] == 'grids' and c['training_seed'] == 0)
    path = ROOT / checkpoint['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint['sha256']
    model, saved, device = load_policy(path)
    corrected = SpatialCorrection(model)
    output = ROOT / 'results/development/spatial_correction'
    output.mkdir(parents=True, exist_ok=True)
    vocab = saved['config']['vocab']
    mean = np.asarray(norm['mean'])
    scale = np.asarray(norm['scale'])
    report = {'checkpoint': checkpoint['sha256'], 'validation': {}, 'rollouts': []}
    weights = {}
    for task, info in manifest['tasks'].items():
        instruction = encode(info['language_instruction'], vocab).to(device)
        with h5py.File(ROOT / info['file'], 'r') as file:
            for split in ('train', 'validation'):
                features, residuals = [], []
                for name in manifest['splits'][split][task]:
                    ep = file['data'][name]
                    images = {c: torch.from_numpy(ep['obs'][c][:-1]).to(device)
                              .permute(0, 3, 1, 2).float().div(255) for c in CAMERAS}
                    targets = (ep['actions'][1:] - mean) / scale
                    base, tokens = corrected.predict(images, instruction.expand(len(targets), -1))
                    features.append(tokens.double())
                    residuals.append(torch.as_tensor(targets, device=device) - base.double())
                x, y = torch.cat(features), torch.cat(residuals)
                if split == 'train':
                    center = x.mean(0)
                    bias = y.mean(0)
                    x = x - center
                    gram = x.T @ x
                    # Ridge penalty scaled to the average training feature energy.
                    penalty = gram.diagonal().mean() * 0.1
                    gram.diagonal().add_(penalty)
                    coef = torch.linalg.solve(gram, x.T @ (y - bias))
                    weights[task] = (center.float(), coef.float(), bias.float())
                else:
                    errors = y - ((x - center) @ coef + bias)
                    report['validation'][task] = {'original_mse': y.square().mean().item(),
                                                  'corrected_mse': errors.square().mean().item()}
                    print(task, report['validation'][task], flush=True)
    torch.save({'base_checkpoint': checkpoint, 'by_instruction_task': weights}, output / 'head.pt')
    for task, info in manifest['tasks'].items():
        _, _, definition, bddl, get_path, Env, suite = task_definition(ROOT, task)
        center, coef, bias = weights[task]
        with h5py.File(ROOT / info['file'], 'r') as file:
            for name in manifest['splits']['validation'][task][:2]:
                ep = file['data'][name]
                xml = local_xml(ep.attrs['model_file'], get_path, suite)
                for use_correction in (False, True):
                    env = Env(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
                    try:
                        env.seed(0)
                        env.reset()
                        env.reset_from_xml_string(xml)
                        env.sim.reset()
                        obs = env.set_init_state(ep['states'][1])
                        assert not env.check_success()
                        low, high = env.env.action_spec
                        for step in range(1, 301):
                            images, text = model_inputs(obs, definition.language, vocab, device)
                            base, tokens = corrected.predict(images, text)
                            if use_correction:
                                base = base + (tokens - center) @ coef + bias
                            action = base[0].cpu().numpy() * scale + mean
                            obs, _, done, _ = env.step(np.clip(action, low, high).tolist())
                            success = bool(env.check_success())
                            if success or done:
                                break
                    finally:
                        env.close()
                    row = dict(task=task, episode=name, corrected=use_correction, success=success, steps=step)
                    report['rollouts'].append(row)
                    (output / 'validation.json').write_text(json.dumps(report, indent=2))
                    print(row, flush=True)
    corrected.hook.remove()


if __name__ == '__main__':
    main()

