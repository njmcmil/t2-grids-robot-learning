from pathlib import Path
import hashlib, json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

CAMERAS = ("agentview_rgb", "eye_in_hand_rgb")


# Read the saved episode split and check which data supplied the action averages.
def preparation(root):
    directory = Path(root) / "configs/data_preparation"
    raw = (directory / "split_manifest.json").read_bytes()
    manifest = json.loads(raw)
    norm = json.loads((directory / "action_normalization.json").read_text())
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == norm["manifest_sha256"]
    assert norm["fit_split"] == "train"
    assert (
        manifest["preprocessing"]["action_target"]
        == "actions[t + 1] for observation[t]"
    )
    return manifest, norm, digest


def vocabulary(manifest):
    # Only instructions associated with training tasks; no pretrained encoder.
    words = sorted(
        {
            word
            for task in manifest["splits"]["train"]
            for word in manifest["tasks"][task]["language_instruction"].lower().split()
        }
    )
    return {"<pad>": 0, "<unk>": 1, **{word: i + 2 for i, word in enumerate(words)}}


def encode(text, vocab, length=16):
    words = text.lower().split()
    if len(words) > length:
        raise ValueError(
            "Instruction exceeds the declared length; do not silently truncate."
        )
    return torch.tensor(
        [vocab.get(word, 1) for word in words] + [0] * (length - len(words))
    )


def image_tensor(image):
    assert image.shape == (128, 128, 3)
    return torch.from_numpy(
        np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0
    )


# Each item pairs two camera views with the next recorded robot command.
class Demonstrations(Dataset):
    def __init__(self, root, manifest, norm, vocab, split):
        self.root, self.manifest, self.vocab = Path(root), manifest, vocab
        self.mean = np.asarray(norm["mean"], dtype=np.float32)
        self.scale = np.asarray(norm["scale"], dtype=np.float32)
        self.index = []
        for task, names in manifest["splits"][split].items():
            with h5py.File(self.root / manifest["tasks"][task]["file"], "r") as f:
                for name in names:
                    self.index.extend(
                        (task, name, t)
                        for t in range(len(f["data"][name]["actions"]) - 1)
                    )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        task, name, t = self.index[index]
        with h5py.File(self.root / self.manifest["tasks"][task]["file"], "r") as f:
            ep = f["data"][name]
            images = {c: image_tensor(ep["obs"][c][t]) for c in CAMERAS}
            action = (ep["actions"][t + 1].astype(np.float32) - self.mean) / self.scale
        return {
            "images": images,
            "action": torch.from_numpy(action),
            "task": task,
            "text": encode(
                self.manifest["tasks"][task]["language_instruction"], self.vocab
            ),
        }
