"""Task strings and frozen text towers."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

LANG_DIM = 384


def encode_task(text: str, length: int = 64) -> list[int]:
    """Map a full instruction to byte ids.

    Both bowl instructions share the prefix "put the bowl on the ".
    The length has to cover the goal word.
    """
    values = [ord(character) % 256 for character in text.lower()]
    if len(values) > length:
        raise ValueError(f"task string is longer than {length} bytes: {text}")
    return values + [0] * (length - len(values))


class FrozenTextStandIn(nn.Module):
    """Frozen 384-d byte embedding with the width of all-MiniLM-L6-v2."""

    def __init__(self, vocab_size: int = 256, width: int = LANG_DIM) -> None:
        super().__init__()
        self.width = width
        self.embed = nn.Embedding(vocab_size, width)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(token_ids).mean(dim=1)


def use_goal_suffix(policy) -> None:
    """Pool bytes from index 20 so stove and plate stay distinct."""
    embed = policy.text.embed

    def forward(token_ids: torch.Tensor) -> torch.Tensor:
        return embed(token_ids[:, 20:]).mean(dim=1)

    policy.text.forward = forward


class FrozenMiniLM(nn.Module):
    """Mean-pool and L2-normalize all-MiniLM-L6-v2. Weights stay frozen."""

    def __init__(self, path: Path, device: torch.device) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True)
        self.model.to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.device = device

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        texts = []
        for row in token_ids.detach().cpu().tolist():
            chars = []
            for value in row:
                if value == 0:
                    break
                chars.append(chr(int(value)))
            texts.append("".join(chars))
        batch = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.no_grad():
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return F.normalize(pooled, p=2, dim=-1)


def attach_minilm(policy, encoder: FrozenMiniLM) -> None:
    for parameter in policy.text.parameters():
        parameter.requires_grad = False
    policy.text.forward = encoder.forward
