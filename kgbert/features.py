"""Sentence features: frozen BERT [CLS] vectors, optionally concatenated with subject/object KG vectors."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .kg import KGEmbeddings
from .retacred import sentence, span_text

MODES = ("bert", "kg", "bert+kg")


class BertEncoder:
    """Frozen BERT encoder returning the final-layer [CLS] vector for each sentence."""

    def __init__(self, model_name: str = "bert-base-uncased", device: str | None = None, max_length: int = 128):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.torch = torch
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.dim = self.model.config.hidden_size

    def __call__(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = self.tokenizer(
                    texts[i : i + batch_size], padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self.device)
                out.append(self.model(**batch).last_hidden_state[:, 0, :].float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.dim), dtype=np.float32)


def build_features(
    examples: list[dict],
    mode: str,
    encoder: Callable[[list[str]], np.ndarray] | None = None,
    kg: KGEmbeddings | None = None,
) -> np.ndarray:
    """Feature matrix for a list of (Re-)TACRED examples.

    bert:    [CLS]                                  (768)
    kg:      [subject KG ; object KG]               (2 x d)
    bert+kg: [CLS ; subject KG ; object KG]         (768 + 2 x d)
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    parts = []
    if mode in ("bert", "bert+kg"):
        if encoder is None:
            raise ValueError("a BERT encoder is required for this mode")
        parts.append(np.asarray(encoder([sentence(e) for e in examples]), dtype=np.float32))
    if mode in ("kg", "bert+kg"):
        if kg is None:
            raise ValueError("KG embeddings are required for this mode")
        subj = np.stack([kg.lookup(span_text(e, "subj")) for e in examples])
        obj = np.stack([kg.lookup(span_text(e, "obj")) for e in examples])
        parts.extend([subj, obj])
    return np.concatenate(parts, axis=1)
