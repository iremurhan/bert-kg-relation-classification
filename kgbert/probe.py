"""Linear relation classifier on fixed features, with TACRED-style scoring."""

from __future__ import annotations

import numpy as np


def train_linear(X: np.ndarray, y: np.ndarray, num_classes: int, epochs: int = 20, lr: float = 1e-3,
                 batch_size: int = 256, weight_decay: float = 0.0, seed: int = 0):
    """Train a single linear layer with cross-entropy (mini-batch Adam). Features are standardized first."""
    import torch

    torch.manual_seed(seed)
    mean, std = X.mean(0), X.std(0) + 1e-6
    Xt = torch.tensor((X - mean) / std, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    layer = torch.nn.Linear(X.shape[1], num_classes)
    opt = torch.optim.Adam(layer.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(yt), generator=gen).split(batch_size):
            opt.zero_grad()
            loss_fn(layer(Xt[idx]), yt[idx]).backward()
            opt.step()

    def predict(Xnew: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = layer(torch.tensor((Xnew - mean) / std, dtype=torch.float32))
        return logits.argmax(1).numpy()

    return predict


def tacred_scores(gold: list[str], pred: list[str], negative: str = "no_relation") -> dict[str, float]:
    """Micro precision/recall/F1 over positive relations, as in the official TACRED scorer.

    A prediction counts as correct only if it is a positive relation equal to the gold one;
    `negative` predictions and gold labels are excluded from the denominators.
    """
    correct = sum(g == p != negative for g, p in zip(gold, pred))
    guessed = sum(p != negative for p in pred)
    actual = sum(g != negative for g in gold)
    precision = correct / guessed if guessed else 0.0
    recall = correct / actual if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = sum(g == p for g, p in zip(gold, pred)) / len(gold) if gold else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}
