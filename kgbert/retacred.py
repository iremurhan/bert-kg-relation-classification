"""Loading TACRED / Re-TACRED and turning examples into (sentence, subject, object, label)."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

SPLITS = ("train", "dev", "test")
NO_RELATION = "no_relation"


def load_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_patch(split: list[dict], id2label: dict[str, str]) -> list[dict]:
    """Relabel a TACRED split with a Re-TACRED patch.

    As in the Re-TACRED release, examples whose id is not in the patch are dropped.
    """
    patched = []
    for example in split:
        if example["id"] in id2label:
            new = deepcopy(example)
            new["relation"] = id2label[example["id"]]
            patched.append(new)
    return patched


def build_retacred(tacred_json_dir: str | Path, patch_dir: str | Path, out_dir: str | Path) -> dict[str, int]:
    """Write Re-TACRED train/dev/test JSON files and return their sizes."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name in SPLITS:
        split = load_json(Path(tacred_json_dir) / f"{name}.json")
        patch = load_json(Path(patch_dir) / f"{name}_id2label.json")
        patched = apply_patch(split, patch)
        with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(patched, f)
        sizes[name] = len(patched)
    return sizes


def span_text(example: dict, role: str) -> str:
    """Surface text of the subject (role='subj') or object (role='obj') span; end index is inclusive."""
    start, end = example[f"{role}_start"], example[f"{role}_end"]
    return " ".join(example["token"][start : end + 1])


def sentence(example: dict) -> str:
    return " ".join(example["token"])


class LabelEncoder:
    """Maps relation names to contiguous integer ids, fitted on the training split."""

    def __init__(self, labels: list[str]):
        self.classes = sorted(set(labels))
        self.index = {c: i for i, c in enumerate(self.classes)}

    def encode(self, labels: list[str]) -> list[int]:
        return [self.index[label] for label in labels]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.classes[i] for i in ids]
