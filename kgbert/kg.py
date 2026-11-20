"""Knowledge-graph entity embeddings (PyKEEN TransE) and name-based entity linking."""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def dbpedia_key(name: str) -> str:
    """DBpedia-style resource name: 'Barack Obama' -> 'Barack_Obama'."""
    return "_".join(name.split())


def _loose(name: str) -> str:
    """Case- and punctuation-insensitive key: 'Barack Obama' -> 'barack_obama'."""
    return re.sub(r"[^\w]", "", dbpedia_key(name).lower())


def read_entity_to_id(path: str | Path) -> dict[str, int]:
    """Read PyKEEN's entity_to_id.tsv(.gz) (columns: id, label)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    mapping = {}
    with opener(path, "rt", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            idx, label = line.rstrip("\n").split("\t", 1)
            mapping[label] = int(idx)
    return mapping


@dataclass
class EntityLinker:
    """Links a mention to a KG entity label: exact DBpedia key, then a loose key, then (optionally) fuzzy."""

    entity_to_id: dict[str, int]
    fuzzy_threshold: float | None = None  # rapidfuzz score in [0, 100]; None disables fuzzy matching
    stats: dict[str, int] = field(default_factory=lambda: {"exact": 0, "loose": 0, "fuzzy": 0, "missing": 0})

    def __post_init__(self):
        self._loose = {}
        for label in self.entity_to_id:
            self._loose.setdefault(_loose(label), label)

    def link(self, mention: str) -> str | None:
        key = dbpedia_key(mention)
        if key in self.entity_to_id:
            self.stats["exact"] += 1
            return key
        loose = _loose(mention)
        if loose and loose in self._loose:
            self.stats["loose"] += 1
            return self._loose[loose]
        if self.fuzzy_threshold is not None and loose:
            from rapidfuzz import process

            match = process.extractOne(loose, self._loose.keys(), score_cutoff=self.fuzzy_threshold)
            if match is not None:
                self.stats["fuzzy"] += 1
                return self._loose[match[0]]
        self.stats["missing"] += 1
        return None

    def coverage(self) -> float:
        total = sum(self.stats.values())
        return 0.0 if total == 0 else 1.0 - self.stats["missing"] / total


class KGEmbeddings:
    """Entity embedding table plus linker. Unlinked mentions get a zero vector."""

    def __init__(self, vectors: np.ndarray, entity_to_id: dict[str, int], fuzzy_threshold: float | None = None):
        self.vectors = vectors
        self.dim = vectors.shape[1]
        self.linker = EntityLinker(entity_to_id, fuzzy_threshold)

    @classmethod
    def from_pykeen_dir(cls, directory: str | Path, fuzzy_threshold: float | None = None) -> "KGEmbeddings":
        """Load a directory written by pykeen's PipelineResult.save_to_directory."""
        import torch

        directory = Path(directory)
        model = torch.load(directory / "trained_model.pkl", map_location="cpu", weights_only=False)
        vectors = model.entity_representations[0](indices=None).detach().cpu().numpy()
        mapping_path = directory / "training_triples" / "entity_to_id.tsv.gz"
        if not mapping_path.exists():
            mapping_path = mapping_path.with_suffix("")
        return cls(vectors, read_entity_to_id(mapping_path), fuzzy_threshold)

    def lookup(self, mention: str) -> np.ndarray:
        label = self.linker.link(mention)
        if label is None:
            return np.zeros(self.dim, dtype=np.float32)
        return self.vectors[self.linker.entity_to_id[label]].astype(np.float32)
