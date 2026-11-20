import numpy as np
import pytest

from kgbert.features import build_features
from kgbert.kg import EntityLinker, KGEmbeddings, dbpedia_key
from kgbert.probe import tacred_scores, train_linear
from kgbert.retacred import LabelEncoder, apply_patch, sentence, span_text

EX = {
    "id": "a1", "relation": "per:origin",
    "token": ["Barack", "Obama", "was", "born", "in", "Hawaii", "."],
    "subj_start": 0, "subj_end": 1, "obj_start": 5, "obj_end": 5,
}


def test_spans_and_sentence():
    assert span_text(EX, "subj") == "Barack Obama"
    assert span_text(EX, "obj") == "Hawaii"
    assert sentence(EX).startswith("Barack Obama was")


def test_apply_patch_relabels_and_drops_unpatched():
    other = dict(EX, id="b2")
    patched = apply_patch([EX, other], {"a1": "per:city_of_birth"})
    assert [e["id"] for e in patched] == ["a1"]
    assert patched[0]["relation"] == "per:city_of_birth"
    assert EX["relation"] == "per:origin"  # input left untouched


def test_label_encoder_roundtrip():
    enc = LabelEncoder(["b", "a", "b", "no_relation"])
    assert enc.classes == ["a", "b", "no_relation"]
    assert enc.decode(enc.encode(["no_relation", "a"])) == ["no_relation", "a"]


def test_linker_exact_loose_fuzzy_and_missing():
    mapping = {"Barack_Obama": 0, "Hawaii": 1, "New_York_City": 2}
    linker = EntityLinker(mapping)
    assert dbpedia_key("Barack  Obama") == "Barack_Obama"
    assert linker.link("Barack Obama") == "Barack_Obama"
    assert linker.link("new york city") == "New_York_City"
    assert linker.link("New York Immigration Coalition") is None
    assert linker.stats == {"exact": 1, "loose": 1, "fuzzy": 0, "missing": 1}
    pytest.importorskip("rapidfuzz")
    fuzzy = EntityLinker(mapping, fuzzy_threshold=85)
    assert fuzzy.link("Hawai") == "Hawaii"


def test_kg_lookup_zero_vector_for_unlinked():
    kg = KGEmbeddings(np.arange(6, dtype=np.float32).reshape(3, 2), {"A": 0, "B": 1, "C": 2})
    assert kg.lookup("B").tolist() == [2.0, 3.0]
    assert kg.lookup("Z").tolist() == [0.0, 0.0]
    assert kg.linker.coverage() == 0.5


def test_build_features_shapes():
    kg = KGEmbeddings(np.ones((2, 4), dtype=np.float32), {"Barack_Obama": 0, "Hawaii": 1})
    encoder = lambda texts: np.zeros((len(texts), 8), dtype=np.float32)
    assert build_features([EX, EX], "bert", encoder=encoder).shape == (2, 8)
    assert build_features([EX], "kg", kg=kg).shape == (1, 8)
    assert build_features([EX], "bert+kg", encoder=encoder, kg=kg).shape == (1, 16)
    with pytest.raises(ValueError):
        build_features([EX], "bert+kg", encoder=encoder)


def test_tacred_scores_ignore_no_relation():
    gold = ["no_relation", "a", "b", "no_relation"]
    pred = ["no_relation", "a", "no_relation", "b"]
    s = tacred_scores(gold, pred)
    assert s["precision"] == 0.5 and s["recall"] == 0.5 and s["f1"] == 0.5
    assert s["accuracy"] == 0.5


def test_linear_probe_learns_separable_data():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    X = rng.normal(size=(300, 5)) + np.eye(3, 5)[y] * 4
    predict = train_linear(X, y, 3, epochs=30, lr=0.05, batch_size=32)
    assert (predict(X) == y).mean() > 0.95
