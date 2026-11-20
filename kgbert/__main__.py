"""Command-line entry point: python -m kgbert {patch,train-kg,coverage,run} ..."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .retacred import SPLITS, LabelEncoder, build_retacred, load_json, span_text


def cmd_patch(args):
    sizes = build_retacred(args.tacred, args.patch, args.out)
    print(json.dumps(sizes))


def cmd_train_kg(args):
    from pykeen.pipeline import pipeline

    result = pipeline(
        model="TransE", dataset=args.dataset, training_loop="slcwa", loss="MarginRankingLoss",
        model_kwargs={"embedding_dim": args.dim}, epochs=args.epochs, random_seed=args.seed,
        stopper="early", stopper_kwargs={"metric": "hits_at_k", "larger_is_better": True, "frequency": 5, "patience": 2},
    )
    result.save_to_directory(args.out)
    print(f"saved to {args.out}")


def _load_kg(args):
    from .kg import KGEmbeddings

    return KGEmbeddings.from_pykeen_dir(args.kg, fuzzy_threshold=args.fuzzy)


def cmd_coverage(args):
    kg = _load_kg(args)
    data = load_json(Path(args.data) / f"{args.split}.json")[: args.limit]
    for example in data:
        kg.linker.link(span_text(example, "subj"))
        kg.linker.link(span_text(example, "obj"))
    print(json.dumps({"mentions": sum(kg.linker.stats.values()), **kg.linker.stats,
                      "coverage": round(kg.linker.coverage(), 4)}))


def _cached(encoder, cache_dir: Path, model_name: str):
    """Wrap an encoder so that features for an identical list of sentences are stored on disk."""
    import hashlib

    import numpy as np

    cache_dir.mkdir(parents=True, exist_ok=True)

    def encode(texts):
        key = hashlib.sha1((model_name + "\n" + "\n".join(texts)).encode("utf-8")).hexdigest()[:16]
        path = cache_dir / f"bert-{key}.npy"
        if path.exists():
            return np.load(path)
        feats = encoder(texts)
        np.save(path, feats)
        return feats

    return encode


def cmd_run(args):
    import numpy as np

    from .features import BertEncoder, build_features
    from .probe import tacred_scores, train_linear

    data = {s: load_json(Path(args.data) / f"{s}.json")[: args.limit] for s in ("train", args.eval_split)}
    kg = _load_kg(args) if args.mode != "bert" else None
    labels = LabelEncoder([e["relation"] for e in data["train"]])
    eval_data = [e for e in data[args.eval_split] if e["relation"] in labels.index]

    encoder = None
    if args.mode != "kg":
        bert = BertEncoder(args.bert)
        encoder = bert
        if args.cache:
            encoder = _cached(bert, Path(args.cache), args.bert)

    X_train = build_features(data["train"], args.mode, encoder, kg)
    X_eval = build_features(eval_data, args.mode, encoder, kg)
    y_train = np.array(labels.encode([e["relation"] for e in data["train"]]))
    predict = train_linear(X_train, y_train, len(labels.classes), epochs=args.epochs, seed=args.seed)
    pred = labels.decode(predict(X_eval).tolist())
    scores = tacred_scores([e["relation"] for e in eval_data], pred)
    if kg is not None:
        scores["kg_coverage"] = kg.linker.coverage()
    print(json.dumps({"mode": args.mode, "n_train": len(X_train), "n_eval": len(X_eval),
                      **{k: round(v, 4) for k, v in scores.items()}}))


def main(argv=None):
    p = argparse.ArgumentParser(prog="kgbert")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("patch", help="build Re-TACRED from TACRED JSON and the Re-TACRED patch files")
    s.add_argument("--tacred", required=True, help="folder with TACRED train/dev/test.json")
    s.add_argument("--patch", required=True, help="folder with {split}_id2label.json")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_patch)

    s = sub.add_parser("train-kg", help="train TransE with PyKEEN and save it")
    s.add_argument("--dataset", default="DBpedia50")
    s.add_argument("--dim", type=int, default=50)
    s.add_argument("--epochs", type=int, default=100)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_train_kg)

    for name, func, helptext in (("coverage", cmd_coverage, "share of entity mentions linked to the KG"),
                                 ("run", cmd_run, "train and evaluate a linear relation classifier")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--data", required=True, help="folder with Re-TACRED train/dev/test.json")
        s.add_argument("--kg", help="PyKEEN model directory")
        s.add_argument("--fuzzy", type=float, default=None, help="rapidfuzz score cutoff (0-100); off by default")
        s.add_argument("--limit", type=int, default=None, help="use only the first N examples of each split")
        s.set_defaults(func=func)
        if name == "coverage":
            s.add_argument("--split", choices=SPLITS, default="train")
        else:
            s.add_argument("--mode", choices=("bert", "kg", "bert+kg"), default="bert+kg")
            s.add_argument("--eval-split", choices=("dev", "test"), default="dev")
            s.add_argument("--bert", default="bert-base-uncased")
            s.add_argument("--epochs", type=int, default=20)
            s.add_argument("--seed", type=int, default=0)
            s.add_argument("--cache", help="folder for cached BERT features")

    args = p.parse_args(argv)
    if args.cmd in ("coverage",) and not args.kg:
        p.error("--kg is required")
    if args.cmd == "run" and args.mode != "bert" and not args.kg:
        p.error("--kg is required unless --mode bert")
    args.func(args)


if __name__ == "__main__":
    main()
