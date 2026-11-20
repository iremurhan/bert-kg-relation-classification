# BERT + Knowledge-Graph Embeddings for Relation Classification

Can structured knowledge about entities help a text encoder classify the relation between them? This project concatenates frozen BERT sentence vectors with TransE entity embeddings for the subject and object, trains a relation classifier on Re-TACRED, and measures how much of the knowledge graph actually reaches the classifier.

Written in Spring 2024 as the application project of CMPE 58T (Advanced Natural Language Processing) at Boğaziçi University, and refactored in 2026. The submitted report is in [`docs/final-report.md`](docs/final-report.md).

**Main finding:** name-based entity linking connects only about 1% of Re-TACRED entity mentions to DBpedia50, so the knowledge-graph features are almost always empty and BERT+KG performs the same as BERT alone (F1 0.265 vs 0.260). Entity linking, not the fusion method, is the bottleneck.

## Method

1. **Data:** TACRED relabelled with the [Re-TACRED](https://github.com/gstoica27/Re-TACRED) patch: 58,465 / 19,584 / 13,418 train / dev / test examples, 40 relations including `no_relation`. Subject and object spans are given in the data.
2. **KG embeddings:** TransE (embedding size 50, margin ranking loss, sLCWA training) trained with [PyKEEN](https://github.com/pykeen/pykeen) on DBpedia50, FB15k-237 and WN18RR.
3. **Entity linking:** a mention is mapped to a KG entity by its DBpedia-style name (`Barack Obama` → `Barack_Obama`), then by a case- and punctuation-insensitive key, and optionally by fuzzy string matching. Unlinked mentions get a zero vector.
4. **Features:** the `[CLS]` vector of frozen `bert-base-uncased` (768), optionally concatenated with the subject and object KG vectors (2 × 50).
5. **Classifier:** a linear layer trained with cross-entropy on standardized features (Adam, 20 epochs). Scores follow the official TACRED scorer: micro precision, recall and F1 over positive relations, with `no_relation` excluded.

## Data

Not included. TACRED is distributed by the LDC ([LDC2018T24](https://catalog.ldc.upenn.edu/LDC2018T24)) under a licence that does not allow redistribution. The Re-TACRED patch files come from the Re-TACRED repository. DBpedia50, FB15k-237 and WN18RR are downloaded by PyKEEN.

## Usage

```bash
pip install -r requirements.txt

# Re-TACRED from the TACRED JSON files and the Re-TACRED patch
python -m kgbert patch --tacred tacred/data/json --patch Re-TACRED/patch --out data/re-tacred

# TransE on DBpedia50 (PyKEEN downloads the graph)
python -m kgbert train-kg --dataset DBpedia50 --dim 50 --epochs 100 --out models/transe-dbpedia50

# share of subject/object mentions that link to the graph
python -m kgbert coverage --data data/re-tacred --kg models/transe-dbpedia50 --split test

# linear classifier on BERT, KG or BERT+KG features
python -m kgbert run --data data/re-tacred --kg models/transe-dbpedia50 --mode bert+kg \
    --eval-split test --cache cache/

pytest -q
```

`--fuzzy 90` turns on fuzzy matching, `--limit N` uses only the first *N* examples of each split, and `--cache` stores BERT features so that the three feature modes encode each split once.

## Results

### Entity linking

Subject and object mentions linked to DBpedia50 (24,624 entities):

| Split | Mentions | Exact | Normalized | Linked |
|---|---|---|---|---|
| Train | 116,930 | 1,328 | 203 | 1.3% |
| Dev | 39,168 | 396 | 87 | 1.2% |
| Test | 26,836 | 100 | 53 | 0.6% |

Fuzzy matching (rapidfuzz score ≥ 90) raises test coverage to 52.7%, but the additional matches are spurious: pronouns and partial names are matched to unrelated entities (*he* → `187_He_Wrote`, *his* → `A_Book_Like_This`, *Graham* → `Chromodoris_grahami`), so fuzzy matching is not used in the classification runs. The other two graphs cannot be linked by name at all: FB15k-237 entities are Freebase machine IDs (`/m/010016`) and WN18RR entities are WordNet synset offsets (`1740`).

### Relation classification

Re-TACRED test set, linear classifier on frozen features, mean ± standard deviation over three seeds (DBpedia50 TransE, 1000 epochs):

| Features | Precision | Recall | F1 |
|---|---|---|---|
| BERT `[CLS]` | 0.430 ± 0.007 | 0.186 ± 0.005 | 0.260 ± 0.005 |
| Subject + object TransE | 0.459 ± 0.024 | 0.004 ± 0.000 | 0.008 ± 0.001 |
| BERT + TransE | 0.434 ± 0.005 | 0.190 ± 0.005 | 0.265 ± 0.004 |

With KG vectors present for about one mention in a hundred, the KG-only classifier predicts `no_relation` almost everywhere, and adding the KG vectors to BERT changes F1 by less than the seed-to-seed variation. The absolute scores are those of a frozen sentence vector without entity markers; fine-tuned, entity-aware encoders reach much higher F1 on Re-TACRED, and the comparison here is between feature sets under the same classifier.

### Knowledge-graph embeddings

TransE models trained in 2024, PyKEEN rank-based evaluation on each graph's test split (realistic ranks, head and tail):

| Graph | Epochs | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|---|
| DBpedia50 | 1000 | 0.219 | 0.114 | 0.295 | 0.374 |
| DBpedia50 | 80 | 0.117 | 0.051 | 0.151 | 0.227 |
| FB15k-237 | 80 | 0.182 | 0.109 | 0.196 | 0.327 |

The raw PyKEEN outputs, including a short WN18RR run, are in [`results/transe/`](results/transe/).

## What this suggests

- **Linking first.** Before fusing knowledge-graph and text representations, the mentions have to be linked to a graph that covers them: an entity linker against Wikidata or full DBpedia, rather than exact names against a 25K-entity subset.
- **Fusion second.** Only once coverage is high does the choice between concatenation, gating or attention over KG neighbours (as in Tokuhisa et al., 2022) become measurable.

## What changed from the 2024 version

- **Structure:** the notebook is split into a small package (`kgbert/`) with a command-line interface; Colab and Google Drive paths are gone.
- **Pipeline:** one entity linker (exact, normalized, optional fuzzy) with coverage counts, batched BERT encoding with an on-disk cache, label encoding, standardized features and mini-batch training over the full training split.
- **Evaluation:** the TACRED micro-F1 over positive relations, three seeds, and the linking-coverage measurement are new; the classification results above were produced with this code.
- **Tests:** unit tests for patching, linking, features, scoring and the classifier.

## References

- Y. Zhang, V. Zhong, D. Chen, G. Angeli and C. D. Manning. *Position-aware Attention and Supervised Data Improve Slot Filling.* EMNLP, 2017.
- G. Stoica, E. A. Platanios and B. Póczos. *Re-TACRED: Addressing Shortcomings of the TACRED Dataset.* AAAI, 2021.
- A. Bordes, N. Usunier, A. Garcia-Durán, J. Weston and O. Yakhnenko. *Translating Embeddings for Modeling Multi-relational Data.* NeurIPS, 2013.
- R. Tokuhisa, K. Kawano, A. Nakamura and S. Koide. *Enhancing Contextual Word Representations Using Embedding of Neighboring Entities in Knowledge Graphs.* COLING, 2022.
- M. Ali et al. *PyKEEN 1.0: A Python Library for Training and Evaluating Knowledge Graph Embeddings.* JMLR, 2021.
