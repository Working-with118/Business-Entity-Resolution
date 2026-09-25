# Business Entity Resolution — Amazon ML Challenge 2026

End-to-end pipeline that matches Source-2/Source-3 business records to
Source-1 reference entities across noisy, multi-source data (US, India, and
an unseen-at-train-time France in the test set).

## Pipeline overview

```
raw TSVs
   │
   ▼
1. Normalization (src/preprocessing.py)
   - legal-suffix / abbreviation canonicalization for names
   - address abbreviation expansion + landmark-clause stripping
   - country alias normalization (never hard-filtered on {US, India})
   │
   ▼
2. Blocking / candidate generation (src/blocking.py)
   - TF-IDF character n-gram cosine nearest-neighbours on normalized names
   - sorted-neighbourhood token-prefix blocking key (catches near-misses
     outside the TF-IDF top-K)
   - union of both, soft country-aware re-ranking if a S1 entity's
     candidate pool is oversized
   │
   ▼
3. Feature engineering (src/features.py)
   - name/address token Jaccard, Levenshtein ratio, length/token-count
     deltas, country match, first-token match
   - optional MiniLM sentence-embedding cosine similarity (MIT license,
     ~22M params — well under the 8B-parameter cap; pipeline still runs
     without it if the model can't be downloaded)
   │
   ▼
4. Classifier (src/train.py)
   - LightGBM (falls back to scikit-learn HistGradientBoostingClassifier
     if LightGBM isn't installed) trained on ground-truth positives vs.
     blocking-candidate negatives
   - decision threshold chosen by maximizing macro-averaged F_0.5
     (the actual leaderboard metric, including singleton credit) on a
     held-out validation split of Source-1 entities
   │
   ▼
5. Inference (src/infer.py)
   - re-run blocking + scoring on the test set
   - writes output/candidate_pairs.tsv and output/matching_results.tsv
     in the exact required schema (one row per test S1 entity, empty
     string for no-match/no-candidates, no duplicate ids, S2-/S3- only)
```

## Setup

```bash
pip install -r requirements.txt
```

`lightgbm` and `sentence-transformers` are used if available but are not
hard requirements — both stages degrade gracefully (scikit-learn GBM,
embedding feature set to 0) if either can't be installed in your
environment, so this also runs in a network-restricted setting once the
pip cache is warm.

## Data layout expected

Point `--data-dir` at a folder shaped like the challenge's `dataset/`:

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

## Run everything

```bash
python main.py --data-dir dataset --output-dir output --model-dir model
```

This trains the model, tunes the threshold, and writes
`output/matching_results.tsv` + `output/candidate_pairs.tsv`.

Then validate locally before uploading, using the challenge's own checker:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
(`utils/validate_submission.py` ships with the challenge's
`student_resource/` folder — drop it alongside this project to use it.)

## Re-running inference only

Once a model is trained, you can iterate on the decision threshold without
retraining:

```bash
python -m src.infer --data-dir dataset --model-dir model --output-dir output --threshold 0.65
```

## Evaluating your own held-out split

`src/evaluate_f05.py` implements the exact macro-averaged F_0.5 formula
(including the singleton 1.0/0.0 rule) used on the leaderboard, so you can
score any predictions file against `train_ground_truth.tsv`-shaped truth:

```python
from src.evaluate_f05 import load_gt_or_pred_tsv, macro_f05

pred = load_gt_or_pred_tsv("output/matching_results.tsv", "source1_entity_id", "matched_entity_ids")
truth = load_gt_or_pred_tsv("dataset/train/train_ground_truth.tsv", "source1_entity_id", "matched_entity_ids")
print(macro_f05(pred, truth))
```

## Design choices worth knowing about

- **No external lookups.** Every normalization rule (suffix maps, address
  abbreviations, country aliases) is a hand-curated dictionary, not a call
  to any geocoding/registry API — required by the challenge's fair-play
  rules.
- **Country is a feature, not a filter.** Test data introduces France,
  which never appears in training; nothing in blocking or feature
  engineering hard-codes `{US, India}`.
- **Precision-first threshold.** Because F_0.5 weights precision 2x over
  recall and rewards correctly-predicted singletons, the threshold is
  tuned directly against macro F_0.5 (not plain accuracy/F1 on pairs),
  which naturally pushes it higher than a balanced-F1 optimum would.
- **candidate_pairs.tsv is the actual inference-time candidate set** — the
  same one scored by the model — not an earlier, unfiltered blocking pass.

## Known limitations / next steps

- Blocking is name-driven; a company that changed its legal name entirely
  between sources (no name overlap at all) would only be caught via
  address similarity, which is weighted lower. Adding an address-driven
  blocking pass (e.g. TF-IDF on normalized address) would raise recall
  further if validation shows this is a meaningful miss category.
- The sentence-embedding feature requires `sentence-transformers` and a
  local model download; if that's unavailable in the grading environment,
  retrain with `--no-embeddings` to confirm the string-similarity-only
  feature set still meets your target score before submitting.
