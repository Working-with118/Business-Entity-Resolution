# Methodology — Business Entity Resolution

> Note: fill in the italicized `[ ]` placeholders with your own numbers once
> you've run `python main.py` on the real dataset — this write-up documents
> the approach precisely, but the actual F_0.5 / recall figures depend on
> the data, which isn't available in this environment.

## 1. Methodology used

We treat entity resolution as a two-stage pipeline: cheap high-recall
**blocking** to shrink the S2/S3 search space per Source-1 entity, followed
by a supervised **pairwise classifier** that scores each (S1, candidate)
pair and a **tuned threshold** that converts scores into final match/no-match
decisions. This split is standard for ER at scale: it keeps the O(n·m)
comparison cost of the full name/address similarity machinery restricted to
a small candidate pool per entity, while still letting the final decision be
learned rather than hand-ruled, since F_0.5 punishes both false merges and
missed matches asymmetrically.

Text normalization is applied before every downstream step (`src/preprocessing.py`):
lowercasing, accent/transliteration folding, legal-suffix canonicalization
for names (`Pvt Ltd` / `Private Limited` / `Ltd.` → one token), address
abbreviation expansion (`Rd` → `Road`, etc.), and landmark-clause stripping
(`Near SBI ATM`) so it doesn't pollute address token overlap.

## 2. Candidate generation / blocking strategy

Two complementary blocking passes are unioned per Source-1 entity:

1. **TF-IDF character n-gram (2–4 gram) cosine nearest neighbours** on the
   normalized business name, computed via sparse matrix multiplication in
   batches. This is robust to typos, partial abbreviation expansion, word
   transpositions, and transliteration spelling variance, because it
   doesn't require token-level agreement.
2. **Sorted-neighbourhood blocking key**: each name's tokens are reduced to
   their normalized "core" (legal suffixes stripped), then to a sorted
   concatenation of 4-character token prefixes. Records sharing a key are
   pulled in as candidates. This catches near-misses that fall just outside
   the TF-IDF top-K, especially for short names where n-gram similarity is
   noisy.

Country is deliberately **not** used as a hard filter — the test set
introduces France, unseen in training, and country labels themselves can be
noisy. Instead, once a candidate pool exceeds a cap (default 25), it is
re-ranked by country agreement + core-token overlap and truncated, so
recall isn't silently destroyed for entities with an unusually large raw
candidate pool.

- *[ ] Recall ceiling on held-out validation: report `len(union of true matches found in candidates) / len(all true matches)` after running blocking on the validation split.*
- *[ ] Reduction ratio: report average candidate-pool size vs. raw S2+S3 size.*

## 3. Model architecture and feature engineering

**Features** (`src/features.py`), computed per (S1, candidate) pair from the
normalized text only:

| Feature | What it captures |
|---|---|
| `name_jaccard` / `name_core_jaccard` | token-set overlap on full vs. suffix-stripped name |
| `name_lev_ratio` | character-level edit-distance similarity (typos) |
| `name_len_diff` | raw string length delta |
| `address_jaccard` / `address_lev_ratio` | same signals on normalized address |
| `country_match` | exact match after alias normalization |
| `name_first_token_match` | cheap signal for shared first word (often a brand root) |
| `token_count_diff` | word-count delta |
| `name_embed_cosine` | cosine similarity of `all-MiniLM-L6-v2` sentence embeddings (MIT license, ~22M parameters — well under the challenge's 8B-parameter cap). Optional: the pipeline still runs, with this feature fixed at 0, if the model isn't available offline. Captures paraphrase-level name similarity that lexical overlap misses (e.g. abbreviation-heavy trade names vs. full legal names). |

**Model**: gradient-boosted decision trees — LightGBM by default
(`class_weight="balanced"` to offset the natural positive/negative
imbalance from candidate generation), falling back automatically to
scikit-learn's `HistGradientBoostingClassifier` if LightGBM isn't installed.
Both are open-source (MIT/BSD) and orders of magnitude smaller than the
8B-parameter cap; a small feature table of string-similarity scores doesn't
benefit from a larger model, and GBTs handle the mix of bounded [0,1]
similarity scores and count-based features well without additional scaling.

Training pairs: positives are every (S1, matched-id) pair from
`train_ground_truth.tsv`; negatives are every *other* blocking candidate for
that same S1 entity (hard negatives — near-name-matches that are still the
wrong business), which is the discriminative signal the classifier actually
needs to learn given F_0.5's precision weighting.

**Threshold selection**: rather than optimizing a generic classification
metric, the decision threshold is chosen by directly maximizing the
challenge's own macro-averaged F_0.5 (`src/evaluate_f05.py`, singleton rule
included) on a held-out 20% split of Source-1 training entities, scanned
over `[0.05, 0.95]` in steps of 0.05.

- *[ ] Chosen threshold: see `model/config.json` after training.*
- *[ ] Validation macro F_0.5 at that threshold: see training log / `model/config.json`.*

## 4. Other relevant information

- **Fair play**: no external APIs, geocoders, or business-registry lookups
  are used anywhere in the pipeline — every normalization rule is a static,
  hand-curated dictionary derived from the noise patterns described in the
  problem statement.
- **Output contract**: `src/infer.py` enforces the submission rules
  directly (one row per test S1 entity even with zero candidates/matches,
  empty string rather than `nan`, de-duplicated ID lists, S2-/S3- ids
  filtered to only those present in the test set) so a bad format doesn't
  cost a submission — still run `utils/validate_submission.py` before
  uploading.
- **Reproducibility**: `python main.py --data-dir dataset` regenerates both
  `output/matching_results.tsv` and `output/candidate_pairs.tsv` end to end
  from the raw TSVs; `requirements.txt` pins all dependencies.
