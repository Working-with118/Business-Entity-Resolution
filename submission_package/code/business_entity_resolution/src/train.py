"""
Trains the pairwise matching classifier and selects a decision threshold
that maximizes macro-averaged F_0.5 on a held-out validation split of the
training Source-1 entities.

Model choice: gradient-boosted trees (LightGBM if available, else scikit-
learn's HistGradientBoostingClassifier as a zero-dependency fallback). Both
are open-source (MIT/BSD), tiny (nowhere near the 8B-parameter cap), and
train in seconds to minutes on this kind of feature table -- appropriate
given the actual signal here is a handful of string-similarity scores, not
raw text a large model would be needed to understand.
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd

from . import blocking as blk
from . import features as feat
from . import evaluate_f05 as ev

RANDOM_SEED = 42


def _load_source(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _split_train_val(s1_df, val_frac=0.2, seed=RANDOM_SEED):
    ids = list(s1_df["entity_id"])
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids


def build_training_table(s1_df, s2_df, s3_df, gt_df, entity_ids, use_embeddings=True):
    """Build a labeled feature table for the given subset of S1 entity_ids.
    Positives = ground-truth matches. Negatives = other blocking candidates
    for the same S1 entity that are NOT in ground truth (hard negatives)."""
    sub_s1 = s1_df[s1_df["entity_id"].isin(entity_ids)].reset_index(drop=True)
    candidates = blk.build_candidates(sub_s1, s2_df, s3_df)

    other_df = pd.concat([s2_df, s3_df], ignore_index=True)
    packed_s1 = feat.row_pack(sub_s1)
    packed_other = feat.row_pack(other_df)

    gt_map = {
        row["source1_entity_id"]: ev.parse_id_list(row["matched_entity_ids"])
        for _, row in gt_df.iterrows()
    }

    embed_a = embed_b = None
    id_to_embed_idx = {}
    if use_embeddings:
        all_ids = list(sub_s1["entity_id"]) + list(other_df["entity_id"])
        all_texts = [packed_s1.get(i, packed_other.get(i, {})).get("name_norm", "") for i in all_ids]
        vecs = feat.embed_texts(all_texts)
        if vecs is not None:
            id_to_embed_idx = {eid: idx for idx, eid in enumerate(all_ids)}
            embed_a = embed_b = vecs

    X, y, pair_ids = [], [], []
    for s1_id in sub_s1["entity_id"]:
        true_ids = gt_map.get(s1_id, set())
        cand_ids = set(candidates.get(s1_id, [])) | true_ids  # ensure positives are trainable even if blocking missed them
        rec_a = packed_s1[s1_id]
        for cand_id in cand_ids:
            rec_b = packed_other.get(cand_id)
            if rec_b is None:
                continue
            cos = 0.0
            if id_to_embed_idx and s1_id in id_to_embed_idx and cand_id in id_to_embed_idx:
                va = embed_a[id_to_embed_idx[s1_id]]
                vb = embed_b[id_to_embed_idx[cand_id]]
                cos = float(np.dot(va, vb))
            X.append(feat.pair_features(rec_a, rec_b, embed_cosine=cos))
            y.append(1 if cand_id in true_ids else 0)
            pair_ids.append((s1_id, cand_id))

    return np.array(X, dtype=float), np.array(y, dtype=int), pair_ids, candidates


def get_classifier():
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=RANDOM_SEED
        )


def tune_threshold(model, X_val, pair_ids_val, gt_val):
    probs = model.predict_proba(X_val)[:, 1]
    by_s1 = {}
    for (s1_id, cand_id), p in zip(pair_ids_val, probs):
        by_s1.setdefault(s1_id, []).append((cand_id, p))

    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        preds = {}
        for s1_id, scored in by_s1.items():
            preds[s1_id] = {cid for cid, p in scored if p >= t}
        for s1_id in gt_val:
            preds.setdefault(s1_id, set())
        score = ev.macro_f05(preds, gt_val)
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t, best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--model-out", default="model")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--no-embeddings", action="store_true")
    args = ap.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    s1 = _load_source(os.path.join(train_dir, "train_source1.tsv"))
    s2 = _load_source(os.path.join(train_dir, "train_source2.tsv"))
    s3 = _load_source(os.path.join(train_dir, "train_source3.tsv"))
    gt = _load_source(os.path.join(train_dir, "train_ground_truth.tsv"))

    train_ids, val_ids = _split_train_val(s1, args.val_frac)
    use_emb = not args.no_embeddings

    print(f"Building training table ({len(train_ids)} S1 entities)...")
    X_train, y_train, _, _ = build_training_table(
        s1, s2, s3, gt, train_ids, use_embeddings=use_emb
    )
    print(f"  {X_train.shape[0]} pairs, {y_train.sum()} positives")

    print(f"Building validation table ({len(val_ids)} S1 entities)...")
    X_val, y_val, pair_ids_val, _ = build_training_table(
        s1, s2, s3, gt, val_ids, use_embeddings=use_emb
    )

    gt_map = {
        row["source1_entity_id"]: ev.parse_id_list(row["matched_entity_ids"])
        for _, row in gt.iterrows()
        if row["source1_entity_id"] in val_ids
    }

    print("Training classifier...")
    model = get_classifier()
    model.fit(X_train, y_train)

    print("Tuning decision threshold for macro F_0.5 on validation split...")
    threshold, val_score = tune_threshold(model, X_val, pair_ids_val, gt_map)
    print(f"  best threshold={threshold:.2f}  val macro F0.5={val_score:.4f}")

    os.makedirs(args.model_out, exist_ok=True)
    import joblib

    joblib.dump(model, os.path.join(args.model_out, "classifier.joblib"))
    with open(os.path.join(args.model_out, "config.json"), "w") as f:
        json.dump(
            {"threshold": threshold, "use_embeddings": use_emb, "val_macro_f05": val_score},
            f,
            indent=2,
        )
    print(f"Saved model + config to {args.model_out}/")


if __name__ == "__main__":
    main()
