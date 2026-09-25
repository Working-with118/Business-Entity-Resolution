"""
Runs the trained pipeline over the test set and writes:
  output/candidate_pairs.tsv   (blocking stage output, fed to the model)
  output/matching_results.tsv  (final thresholded matches)

Guarantees enforced here (matching the challenge's hard validation rules):
  - exactly one row per test Source-1 entity in both files
  - empty string (not "nan") for entities with no candidates/matches
  - no duplicate IDs within a list
  - only S2-/S3- ids that exist in the test set are ever emitted
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from . import blocking as blk
from . import features as feat

try:
    import joblib
except ImportError:
    joblib = None


def _load_source(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def score_candidates(model, s1_df, other_df, candidates, use_embeddings=True):
    packed_s1 = feat.row_pack(s1_df)
    packed_other = feat.row_pack(other_df)

    id_to_embed_idx = {}
    vecs = None
    if use_embeddings:
        all_ids = list(s1_df["entity_id"]) + list(other_df["entity_id"])
        all_texts = [
            packed_s1.get(i, packed_other.get(i, {})).get("name_norm", "") for i in all_ids
        ]
        vecs = feat.embed_texts(all_texts)
        if vecs is not None:
            id_to_embed_idx = {eid: idx for idx, eid in enumerate(all_ids)}

    scored = {}  # s1_id -> list[(cand_id, prob)]
    for s1_id in s1_df["entity_id"]:
        rec_a = packed_s1[s1_id]
        cand_ids = candidates.get(s1_id, [])
        if not cand_ids:
            scored[s1_id] = []
            continue
        rows = []
        keep_ids = []
        for cand_id in cand_ids:
            rec_b = packed_other.get(cand_id)
            if rec_b is None:
                continue
            cos = 0.0
            if vecs is not None and s1_id in id_to_embed_idx and cand_id in id_to_embed_idx:
                cos = float(np.dot(vecs[id_to_embed_idx[s1_id]], vecs[id_to_embed_idx[cand_id]]))
            rows.append(feat.pair_features(rec_a, rec_b, embed_cosine=cos))
            keep_ids.append(cand_id)
        if not rows:
            scored[s1_id] = []
            continue
        probs = model.predict_proba(np.array(rows, dtype=float))[:, 1]
        scored[s1_id] = list(zip(keep_ids, probs))
    return scored


def write_id_list_tsv(path, id_col, list_col, mapping: dict, ordered_ids):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{id_col}\t{list_col}\n")
        for eid in ordered_ids:
            ids = mapping.get(eid, [])
            f.write(f"{eid}\t{','.join(ids)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--threshold", type=float, default=None, help="override tuned threshold")
    args = ap.parse_args()

    test_dir = os.path.join(args.data_dir, "test")
    s1 = _load_source(os.path.join(test_dir, "test_source1.tsv"))
    s2 = _load_source(os.path.join(test_dir, "test_source2.tsv"))
    s3 = _load_source(os.path.join(test_dir, "test_source3.tsv"))
    other_df = pd.concat([s2, s3], ignore_index=True)

    with open(os.path.join(args.model_dir, "config.json")) as f:
        config = json.load(f)
    threshold = args.threshold if args.threshold is not None else config["threshold"]
    use_embeddings = config.get("use_embeddings", True)

    if joblib is None:
        raise RuntimeError("joblib is required to load the trained model (see requirements.txt)")
    model = joblib.load(os.path.join(args.model_dir, "classifier.joblib"))

    print(f"Generating candidates for {len(s1)} test Source-1 entities...")
    candidates = blk.build_candidates(s1, s2, s3)

    valid_other_ids = set(other_df["entity_id"])
    # sanity: candidates must only reference ids that exist in the test set
    candidates = {
        eid: [c for c in cids if c in valid_other_ids] for eid, cids in candidates.items()
    }

    print("Scoring candidates...")
    scored = score_candidates(model, s1, other_df, candidates, use_embeddings=use_embeddings)

    matches = {}
    for s1_id, scored_list in scored.items():
        chosen = sorted(
            [cid for cid, p in scored_list if p >= threshold],
            key=lambda cid: -dict(scored_list)[cid],
        )
        matches[s1_id] = list(dict.fromkeys(chosen))  # de-dup, preserve order

    os.makedirs(args.output_dir, exist_ok=True)
    ordered_ids = list(s1["entity_id"])

    write_id_list_tsv(
        os.path.join(args.output_dir, "candidate_pairs.tsv"),
        "source1_entity_id",
        "candidate_entity_ids",
        candidates,
        ordered_ids,
    )
    write_id_list_tsv(
        os.path.join(args.output_dir, "matching_results.tsv"),
        "source1_entity_id",
        "matched_entity_ids",
        matches,
        ordered_ids,
    )
    n_matched = sum(1 for v in matches.values() if v)
    print(
        f"Wrote {args.output_dir}/matching_results.tsv "
        f"({n_matched}/{len(ordered_ids)} entities matched, threshold={threshold})"
    )
    print(f"Wrote {args.output_dir}/candidate_pairs.tsv")


if __name__ == "__main__":
    main()
