"""
Candidate generation ("blocking") stage.

Goal: for every Source-1 entity, cheaply narrow the S2+S3 universe down to a
small, high-recall candidate set before the (more expensive) pairwise
classifier scores anything. Recall here is the hard ceiling on the whole
pipeline's final recall, so we deliberately combine two complementary
strategies:

1. TF-IDF (character n-gram) cosine nearest-neighbours on the normalized
   business name. Robust to typos, word-order swaps, partial abbreviation
   expansion, transliteration spelling differences.
2. Sorted-neighbourhood token-prefix blocking key. Cheap, catches cases the
   TF-IDF neighbour search might rank just outside top-K (e.g. very short
   names where char n-gram similarity is noisy).

Candidates from both strategies are unioned. Country is NOT used as a hard
filter (test set introduces an unseen country label, and country itself can
be noisy/mislabeled), but it is used to re-rank / cap the candidate list so
that same-country pairs are preferred when the pool is large.
"""

from collections import defaultdict

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from . import preprocessing as pp


def _char_ngram_vectorizer():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)


def build_candidates(
    s1_df,
    s2_df,
    s3_df,
    top_k: int = 15,
    min_sim: float = 0.10,
    max_candidates_per_entity: int = 25,
):
    """
    Returns: dict {s1_entity_id: [candidate_entity_id, ...]} (S2/S3 ids only).
    Expects each df to have columns: entity_id, business_name, business_address, country
    (raw, un-normalized is fine — normalization happens here).
    """
    s1 = s1_df.copy()
    s2 = s2_df.copy() if s2_df is not None else s2_df.iloc[0:0]
    s3 = s3_df.copy() if s3_df is not None else s3_df.iloc[0:0]

    for df in (s1, s2, s3):
        df["_name_norm"] = df["business_name"].map(pp.normalize_name)
        df["_name_core"] = df["business_name"].map(pp.normalize_name_core)
        df["_block_key"] = df["_name_core"].map(pp.blocking_key)
        df["_country_norm"] = df["country"].map(pp.normalize_country)

    others = []
    if len(s2):
        others.append(s2)
    if len(s3):
        others.append(s3)
    if not others:
        return {eid: [] for eid in s1["entity_id"]}
    other_df = others[0] if len(others) == 1 else __import__("pandas").concat(others, ignore_index=True)

    candidates = defaultdict(set)

    # --- Strategy 2: sorted-neighbourhood blocking key -----------------
    key_index = defaultdict(list)
    for idx, key in enumerate(other_df["_block_key"].values):
        if key:
            key_index[key].append(idx)
    other_ids = other_df["entity_id"].values
    for _, row in s1.iterrows():
        key = row["_block_key"]
        if key and key in key_index:
            for idx in key_index[key]:
                candidates[row["entity_id"]].add(other_ids[idx])

    # --- Strategy 1: TF-IDF char n-gram cosine nearest neighbours -------
    corpus = list(s1["_name_norm"].values) + list(other_df["_name_norm"].values)
    vec = _char_ngram_vectorizer()
    try:
        X = vec.fit_transform([c if c else " " for c in corpus])
    except ValueError:
        X = None

    if X is not None:
        n1 = len(s1)
        X1 = X[:n1]
        X2 = X[n1:]
        # Batch the similarity computation to keep memory bounded.
        batch = 500
        for start in range(0, n1, batch):
            end = min(start + batch, n1)
            sims = X1[start:end].dot(X2.T)  # sparse cosine-like scores (TF-IDF is L2-normalized)
            sims = sims.toarray()
            for i in range(sims.shape[0]):
                row_sims = sims[i]
                if row_sims.size == 0:
                    continue
                top_idx = np.argpartition(-row_sims, min(top_k, row_sims.size - 1))[:top_k]
                s1_id = s1["entity_id"].iloc[start + i]
                for j in top_idx:
                    if row_sims[j] >= min_sim:
                        candidates[s1_id].add(other_ids[j])

    # --- Cap + light country-aware ranking ------------------------------
    country_map = dict(zip(other_df["entity_id"], other_df["_country_norm"]))
    name_map = dict(zip(other_df["entity_id"], other_df["_name_core"]))
    result = {}
    for _, row in s1.iterrows():
        eid = row["entity_id"]
        cands = list(candidates.get(eid, []))
        if len(cands) > max_candidates_per_entity:
            s1_country = row["_country_norm"]
            s1_tokens = pp.tokens(row["_name_core"])

            def score(cid):
                country_bonus = 1.0 if country_map.get(cid) == s1_country else 0.0
                overlap = len(s1_tokens & pp.tokens(name_map.get(cid, "")))
                return (country_bonus, overlap)

            cands.sort(key=score, reverse=True)
            cands = cands[:max_candidates_per_entity]
        result[eid] = sorted(cands)
    return result


def candidates_to_df(candidates: dict):
    import pandas as pd

    rows = [
        {"source1_entity_id": s1_id, "candidate_entity_ids": ",".join(cids)}
        for s1_id, cids in candidates.items()
    ]
    return pd.DataFrame(rows)
