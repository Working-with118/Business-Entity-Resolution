"""
Pairwise feature engineering for the S1-vs-candidate matching classifier.

All features are cheap, symmetric string-similarity signals computed on the
normalized name/address text produced by `preprocessing.py`, plus one
optional semantic-embedding feature (small, MIT-licensed sentence encoder)
that helps with paraphrase-style name variants that character n-grams miss
(e.g. "Global Freight Movers" vs "GFM Logistics" — unlikely to be caught by
string overlap alone, but any lexical/positional signal here is still
computed locally from the provided data only).
"""

import difflib

import numpy as np

from . import preprocessing as pp

try:
    import Levenshtein  # python-Levenshtein, optional but much faster

    def _lev_ratio(a, b):
        if not a and not b:
            return 1.0
        return Levenshtein.ratio(a, b)

except ImportError:  # pragma: no cover - fallback keeps the pipeline runnable

    def _lev_ratio(a, b):
        if not a and not b:
            return 1.0
        return difflib.SequenceMatcher(None, a, b).ratio()


def _jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


FEATURE_NAMES = [
    "name_jaccard",
    "name_core_jaccard",
    "name_lev_ratio",
    "name_len_diff",
    "address_jaccard",
    "address_lev_ratio",
    "country_match",
    "name_first_token_match",
    "token_count_diff",
    "name_embed_cosine",
]


def row_pack(df):
    """Precompute normalized fields once per dataframe; returns dict keyed by entity_id."""
    packed = {}
    for _, r in df.iterrows():
        name_norm = pp.normalize_name(r["business_name"])
        name_core = pp.normalize_name_core(r["business_name"])
        addr_norm = pp.normalize_address(r["business_address"])
        packed[r["entity_id"]] = {
            "name_norm": name_norm,
            "name_core": name_core,
            "name_tokens": pp.tokens(name_norm),
            "core_tokens": pp.tokens(name_core),
            "addr_norm": addr_norm,
            "addr_tokens": pp.tokens(addr_norm),
            "country": pp.normalize_country(r["country"]),
        }
    return packed


def pair_features(rec_a: dict, rec_b: dict, embed_cosine: float = 0.0) -> list:
    name_len_diff = abs(len(rec_a["name_norm"]) - len(rec_a["name_norm"])) if False else abs(
        len(rec_a["name_norm"]) - len(rec_b["name_norm"])
    )
    a_first = next(iter(sorted(rec_a["name_tokens"])), "")
    b_first = next(iter(sorted(rec_b["name_tokens"])), "")
    return [
        _jaccard(rec_a["name_tokens"], rec_b["name_tokens"]),
        _jaccard(rec_a["core_tokens"], rec_b["core_tokens"]),
        _lev_ratio(rec_a["name_norm"], rec_b["name_norm"]),
        name_len_diff,
        _jaccard(rec_a["addr_tokens"], rec_b["addr_tokens"]),
        _lev_ratio(rec_a["addr_norm"], rec_b["addr_norm"]),
        1.0 if rec_a["country"] == rec_b["country"] else 0.0,
        1.0 if (a_first and a_first == b_first) else 0.0,
        abs(len(rec_a["name_tokens"]) - len(rec_b["name_tokens"])),
        embed_cosine,
    ]


# ---------------------------------------------------------------------------
# Optional semantic embedding similarity (small MIT-licensed encoder).
# Kept fully optional: pipeline runs (with embed_cosine=0.0) even if
# sentence-transformers / the model weights aren't available offline.
# ---------------------------------------------------------------------------
_EMBEDDER = None


def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer

            # all-MiniLM-L6-v2: MIT license, ~22M params, well under the 8B cap.
            _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _EMBEDDER = False
    return _EMBEDDER


def embed_texts(texts):
    model = get_embedder()
    if not model:
        return None
    vecs = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs)
