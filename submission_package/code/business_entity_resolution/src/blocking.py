"""
Candidate generation ("blocking") stage.

Goal: for every Source-1 entity, cheaply narrow the S2+S3 universe down to a
small, high-recall candidate set before the (more expensive) pairwise
classifier scores anything. Recall here is the hard ceiling on the whole
pipeline's final recall.

History / why this is purely key-based now: an earlier version added a
TF-IDF (character n-gram) cosine nearest-neighbour search on top of the
blocking keys, meant to catch near-misses the keys alone would drop. At real
dataset scale (~10.3M "other" records) that similarity search was measured,
across several rounds of fixing, at anywhere from ~150 hours to ~9.6 DAYS
projected -- the fundamental problem was comparing every S1 record against
(effectively) the whole other-side corpus, whether done as a dense matrix,
a chunked sparse matrix, or an inverted-index-restricted lookup: any
absolute-count max_df cap still leaves per-document candidate pools that
don't shrink fast enough, and the up-front TF-IDF fit_transform over the
full corpus is itself a real cost at that size. No amount of chunking or
vocabulary tuning got it under a few dozen hours.

The fix that actually worked: measuring how many Source-1 entities the cheap
key-based strategy alone fails to cover (~11%), then adding two MORE cheap,
purely deterministic O(n) dict-lookup keys targeted at exactly that gap
(address numeric identifiers, and a looser 2-token name key) instead of
reaching for a similarity search at all. Together the three keys covered
99.4% of entities in testing -- so there's no similarity search left in this
module, and blocking is now O(n) in total record count, not O(n1 * n_other).
"""

from collections import defaultdict

from . import preprocessing as pp


def build_candidates(
    s1_df,
    s2_df,
    s3_df,
    max_candidates_per_entity: int = 25,
):
    """
    Returns: dict {s1_entity_id: [candidate_entity_id, ...]} (S2/S3 ids only).
    Expects each df to have columns: entity_id, business_name, business_address, country
    (raw, un-normalized is fine — normalization happens here).

    Memory note: this works off plain numpy/list arrays derived from the input
    frames rather than `.copy()`-ing and `pd.concat()`-ing them, to avoid
    holding several redundant full copies of the ~10.3M-row "other" pool.
    """
    n1 = len(s1_df)
    s1_name_core = s1_df["business_name"].map(pp.normalize_name_core).to_numpy()
    s1_addr_norm = s1_df["business_address"].map(pp.normalize_address).to_numpy()
    s1_country = s1_df["country"].map(pp.normalize_country).to_numpy()
    s1_ids = s1_df["entity_id"].to_numpy()

    s1_key1 = [pp.blocking_key(nc) for nc in s1_name_core]
    s1_key2 = [pp.address_block_key(a, c) for a, c in zip(s1_addr_norm, s1_country)]
    s1_key3 = [pp.loose_name_key(nc) for nc in s1_name_core]

    other_ids_parts, other_name_core_parts, other_country_parts = [], [], []
    other_key1_parts, other_key2_parts, other_key3_parts = [], [], []
    for df in (s2_df, s3_df):
        if df is None or len(df) == 0:
            continue
        name_core = df["business_name"].map(pp.normalize_name_core)
        addr_norm = df["business_address"].map(pp.normalize_address)
        country = df["country"].map(pp.normalize_country)
        other_ids_parts.append(df["entity_id"].to_numpy())
        other_name_core_parts.append(name_core.to_numpy())
        other_country_parts.append(country.to_numpy())
        other_key1_parts.append([pp.blocking_key(nc) for nc in name_core])
        other_key2_parts.append([pp.address_block_key(a, c) for a, c in zip(addr_norm, country)])
        other_key3_parts.append([pp.loose_name_key(nc) for nc in name_core])

    if not other_ids_parts:
        return {eid: [] for eid in s1_ids}

    other_ids = [x for part in other_ids_parts for x in part]
    other_name_core = [x for part in other_name_core_parts for x in part]
    other_country = [x for part in other_country_parts for x in part]
    other_key1 = [x for part in other_key1_parts for x in part]
    other_key2 = [x for part in other_key2_parts for x in part]
    other_key3 = [x for part in other_key3_parts for x in part]
    del other_ids_parts, other_name_core_parts, other_country_parts
    del other_key1_parts, other_key2_parts, other_key3_parts

    candidates = defaultdict(set)

    def _index_and_match(s1_keys, other_keys):
        idx = defaultdict(list)
        for j, key in enumerate(other_keys):
            if key:
                idx[key].append(j)
        for i in range(n1):
            key = s1_keys[i]
            if key and key in idx:
                for j in idx[key]:
                    candidates[s1_ids[i]].add(other_ids[j])

    # Three cheap, complementary, purely deterministic O(n) keys -- see
    # module docstring for why there's no similarity search here anymore.
    _index_and_match(s1_key1, other_key1)  # strict: every token's prefix must match
    _index_and_match(s1_key2, other_key2)  # address numeric identifiers + country
    _index_and_match(s1_key3, other_key3)  # loose: only the two leading tokens

    # --- Cap + light country-aware ranking ------------------------------
    country_map = dict(zip(other_ids, other_country))
    name_map = dict(zip(other_ids, other_name_core))
    result = {}
    for i in range(n1):
        eid = s1_ids[i]
        cands = list(candidates.get(eid, []))
        if len(cands) > max_candidates_per_entity:
            s1_country_i = s1_country[i]
            s1_tokens = pp.tokens(s1_name_core[i])

            def score(cid):
                country_bonus = 1.0 if country_map.get(cid) == s1_country_i else 0.0
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