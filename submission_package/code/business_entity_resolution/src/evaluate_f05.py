"""
Macro-averaged F_0.5 scorer, matching the challenge's evaluation exactly:
computed per Source-1 entity, then averaged. A correctly-predicted empty
match list (true singleton) scores 1.0; any predicted match for a true
singleton scores 0.0.
"""

BETA2 = 0.5 ** 2  # beta = 0.5 -> beta^2 = 0.25


def f_beta_for_entity(pred_ids: set, true_ids: set) -> float:
    if not true_ids and not pred_ids:
        return 1.0
    if not true_ids and pred_ids:
        return 0.0
    if true_ids and not pred_ids:
        return 0.0
    tp = len(pred_ids & true_ids)
    precision = tp / len(pred_ids) if pred_ids else 0.0
    recall = tp / len(true_ids) if true_ids else 0.0
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return (1 + BETA2) * precision * recall / (BETA2 * precision + recall)


def macro_f05(predictions: dict, ground_truth: dict) -> float:
    """
    predictions / ground_truth: {source1_entity_id: set_of_matched_ids}
    Averaged over every key present in ground_truth (missing prediction keys
    are treated as an empty predicted set).
    """
    scores = []
    for s1_id, true_ids in ground_truth.items():
        pred_ids = predictions.get(s1_id, set())
        scores.append(f_beta_for_entity(pred_ids, true_ids))
    return sum(scores) / len(scores) if scores else 0.0


def parse_id_list(cell) -> set:
    if cell is None:
        return set()
    cell = str(cell).strip()
    if not cell or cell.lower() == "nan":
        return set()
    return set(x.strip() for x in cell.split(",") if x.strip())


def load_gt_or_pred_tsv(path, id_col, list_col):
    import pandas as pd

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {row[id_col]: parse_id_list(row[list_col]) for _, row in df.iterrows()}
