"""Span-alignment scoring for validating the engine's own findings.

Predicted discrepancy spans and ground-truth spans rarely coincide exactly, so
scoring by string equality understates and scoring by any-overlap overstates.
Instead we build a bipartite graph over normalised sentence units: an edge joins
a predicted span to a ground-truth span when their unit sets overlap beyond a
threshold, and each connected component containing at least one span of each
kind counts as one true positive (a "match group").
"""
import re


def sentence_units(text: str) -> list[tuple[int, int, str]]:
    units, position = [], 0
    for raw in re.split(r"(?<=[.;:!?])\s+|\n{2,}", text):
        stripped = raw.strip()
        if not stripped:
            position += len(raw) + 1
            continue
        start = text.find(stripped, position)
        if start < 0:
            start = position
        units.append((start, start + len(stripped), re.sub(r"\s+", " ", stripped.lower())))
        position = start + len(stripped)
    return units


def _units_for_span(units, start: int, end: int) -> set[int]:
    return {i for i, (u_start, u_end, _) in enumerate(units) if u_start < end and start < u_end}


def score(text: str, predicted: list[dict], ground_truth: list[dict], overlap: float = 0.3) -> dict:
    """predicted/ground_truth: [{"start": int, "end": int, ...}]"""
    units = sentence_units(text)
    pred_units = [_units_for_span(units, p["start"], p["end"]) for p in predicted]
    gt_units = [_units_for_span(units, g["start"], g["end"]) for g in ground_truth]

    # Union-find over (predicted, ground-truth) nodes.
    n_p, n_g = len(predicted), len(ground_truth)
    parent = list(range(n_p + n_g))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    edges = []
    for pi, pu in enumerate(pred_units):
        for gi, gu in enumerate(gt_units):
            if not pu or not gu:
                continue
            jaccard = len(pu & gu) / len(pu | gu)
            if jaccard >= overlap:
                union(pi, n_p + gi)
                edges.append({"predicted": pi, "ground_truth": gi, "overlap": round(jaccard, 3)})

    components: dict[int, dict] = {}
    for i in range(n_p + n_g):
        root = find(i)
        comp = components.setdefault(root, {"predicted": [], "ground_truth": []})
        (comp["predicted"] if i < n_p else comp["ground_truth"]).append(i if i < n_p else i - n_p)

    match_groups = [c for c in components.values() if c["predicted"] and c["ground_truth"]]
    tp = len(match_groups)
    matched_pred = {i for c in match_groups for i in c["predicted"]}
    matched_gt = {i for c in match_groups for i in c["ground_truth"]}
    fp = n_p - len(matched_pred)
    fn = n_g - len(matched_gt)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "match_groups": match_groups,
        "edges": edges,
        "units": len(units),
        "unmatched_predicted": sorted(set(range(n_p)) - matched_pred),
        "unmatched_ground_truth": sorted(set(range(n_g)) - matched_gt),
    }
