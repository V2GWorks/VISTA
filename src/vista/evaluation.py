"""Image-level evaluation for anomaly detection.

Two things are kept apart on purpose:

* **Ranking quality** (AUROC) does not depend on a threshold, so it measures
  the model alone.
* **Operating point** (precision, recall, F1) depends on a threshold, which
  must be chosen on held-out *good* images, never on the test set. We pick it
  to hit a target false-positive rate on those images.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from vista.schema import InspectionResult


def is_defective_from_id(image_id: str) -> bool:
    """Ground truth from an MVTec-style id, e.g. ``bottle/test/broken_large/000.png``."""
    parts = image_id.replace("\\", "/").split("/")
    if "test" not in parts:
        raise ValueError(f"not a test image id: {image_id}")
    defect_type = parts[parts.index("test") + 1]
    return defect_type != "good"


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic.

    Equals the probability that a random defective image scores higher than a
    random good one, with ties counted as half. O(n log n), no dependencies.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must be the same length")
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs at least one defective and one good image")

    # Average ranks (1-based) with ties sharing the mean rank.
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        mean_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        i = j + 1

    rank_sum_pos = sum(r for r, y in zip(ranks, labels, strict=True) if y)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def threshold_for_fpr(good_scores: Sequence[float], target_fpr: float) -> float:
    """Smallest threshold whose false-positive rate on ``good_scores`` is <= ``target_fpr``.

    An image is flagged when ``score >= threshold``.
    """
    if not good_scores:
        raise ValueError("need at least one held-out good score")
    if not 0.0 <= target_fpr < 1.0:
        raise ValueError("target_fpr must be in [0, 1)")
    ordered = sorted(good_scores)
    allowed = math.floor(target_fpr * len(ordered))  # good images we may flag
    if allowed == 0:
        return math.nextafter(ordered[-1], math.inf)
    candidate = ordered[len(ordered) - allowed]
    # Ties at the cut would flag more than allowed; step above them.
    if ordered.count(candidate) > allowed:
        return math.nextafter(candidate, math.inf)
    return candidate


@dataclass(frozen=True)
class ImageMetrics:
    n_images: int
    n_defective: int
    auroc: float
    threshold: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_images(results: Sequence[InspectionResult], threshold: float) -> ImageMetrics:
    """Score a run of test-set results against MVTec ground truth at ``threshold``."""
    scores = [r.image_score for r in results]
    labels = [is_defective_from_id(r.image_id) for r in results]
    predicted = [s >= threshold for s in scores]

    tp = sum(p and y for p, y in zip(predicted, labels, strict=True))
    fp = sum(p and not y for p, y in zip(predicted, labels, strict=True))
    fn = sum(not p and y for p, y in zip(predicted, labels, strict=True))
    n_good = len(labels) - sum(labels)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return ImageMetrics(
        n_images=len(results),
        n_defective=sum(labels),
        auroc=roc_auc(scores, labels),
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fp / n_good if n_good else 0.0,
    )
