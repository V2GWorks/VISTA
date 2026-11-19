import math

import pytest

from vista.evaluation import evaluate_images, is_defective_from_id, roc_auc, threshold_for_fpr
from vista.schema import InspectionResult


def r(image_id: str, score: float) -> InspectionResult:
    return InspectionResult(
        image_id=image_id,
        image_width=10,
        image_height=10,
        model="m",
        image_score=score,
        threshold=0.5,
    )


@pytest.mark.parametrize(
    ("image_id", "defective"),
    [
        ("bottle/test/good/000.png", False),
        ("bottle/test/broken_large/003.png", True),
        ("data/raw/mvtec_ad/carpet/test/hole/010.png", True),
    ],
)
def test_ground_truth_from_id(image_id, defective):
    assert is_defective_from_id(image_id) is defective


def test_ground_truth_rejects_train_ids():
    with pytest.raises(ValueError):
        is_defective_from_id("bottle/train/good/000.png")


def test_auroc_perfect_and_inverted():
    labels = [False, False, True, True]
    assert roc_auc([0.1, 0.2, 0.8, 0.9], labels) == 1.0
    assert roc_auc([0.9, 0.8, 0.2, 0.1], labels) == 0.0


def test_auroc_ties_count_half():
    assert roc_auc([0.5, 0.5], [False, True]) == 0.5


def test_auroc_matches_pairwise_definition():
    scores = [0.3, 0.7, 0.2, 0.9, 0.7, 0.1, 0.6]
    labels = [False, True, False, True, False, False, True]
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    pairwise = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
    assert math.isclose(roc_auc(scores, labels), pairwise)


def test_auroc_needs_both_classes():
    with pytest.raises(ValueError):
        roc_auc([0.1, 0.2], [False, False])


def test_threshold_respects_target_fpr():
    good = [i / 100 for i in range(100)]  # 0.00 .. 0.99
    t = threshold_for_fpr(good, 0.05)
    assert sum(s >= t for s in good) / len(good) <= 0.05
    assert sum(s >= t for s in good) == 5


def test_threshold_zero_fpr_flags_no_good_images():
    good = [0.1, 0.4, 0.4]
    t = threshold_for_fpr(good, 0.0)
    assert not any(s >= t for s in good)


def test_threshold_steps_above_ties():
    good = [0.1, 0.2, 0.9, 0.9, 0.9]
    t = threshold_for_fpr(good, 0.2)  # may flag 1 image, but 0.9 appears 3 times
    assert sum(s >= t for s in good) == 0


def test_evaluate_images():
    results = [
        r("bottle/test/good/000.png", 0.1),
        r("bottle/test/good/001.png", 0.6),
        r("bottle/test/broken_large/000.png", 0.7),
        r("bottle/test/contamination/000.png", 0.3),
    ]
    m = evaluate_images(results, threshold=0.5)
    assert (m.n_images, m.n_defective) == (4, 2)
    assert m.auroc == 0.75
    assert (m.precision, m.recall, m.false_positive_rate) == (0.5, 0.5, 0.5)
    assert m.f1 == 0.5
