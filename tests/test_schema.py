import pytest
from pydantic import ValidationError

from vista.schema import BoundingBox, Defect, InspectionResult


def make_result(**overrides):
    base = dict(
        image_id="bottle/test/broken_large/000.png",
        image_width=900,
        image_height=900,
        model="patchcore-wrn50",
        image_score=0.82,
        threshold=0.5,
        defects=[Defect(bbox=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=70), score=0.9)],
    )
    base.update(overrides)
    return InspectionResult(**base)


def test_valid_result_round_trips_through_json():
    result = make_result()
    restored = InspectionResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.is_anomalous


def test_below_threshold_is_not_anomalous():
    assert not make_result(image_score=0.2, defects=[]).is_anomalous


def test_bbox_area():
    assert BoundingBox(x_min=0, y_min=0, x_max=10, y_max=5).area == 50


@pytest.mark.parametrize(
    "coords",
    [
        dict(x_min=10, y_min=0, x_max=10, y_max=5),  # zero width
        dict(x_min=0, y_min=8, x_max=5, y_max=3),  # inverted
    ],
)
def test_degenerate_bbox_rejected(coords):
    with pytest.raises(ValidationError):
        BoundingBox(**coords)


def test_defect_outside_image_rejected():
    outside = Defect(bbox=BoundingBox(x_min=850, y_min=850, x_max=950, y_max=890), score=0.7)
    with pytest.raises(ValidationError, match="beyond image"):
        make_result(defects=[outside])


@pytest.mark.parametrize("score", [-0.1, 1.5])
def test_scores_must_be_normalised(score):
    with pytest.raises(ValidationError):
        make_result(image_score=score)
