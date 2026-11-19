import pytest

from vista.schema import BoundingBox, Defect, InspectionResult
from vista.store import iter_results, load_results, save_results


def result(image_id: str, score: float) -> InspectionResult:
    return InspectionResult(
        image_id=image_id,
        image_width=900,
        image_height=900,
        model="patchcore-wrn50",
        image_score=score,
        threshold=0.5,
        defects=[Defect(bbox=BoundingBox(x_min=1, y_min=1, x_max=9, y_max=9), score=score)],
    )


def test_round_trip_and_append(tmp_path):
    path = tmp_path / "runs" / "run-001.jsonl"
    first = [result("bottle/test/good/000.png", 0.1)]
    second = [result("bottle/test/broken_large/000.png", 0.9)]

    assert save_results(path, first) == 1
    assert save_results(path, second) == 1
    assert load_results(path) == first + second


def test_blank_lines_skipped(tmp_path):
    path = tmp_path / "run.jsonl"
    save_results(path, [result("bottle/test/good/000.png", 0.2)])
    path.write_text(path.read_text() + "\n\n")
    assert len(load_results(path)) == 1


def test_invalid_line_reports_location(tmp_path):
    path = tmp_path / "run.jsonl"
    save_results(path, [result("bottle/test/good/000.png", 0.2)])
    with path.open("a") as f:
        f.write('{"image_id": "x"}\n')
    with pytest.raises(ValueError, match=r"run\.jsonl:2"):
        list(iter_results(path))
