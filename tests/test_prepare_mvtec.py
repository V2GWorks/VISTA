import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_mvtec.py"
spec = importlib.util.spec_from_file_location("prepare_mvtec", SCRIPT)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def _add(tar: tarfile.TarFile, name: str, data: bytes = b"png") -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def make_archive(path: Path, *, drop_mask: bool = False) -> Path:
    """A miniature MVTec-style archive for one category."""
    with tarfile.open(path, "w:xz") as tar:
        _add(tar, "bottle/train/good/000.png")
        _add(tar, "bottle/train/good/001.png")
        _add(tar, "bottle/test/good/000.png")
        _add(tar, "bottle/test/broken_large/000.png")
        if not drop_mask:
            _add(tar, "bottle/ground_truth/broken_large/000_mask.png")
        _add(tar, "carpet/train/good/000.png")
    return path


def test_manifest_lists_all_fifteen_categories():
    cats = prepare.known_categories()
    assert len(cats) == 15
    assert {"bottle", "carpet", "zipper"} <= set(cats)


def test_extracts_and_counts_only_requested_category(tmp_path):
    archive = make_archive(tmp_path / "mvtec.tar.xz")
    root = prepare.extract_category(archive, "bottle", tmp_path / "out")
    summary = prepare.verify_category(root)
    assert summary == {
        "category": "bottle",
        "train_good": 2,
        "test_good": 1,
        "test_defective": {"broken_large": 1},
    }
    assert not (tmp_path / "out" / "carpet").exists()


def test_missing_mask_is_reported(tmp_path):
    archive = make_archive(tmp_path / "mvtec.tar.xz", drop_mask=True)
    root = prepare.extract_category(archive, "bottle", tmp_path / "out")
    with pytest.raises(ValueError, match="lack masks"):
        prepare.verify_category(root)


def test_unknown_category_in_archive(tmp_path):
    archive = make_archive(tmp_path / "mvtec.tar.xz")
    with pytest.raises(ValueError, match="not found"):
        prepare.extract_category(archive, "screw", tmp_path / "out")


def test_path_traversal_rejected(tmp_path):
    archive = tmp_path / "evil.tar.xz"
    with tarfile.open(archive, "w:xz") as tar:
        _add(tar, "bottle/../../escape.png")
    with pytest.raises(ValueError, match="unsafe"):
        prepare.extract_category(archive, "bottle", tmp_path / "out")
