"""Extract and verify MVTec AD categories from a downloaded archive.

MVTec AD has to be downloaded from MVTec's site after accepting its licence
(CC BY-NC-SA 4.0), so this script works from a local archive rather than
fetching anything itself.

    python scripts/prepare_mvtec.py --archive ~/Downloads/mvtec_anomaly_detection.tar.xz \
        --category bottle --category carpet

Accepts the full archive or a single-category archive (e.g. bottle.tar.xz).
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "data" / "datasets" / "mvtec_ad.yaml"
DEFAULT_OUT = REPO_ROOT / "data" / "raw" / "mvtec_ad"
IMAGE_SUFFIX = ".png"


def known_categories(manifest: Path = MANIFEST) -> list[str]:
    spec = yaml.safe_load(manifest.read_text())
    cats = spec["categories"]
    return sorted(cats["objects"] + cats["textures"])


def _safe_members(tar: tarfile.TarFile, category: str) -> list[tarfile.TarInfo]:
    """Members belonging to `category`, rejecting any path that escapes the target."""
    members = []
    for m in tar.getmembers():
        parts = Path(m.name).parts
        if category not in parts:
            continue
        if m.name.startswith("/") or ".." in parts or not (m.isfile() or m.isdir()):
            raise ValueError(f"refusing unsafe archive member: {m.name}")
        members.append(m)
    return members


def extract_category(archive: Path, category: str, out_dir: Path) -> Path:
    with tarfile.open(archive) as tar:
        members = _safe_members(tar, category)
        if not members:
            raise ValueError(f"category '{category}' not found in {archive.name}")
        tar.extractall(out_dir, members=members, filter="data")
    # The full archive has no top-level folder, so the category may sit at
    # out_dir/<category>; locate it rather than assume.
    matches = [p for p in out_dir.rglob(category) if p.is_dir() and (p / "train").exists()]
    if not matches:
        raise ValueError(f"extracted '{category}' but found no train/ folder")
    return matches[0]


def verify_category(root: Path) -> dict:
    """Check the expected layout and return image counts per split."""
    train_good = root / "train" / "good"
    test_dir = root / "test"
    gt_dir = root / "ground_truth"
    for required in (train_good, test_dir / "good"):
        if not required.is_dir():
            raise ValueError(f"missing {required.relative_to(root)}")

    def count(d: Path) -> int:
        return sum(1 for p in d.iterdir() if p.suffix == IMAGE_SUFFIX)

    defect_types = sorted(d.name for d in test_dir.iterdir() if d.is_dir() and d.name != "good")
    missing_masks = []
    for defect in defect_types:
        for img in (test_dir / defect).glob(f"*{IMAGE_SUFFIX}"):
            if not (gt_dir / defect / f"{img.stem}_mask.png").exists():
                missing_masks.append(f"{defect}/{img.name}")
    if missing_masks:
        raise ValueError(f"{len(missing_masks)} test images lack masks, e.g. {missing_masks[0]}")

    return {
        "category": root.name,
        "train_good": count(train_good),
        "test_good": count(test_dir / "good"),
        "test_defective": {d: count(test_dir / d) for d in defect_types},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--category", action="append", required=True, dest="categories")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    valid = known_categories()
    unknown = [c for c in args.categories if c not in valid]
    if unknown:
        parser.error(f"unknown categories {unknown}; choose from {valid}")

    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for category in args.categories:
        root = extract_category(args.archive, category, args.out)
        summaries.append(verify_category(root))
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
