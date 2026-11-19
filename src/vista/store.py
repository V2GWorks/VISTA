"""Results storage: one JSON Lines file per run, one InspectionResult per line.

Append-only and diffable, so offline evaluation runs are easy to reproduce
and compare. Everything goes through ``InspectionResult``, so a database
backend can replace this module later without touching callers.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from vista.schema import InspectionResult


def save_results(path: Path, results: Iterable[InspectionResult]) -> int:
    """Append results to ``path``, creating parent folders. Returns the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for result in results:
            f.write(result.model_dump_json())
            f.write("\n")
            n += 1
    return n


def iter_results(path: Path) -> Iterator[InspectionResult]:
    """Yield results from ``path``, validating each line. Blank lines are skipped."""
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield InspectionResult.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: invalid result: {exc}") from exc


def load_results(path: Path) -> list[InspectionResult]:
    return list(iter_results(path))
