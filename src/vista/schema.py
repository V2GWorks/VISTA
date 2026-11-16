"""Inspection result schema.

Every stage of the pipeline (model, API, report writer, evaluation) exchanges
data in this shape, so it is defined once and validated at the boundaries.
Coordinates are in pixels of the original image, origin top-left.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "0.1"


class BoundingBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    x_min: int = Field(ge=0)
    y_min: int = Field(ge=0)
    x_max: int = Field(gt=0)
    y_max: int = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> BoundingBox:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("bounding box must have positive width and height")
        return self

    @property
    def area(self) -> int:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)


class Defect(BaseModel):
    """One localised region the model considers anomalous."""

    model_config = ConfigDict(frozen=True)

    bbox: BoundingBox
    score: float = Field(ge=0.0, le=1.0, description="Normalised anomaly score for the region.")
    label: str | None = Field(
        default=None,
        description="Defect type if a classifier assigned one; None for unsupervised detection.",
    )
    mask_path: str | None = Field(
        default=None, description="Relative path to a binary mask PNG for this region."
    )


class InspectionResult(BaseModel):
    """The outcome of inspecting one image."""

    schema_version: Literal["0.1"] = SCHEMA_VERSION
    image_id: str = Field(min_length=1)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    model: str = Field(min_length=1, description="Model identifier, e.g. 'patchcore-wrn50'.")
    image_score: float = Field(ge=0.0, le=1.0, description="Image-level anomaly score.")
    threshold: float = Field(ge=0.0, le=1.0)
    defects: list[Defect] = Field(default_factory=list)
    inspected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_anomalous(self) -> bool:
        return self.image_score >= self.threshold

    @model_validator(mode="after")
    def _defects_inside_image(self) -> InspectionResult:
        for d in self.defects:
            if d.bbox.x_max > self.image_width or d.bbox.y_max > self.image_height:
                raise ValueError(
                    f"defect bbox {d.bbox} extends beyond image "
                    f"{self.image_width}x{self.image_height}"
                )
        return self
