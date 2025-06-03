# Copyright (C) 2021-2025, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from collections.abc import Sequence
from typing import Any

import numpy as np

from onnxtr.models.preprocessor import PreProcessor
from onnxtr.utils.repr import NestedObject

from ._utils import remap_preds, split_crops

__all__ = ["RecognitionPredictor"]


class RecognitionPredictor(NestedObject):
    """Implements an object able to identify character sequences in images

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core recognition architecture
        split_wide_crops: wether to use crop splitting for high aspect ratio crops
    """

    def __init__(
        self,
        pre_processor: PreProcessor,
        model: Any,
        split_wide_crops: bool = True,
    ) -> None:
        super().__init__()
        self.pre_processor = pre_processor
        self.model = model
        self.split_wide_crops = split_wide_crops
        self.critical_ar = 8  # Critical aspect ratio
        self.overlap_ratio = 0.5  # Ratio of overlap between neighboring crops
        self.target_ar = 6  # Target aspect ratio

    def __call__(
        self,
        crops: Sequence[np.ndarray],
        **kwargs: Any,
    ) -> list[tuple[str, float]]:
        if len(crops) == 0:
            return []
        # Dimension check
        if any(crop.ndim != 3 for crop in crops):
            raise ValueError("incorrect input shape: all crops are expected to be multi-channel 2D images.")

        # Split crops that are too wide
        remapped = False
        if self.split_wide_crops:
            new_crops, crop_map, remapped = split_crops(
                crops,  # type: ignore[arg-type]
                self.critical_ar,
                self.target_ar,
                self.overlap_ratio,
                True,
            )
            if remapped:
                crops = new_crops

        # Resize & batch them
        processed_batches = self.pre_processor(crops)  # type: ignore[arg-type]

        # Forward it
        raw = [self.model(batch, **kwargs)["preds"] for batch in processed_batches]

        # Process outputs
        out = [charseq for batch in raw for charseq in batch]

        # Remap crops
        if self.split_wide_crops and remapped:
            out = remap_preds(out, crop_map, self.overlap_ratio)

        return out
