# Copyright (C) 2021-2024, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, List, Union

import numpy as np
from scipy.special import softmax

from onnxtr.models.preprocessor import PreProcessor
from onnxtr.utils.repr import NestedObject

__all__ = ["OrientationPredictor"]


class OrientationPredictor(NestedObject):
    """Implements an object able to detect the reading direction of a text box or a page.
    4 possible orientations: 0, 90, 180, 270 (-90) degrees counter clockwise.

    Args:
    ----
        pre_processor: transform inputs for easier batched model inference
        model: core classification architecture (backbone + classification head)
    """

    _children_names: List[str] = ["pre_processor", "model"]

    def __init__(
        self,
        pre_processor: PreProcessor,
        model: Any,
    ) -> None:
        self.pre_processor = pre_processor
        self.model = model

    def __call__(
        self,
        inputs: List[np.ndarray],
    ) -> List[Union[List[int], List[float]]]:
        # Dimension check
        if any(input.ndim != 3 for input in inputs):
            raise ValueError("incorrect input shape: all inputs are expected to be multi-channel 2D images.")

        processed_batches = self.pre_processor(inputs)
        predicted_batches = [self.model(batch) for batch in processed_batches]

        # confidence
        probs = [np.max(softmax(batch, axis=1), axis=1) for batch in predicted_batches]
        # Postprocess predictions
        predicted_batches = [np.argmax(out_batch, axis=1) for out_batch in predicted_batches]

        class_idxs = [int(pred) for batch in predicted_batches for pred in batch]
        classes = [int(self.model.cfg["classes"][idx]) for idx in class_idxs]
        confs = [round(float(p), 2) for prob in probs for p in prob]

        return [class_idxs, classes, confs]
