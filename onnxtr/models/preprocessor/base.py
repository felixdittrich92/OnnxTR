# Copyright (C) 2021-2025, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from typing import Any

import numpy as np

from onnxtr.transforms import Normalize, Resize
from onnxtr.utils.geometry import shape_translate
from onnxtr.utils.multithreading import multithread_exec
from onnxtr.utils.repr import NestedObject

__all__ = ["PreProcessor"]


class PreProcessor(NestedObject):
    """Implements an abstract preprocessor object which performs casting, resizing, batching and normalization.

    Args:
        output_size: expected size of each page in format (H, W)
        batch_size: the size of page batches
        mean: mean value of the training distribution by channel
        std: standard deviation of the training distribution by channel
        **kwargs: additional arguments for the resizing operation
    """

    _children_names: list[str] = ["resize", "normalize"]

    def __init__(
        self,
        output_size: tuple[int, int],
        batch_size: int,
        mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        **kwargs: Any,
    ) -> None:
        self.batch_size = batch_size
        self.resize = Resize(output_size, **kwargs)
        self.normalize = Normalize(mean, std)

    def batch_inputs(self, samples: list[np.ndarray]) -> list[np.ndarray]:
        """Gather samples into batches for inference purposes

        Args:
            samples: list of samples (tf.Tensor)

        Returns:
            list of batched samples
        """
        num_batches = int(math.ceil(len(samples) / self.batch_size))
        batches = [
            np.stack(samples[idx * self.batch_size : min((idx + 1) * self.batch_size, len(samples))], axis=0)
            for idx in range(int(num_batches))
        ]

        return batches

    def sample_transforms(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 3:
            raise AssertionError("expected list of 3D Tensors")
        if isinstance(x, np.ndarray):
            if x.dtype not in (np.uint8, np.float32):
                raise TypeError("unsupported data type for numpy.ndarray")
        x = shape_translate(x, "HWC")

        # Resizing
        x = self.resize(x)
        # Data type & 255 division
        if x.dtype == np.uint8:
            x = x.astype(np.float32) / 255.0

        return x

    def __call__(self, x: np.ndarray | list[np.ndarray]) -> list[np.ndarray]:
        """Prepare document data for model forwarding

        Args:
            x: list of images (np.array) or tensors (already resized and batched)

        Returns:
            list of page batches
        """
        # Input type check
        if isinstance(x, np.ndarray):
            if x.ndim != 4:
                raise AssertionError("expected 4D Tensor")
            if isinstance(x, np.ndarray):
                if x.dtype not in (np.uint8, np.float32):
                    raise TypeError("unsupported data type for numpy.ndarray")
            x = shape_translate(x, "BHWC")

            # Resizing
            if (x.shape[1], x.shape[2]) != self.resize.output_size:
                x = np.array([self.resize(sample) for sample in x])
            # Data type & 255 division
            if x.dtype == np.uint8:
                x = x.astype(np.float32) / 255.0
            batches = [x]

        elif isinstance(x, list) and all(isinstance(sample, np.ndarray) for sample in x):
            # Sample transform (to tensor, resize)
            samples = list(multithread_exec(self.sample_transforms, x))
            # Batching
            batches = self.batch_inputs(samples)
        else:
            raise TypeError(f"invalid input type: {type(x)}")

        # Batch transforms (normalize)
        batches = list(multithread_exec(self.normalize, batches))

        return batches
