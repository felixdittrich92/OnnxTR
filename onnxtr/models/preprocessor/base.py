# Copyright (C) 2021-2026, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from typing import Any, overload

import numpy as np

from onnxtr.transforms import Normalize, Resize
from onnxtr.utils.common_types import Sample
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

    @overload
    def batch_inputs(self, samples: list[np.ndarray]) -> list[np.ndarray]: ...

    @overload
    def batch_inputs(self, samples: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]: ...

    def batch_inputs(
        self, samples: list[np.ndarray] | list[tuple[np.ndarray, np.ndarray]]
    ) -> list[np.ndarray] | list[tuple[np.ndarray, np.ndarray]]:
        """Gather samples into batches for inference purposes

        Args:
            samples: list of samples of shape (H, W, C) or list of tuples of samples and padding
                masks of shape (H, W, C) and (H, W) respectively

        Returns:
            list of batched samples, or list of tuples of batched samples and batched masks
        """
        num_batches = int(math.ceil(len(samples) / self.batch_size))

        if isinstance(samples[0], tuple):
            imgs, masks = zip(*samples)
            return [
                (
                    np.stack(imgs[idx * self.batch_size : min((idx + 1) * self.batch_size, len(imgs))], axis=0),
                    np.stack(masks[idx * self.batch_size : min((idx + 1) * self.batch_size, len(masks))], axis=0),
                )
                for idx in range(num_batches)
            ]

        return [
            np.stack(samples[idx * self.batch_size : min((idx + 1) * self.batch_size, len(samples))], axis=0)
            for idx in range(int(num_batches))
        ]

    def _check_sample(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 3:
            raise AssertionError("expected list of 3D Tensors")
        if isinstance(x, np.ndarray):
            if x.dtype not in (np.uint8, np.float32):
                raise TypeError("unsupported data type for numpy.ndarray")
        return shape_translate(x, "HWC")

    @staticmethod
    def _to_float(x: np.ndarray) -> np.ndarray:
        # Data type & 255 division
        return x.astype(np.float32) / 255.0 if x.dtype == np.uint8 else x

    def sample_transforms(self, x: np.ndarray) -> np.ndarray:
        return self._to_float(self.resize(Sample(image=self._check_sample(x))).image)

    def sample_transforms_with_mask(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sample = self.resize(Sample(image=self._check_sample(x)))
        # `Sample.mask` is loosely typed (Any | None); the caller only reaches this method when
        # the resize is configured to produce a padding mask, so it is always set here
        return self._to_float(sample.image), np.asarray(sample.mask)

    def __call__(self, x: np.ndarray | list[np.ndarray]) -> list[np.ndarray] | list[tuple[np.ndarray, np.ndarray]]:
        """Prepare document data for model forwarding

        Args:
            x: list of images (np.array) or tensors (already resized and batched)

        Returns:
            list of page batches, or - if the resize is configured to return padding masks -
            a list of (page batch, padding mask batch) tuples
        """
        return_mask = self.resize.return_padding_mask

        # Input type check
        if isinstance(x, np.ndarray):
            if x.ndim != 4:
                raise AssertionError("expected 4D Tensor")
            if x.dtype not in (np.uint8, np.float32):
                raise TypeError("unsupported data type for numpy.ndarray")
            x = shape_translate(x, "BHWC")

            masks: np.ndarray | None = None
            # Resizing
            if (x.shape[1], x.shape[2]) != self.resize.output_size:
                resized = [self.resize(Sample(image=sample)) for sample in x]
                x = np.stack([sample.image for sample in resized], axis=0)
                if return_mask:
                    masks = np.stack([np.asarray(sample.mask) for sample in resized], axis=0)
            # Data type & 255 division
            x = self._to_float(x)

            if return_mask:
                if masks is None:
                    # No resize took place: every pixel is valid image content
                    masks = np.ones((x.shape[0], x.shape[1], x.shape[2]), dtype=bool)
                normalized = list(multithread_exec(self.normalize, [x]))
                return [(normalized[0], masks)]

            return list(multithread_exec(self.normalize, [x]))

        if isinstance(x, list) and all(isinstance(sample, np.ndarray) for sample in x):
            if return_mask:
                # Sample transform (resize + mask), then batching
                mask_samples: list[tuple[np.ndarray, np.ndarray]] = list(
                    multithread_exec(self.sample_transforms_with_mask, x)
                )
                mask_batches = self.batch_inputs(mask_samples)
                # Batch transforms (normalize)
                img_batches = list(multithread_exec(self.normalize, [img for img, _ in mask_batches]))
                return [(img, mask) for img, (_, mask) in zip(img_batches, mask_batches)]

            # Sample transform (resize), batching, then batch transforms (normalize)
            samples: list[np.ndarray] = list(multithread_exec(self.sample_transforms, x))
            return list(multithread_exec(self.normalize, self.batch_inputs(samples)))

        raise TypeError(f"invalid input type: {type(x)}")
