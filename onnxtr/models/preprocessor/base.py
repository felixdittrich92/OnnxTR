# Copyright (C) 2021-2026, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
import os
from typing import Any, overload

import numpy as np

from onnxtr.transforms import Normalize, Resize
from onnxtr.utils.common_types import Sample
from onnxtr.utils.geometry import shape_translate, to_rgb
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
        return shape_translate(to_rgb(x), "HWC")

    @staticmethod
    def _to_float(x: np.ndarray) -> np.ndarray:
        # Data type & 255 division
        if x.dtype != np.uint8:
            return x
        # In place on the fresh float copy: a single allocation instead of two
        out = x.astype(np.float32)
        out /= 255.0
        return out

    # Below this average amount of work per sample (input pixels + a quarter of the output pixels, the padding of the
    # output being cheap), the sample transforms run faster on the calling thread: e.g. for word crops resized to
    # 32 x 128 or 256 x 256, threading is up to ~3x slower because the resize of such small images is too short to
    # make up for the GIL hand-offs between the worker threads, while larger images are resized ~3-4x faster with it
    _min_pixels_per_threaded_sample: int = 64 * 1024

    def _sample_threads(self, samples: list[np.ndarray]) -> int | None:
        """Number of worker threads for the sample transforms: 1 (no threading) for small samples, else the default"""
        if not samples:
            return 1
        out_h, out_w = self.resize.output_size
        # `size` counts the channels too: divide by 3 to approximate the number of pixels
        mean_pixels = sum(sample.size for sample in samples) / (3 * len(samples))
        return 1 if mean_pixels + out_h * out_w / 4 < self._min_pixels_per_threaded_sample else None

    def _resize_sample(self, x: np.ndarray) -> np.ndarray:
        return self.resize(Sample(image=self._check_sample(x))).image

    def _resize_with_mask(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sample = self.resize(Sample(image=self._check_sample(x)))
        return sample.image, np.asarray(sample.mask)

    def _batch_transforms(self, batch: np.ndarray, threads: int = 1) -> np.ndarray:
        """Cast & normalize a batch built by `batch_inputs`: a fresh array, which is normalized in place, by chunks of
        rows on `threads` worker threads"""
        if threads < 2 or batch.ndim != 4 or batch.shape[1] < threads:
            # `_to_float` returns a new float array for uint8 batches, else the (freshly stacked) batch itself
            return self.normalize(self._to_float(batch), inplace=True)
        out = np.empty(batch.shape, dtype=np.float32) if batch.dtype == np.uint8 else batch
        bounds = np.linspace(0, batch.shape[1], threads + 1).astype(int)

        def _transform_rows(idx: int) -> None:
            rows = slice(bounds[idx], bounds[idx + 1])
            if out is not batch:
                np.divide(batch[:, rows], np.float32(255.0), out=out[:, rows])
            self.normalize(out[:, rows], inplace=True)

        list(multithread_exec(_transform_rows, range(threads), threads))
        return out

    # From this size (in elements), a single batch is cast & normalized ~3x faster on several threads
    _min_elements_per_threaded_batch: int = 1024 * 1024

    def _transform_batches(self, batches: list[np.ndarray]) -> list[np.ndarray]:
        if len(batches) == 1 and batches[0].size >= self._min_elements_per_threaded_batch:
            return [self._batch_transforms(batches[0], threads=min(8, os.cpu_count() or 1))]
        return list(multithread_exec(self._batch_transforms, batches))

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
            threads = self._sample_threads(x)
            # The samples are resized (uint8 images), batched, and only then cast to float & normalized batch by
            # batch: same values as casting each sample, while moving 4x less data around (uint8 stacking) and
            # normalizing the freshly allocated batches in place instead of allocating two float copies of each
            if return_mask:
                # Sample transform (resize + mask), then batching
                mask_samples: list[tuple[np.ndarray, np.ndarray]] = list(
                    multithread_exec(self._resize_with_mask, x, threads)
                )
                mask_batches = self.batch_inputs(mask_samples)
                # Batch transforms (cast & normalize)
                img_batches = self._transform_batches([img for img, _ in mask_batches])
                return [(img, mask) for img, (_, mask) in zip(img_batches, mask_batches)]

            # Sample transform (resize), batching, then batch transforms (cast & normalize)
            samples: list[np.ndarray] = list(multithread_exec(self._resize_sample, x, threads))
            return self._transform_batches(self.batch_inputs(samples))

        raise TypeError(f"invalid input type: {type(x)}")
