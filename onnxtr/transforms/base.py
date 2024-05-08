# Copyright (C) 2021-2024, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from typing import Tuple, Union

import cv2
import numpy as np

__all__ = ["Resize", "Normalize"]


class Resize:
    """Resize the input image to the given size"""

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        interpolation=cv2.INTER_LINEAR,
        preserve_aspect_ratio: bool = False,
        symmetric_pad: bool = False,
    ) -> None:
        super().__init__()
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad
        self.interpolation = interpolation

        if not isinstance(size, (int, tuple, list)):
            raise AssertionError("size should be either a tuple, a list or an int")
        self.size = size

    def __call__(
        self,
        img: np.ndarray,
    ) -> np.ndarray:
        if isinstance(self.size, int):
            target_ratio = img.shape[1] / img.shape[0]
        else:
            target_ratio = self.size[0] / self.size[1]
        actual_ratio = img.shape[1] / img.shape[0]

        # Resize
        if isinstance(self.size, (tuple, list)):
            if actual_ratio > target_ratio:
                tmp_size = (self.size[0], max(int(self.size[0] / actual_ratio), 1))
            else:
                tmp_size = (max(int(self.size[1] * actual_ratio), 1), self.size[1])
        elif isinstance(self.size, int):  # self.size is the longest side, infer the other
            if img.shape[0] <= img.shape[1]:
                tmp_size = (max(int(self.size * actual_ratio), 1), self.size)
            else:
                tmp_size = (self.size, max(int(self.size / actual_ratio), 1))

        # Scale image
        img = cv2.resize(img, tmp_size, interpolation=self.interpolation)

        if isinstance(self.size, (tuple, list)):
            # Pad
            _pad = (0, self.size[1] - img.shape[0], 0, self.size[0] - img.shape[1])
            if self.symmetric_pad:
                half_pad = (math.ceil(_pad[1] / 2), math.ceil(_pad[3] / 2))
                _pad = (half_pad[0], _pad[1] - half_pad[0], half_pad[1], _pad[3] - half_pad[1])
            img = np.pad(img, ((_pad[0], _pad[1]), (_pad[2], _pad[3]), (0, 0)), mode="constant")

        return img

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        _repr = f"output_size={self.size}, interpolation='{interpolate_str}'"
        if self.preserve_aspect_ratio:
            _repr += f", preserve_aspect_ratio={self.preserve_aspect_ratio}, symmetric_pad={self.symmetric_pad}"
        return f"{self.__class__.__name__}({_repr})"


class Normalize:
    """Normalize the input image"""

    def __init__(
        self,
        mean: Union[float, Tuple[float, float, float]] = (0.485, 0.456, 0.406),
        std: Union[float, Tuple[float, float, float]] = (0.229, 0.224, 0.225),
    ) -> None:
        self.mean = mean
        self.std = std

        if not isinstance(self.mean, (float, tuple, list)):
            raise AssertionError("mean should be either a tuple, a list or a float")
        if not isinstance(self.std, (float, tuple, list)):
            raise AssertionError("std should be either a tuple, a list or a float")

    def __call__(
        self,
        img: np.ndarray,
    ) -> np.ndarray:
        # Normalize image
        return (img - self.mean) / self.std

    def __repr__(self) -> str:
        _repr = f"mean={self.mean}, std={self.std}"
        return f"{self.__class__.__name__}({_repr})"
