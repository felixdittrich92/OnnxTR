# Copyright (C) 2021-2024, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Tuple, Union

import numpy as np
from PIL import Image, ImageOps

__all__ = ["Resize", "Normalize"]


class Resize:
    """Resize the input image to the given size

    Args:
        size: the target size of the image
        interpolation: the interpolation method to use
        preserve_aspect_ratio: whether to preserve the aspect ratio of the image
        symmetric_pad: whether to symmetrically pad the image
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        interpolation=Image.Resampling.BILINEAR,
        preserve_aspect_ratio: bool = False,
        symmetric_pad: bool = False,
    ) -> None:
        self.size = size if isinstance(size, tuple) else (size, size)
        self.interpolation = interpolation
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad
        self.output_size = size if isinstance(size, tuple) else (size, size)

        if not isinstance(self.size, (tuple, int)):
            raise AssertionError("size should be either a tuple or an int")

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img
        h, w = img.shape[:2] if img.ndim == 3 else img.shape[1:3]
        sh, sw = self.size

        if not self.preserve_aspect_ratio:
            return np.array(Image.fromarray(img).resize((sw, sh), resample=self.interpolation))

        actual_ratio = h / w
        target_ratio = sh / sw

        if target_ratio == actual_ratio:
            return np.array(Image.fromarray(img).resize((sw, sh), resample=self.interpolation))

        if actual_ratio > target_ratio:
            tmp_size = (int(sh / actual_ratio), sh)
        else:
            tmp_size = (sw, int(sw * actual_ratio))

        img_resized = Image.fromarray(img).resize(tmp_size, resample=self.interpolation)
        pad_left = pad_top = 0
        pad_right = sw - img_resized.width
        pad_bottom = sh - img_resized.height

        if self.symmetric_pad:
            pad_left = pad_right // 2
            pad_right -= pad_left
            pad_top = pad_bottom // 2
            pad_bottom -= pad_top

        img_resized = ImageOps.expand(img_resized, (pad_left, pad_top, pad_right, pad_bottom))
        return np.array(img_resized)

    def __repr__(self) -> str:
        interpolate_str = self.interpolation
        _repr = f"output_size={self.size}, interpolation='{interpolate_str}'"
        if self.preserve_aspect_ratio:
            _repr += f", preserve_aspect_ratio={self.preserve_aspect_ratio}, symmetric_pad={self.symmetric_pad}"
        return f"{self.__class__.__name__}({_repr})"


class Normalize:
    """Normalize the input image

    Args:
        mean: mean values to subtract
        std: standard deviation values to divide
    """

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
        return (img - np.array(self.mean).astype(img.dtype)) / np.array(self.std).astype(img.dtype)

    def __repr__(self) -> str:
        _repr = f"mean={self.mean}, std={self.std}"
        return f"{self.__class__.__name__}({_repr})"
