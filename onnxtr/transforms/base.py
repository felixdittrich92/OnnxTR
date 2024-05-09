# Copyright (C) 2021-2024, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

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
        self.size = size
        self.interpolation = interpolation
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad
        self.output_size = size if isinstance(size, tuple) else (size, size)

        if not isinstance(self.size, (int, tuple, list)):
            raise AssertionError("size should be either a tuple, a list or an int")

    def __call__(
        self,
        img: np.ndarray,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if img.ndim == 3:
            h, w = img.shape[0:2]
        else:
            h, w = img.shape[1:3]
        sh, sw = self.size

        # Calculate aspect ratio of the image
        aspect = w / h

        # Compute scaling and padding sizes
        if self.preserve_aspect_ratio:
            if aspect > 1:  # Horizontal image
                new_w = sw
                new_h = int(sw / aspect)
            elif aspect < 1:  # Vertical image
                new_h = sh
                new_w = int(sh * aspect)
            else:  # Square image
                new_h, new_w = sh, sw

            img_resized = cv2.resize(img, (new_w, new_h), interpolation=self.interpolation)

            # Calculate padding
            pad_top = max((sh - new_h) // 2, 0)
            pad_bottom = max(sh - new_h - pad_top, 0)
            pad_left = max((sw - new_w) // 2, 0)
            pad_right = max(sw - new_w - pad_left, 0)

            # Pad the image
            img_resized = cv2.copyMakeBorder(
                img_resized, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=0
            )

            # Ensure the image matches the target size by resizing it again if needed
            img_resized = cv2.resize(img_resized, (sw, sh), interpolation=self.interpolation)
        else:
            # Resize the image without preserving aspect ratio
            img_resized = cv2.resize(img, (sw, sh), interpolation=self.interpolation)

        return img_resized

    def __repr__(self) -> str:
        interpolate_str = self.interpolation
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
        return (img - np.array(self.mean).astype(img.dtype)) / np.array(self.std).astype(img.dtype)

    def __repr__(self) -> str:
        _repr = f"mean={self.mean}, std={self.std}"
        return f"{self.__class__.__name__}({_repr})"
