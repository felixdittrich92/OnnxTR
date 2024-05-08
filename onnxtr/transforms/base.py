# Copyright (C) 2021-2024, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from typing import Tuple, Union, Optional

import cv2
import numpy as np

__all__ = ["Resize", "Normalize"]


from torch.nn.functional import pad
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T

import torch


class Resize(T.Resize):  # TODO: Translate me correct !!!
    """Resize the input image to the given size"""

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        interpolation=F.InterpolationMode.BILINEAR,
        preserve_aspect_ratio: bool = False,
        symmetric_pad: bool = False,
    ) -> None:
        super().__init__(size, interpolation, antialias=True)
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad

        if not isinstance(self.size, (int, tuple, list)):
            raise AssertionError("size should be either a tuple, a list or an int")

    def forward(
        self,
        img: torch.Tensor,
        target: Optional[np.ndarray] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, np.ndarray]]:
        img = np.transpose(img, (2, 0, 1))
        print(img.shape)
        img = torch.from_numpy(img)
        if isinstance(self.size, int):
            target_ratio = img.shape[-2] / img.shape[-1]
        else:
            target_ratio = self.size[0] / self.size[1]
        actual_ratio = img.shape[-2] / img.shape[-1]

        if not self.preserve_aspect_ratio or (target_ratio == actual_ratio and (isinstance(self.size, (tuple, list)))):
            return super().forward(img).numpy()
        else:
            # Resize
            if isinstance(self.size, (tuple, list)):
                if actual_ratio > target_ratio:
                    tmp_size = (self.size[0], max(int(self.size[0] / actual_ratio), 1))
                else:
                    tmp_size = (max(int(self.size[1] * actual_ratio), 1), self.size[1])
            elif isinstance(self.size, int):  # self.size is the longest side, infer the other
                if img.shape[-2] <= img.shape[-1]:
                    tmp_size = (max(int(self.size * actual_ratio), 1), self.size)
                else:
                    tmp_size = (self.size, max(int(self.size / actual_ratio), 1))

            # Scale image
            img = F.resize(img, tmp_size, self.interpolation, antialias=True)
            raw_shape = img.shape[-2:]
            if isinstance(self.size, (tuple, list)):
                # Pad (inverted in pytorch)
                _pad = (0, self.size[1] - img.shape[-1], 0, self.size[0] - img.shape[-2])
                if self.symmetric_pad:
                    half_pad = (math.ceil(_pad[1] / 2), math.ceil(_pad[3] / 2))
                    _pad = (half_pad[0], _pad[1] - half_pad[0], half_pad[1], _pad[3] - half_pad[1])
                img = pad(img, _pad)

            return img.numpy()

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
        print(self.mean, self.std)
        print(img.shape)
        img = np.transpose(img, (0, 3, 2, 1))
        mean = np.array(self.mean).astype(img.dtype)
        std = np.array(self.std).astype(img.dtype)
        img = (img - mean) / std
        img = np.transpose(img, (0, 3, 2, 1))
        print(img.shape)
        return img

    def __repr__(self) -> str:
        _repr = f"mean={self.mean}, std={self.std}"
        return f"{self.__class__.__name__}({_repr})"
