# Copyright (C) 2021-2026, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import math
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageOps

from onnxtr.utils.common_types import Sample

__all__ = ["Resize", "Normalize"]


class Resize:
    """Resize the input image to the given size

    >>> import numpy as np
    >>> from onnxtr.transforms import Resize
    >>> from onnxtr.utils import Sample
    >>> transfo = Resize((64, 64), preserve_aspect_ratio=True, symmetric_pad=True)
    >>> out = transfo(Sample(image=np.zeros((64, 64, 3), dtype=np.uint8)))

    Args:
        size: the target size of the image
        interpolation: the interpolation method to use
        preserve_aspect_ratio: whether to preserve the aspect ratio of the image
        symmetric_pad: whether to symmetrically pad the image
        return_padding_mask: whether to return a boolean mask alongside the image, with True on valid
            (image content) pixels and False on padded areas. Only used when the sample carries no
            mask of its own; an incoming mask is resized alongside the image instead.
    """

    def __init__(
        self,
        size: int | tuple[int, int],
        interpolation=Image.Resampling.BILINEAR,
        preserve_aspect_ratio: bool = False,
        symmetric_pad: bool = False,
        return_padding_mask: bool = False,
    ) -> None:
        self.size = size if isinstance(size, tuple) else (size, size)
        self.interpolation = interpolation
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad
        self.return_padding_mask = return_padding_mask
        self.output_size = size if isinstance(size, tuple) else (size, size)

        if not isinstance(self.size, (tuple, int)):
            raise AssertionError("size should be either a tuple or an int")

    def _resize_target(
        self,
        target: np.ndarray,
        raw_shape: Sequence[int],
        final_shape: Sequence[int],
        symmetric_pad: bool = False,
        offset: tuple[float, float] = (0, 0),
    ) -> np.ndarray:
        """Resize the target boxes according to the resizing of the image and the padding if needed"""
        target = target.copy()

        if target.shape[1:] == (4,):
            if symmetric_pad:
                target[:, [0, 2]] = offset[0] + target[:, [0, 2]] * raw_shape[-1] / final_shape[-1]
                target[:, [1, 3]] = offset[1] + target[:, [1, 3]] * raw_shape[-2] / final_shape[-2]
            else:
                target[:, [0, 2]] *= raw_shape[-1] / final_shape[-1]
                target[:, [1, 3]] *= raw_shape[-2] / final_shape[-2]

        elif target.shape[1:] == (4, 2):
            if symmetric_pad:
                target[..., 0] = offset[0] + target[..., 0] * raw_shape[-1] / final_shape[-1]
                target[..., 1] = offset[1] + target[..., 1] * raw_shape[-2] / final_shape[-2]
            else:
                target[..., 0] *= raw_shape[-1] / final_shape[-1]
                target[..., 1] *= raw_shape[-2] / final_shape[-2]

        else:
            raise AssertionError("Boxes should be in the format (n_boxes, 4, 2) or (n_boxes, 4)")

        return np.clip(target, 0, 1)

    @staticmethod
    def _to_pil(img: np.ndarray) -> Image.Image:
        if img.dtype != np.uint8:
            return Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))
        return Image.fromarray(img)

    @staticmethod
    def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        # Masks use nearest interpolation to preserve label integrity
        resized = Image.fromarray(mask.astype(np.uint8)).resize((size[1], size[0]), resample=Image.Resampling.NEAREST)
        return np.array(resized).astype(mask.dtype)

    def __call__(self, sample: Sample) -> Sample:
        img = sample.image
        target = sample.target
        mask = sample.mask
        # An incoming mask is resized alongside the image; the padding mask is only synthesised
        # when the sample carries no mask of its own
        resize_mask = mask is not None

        img_pil = self._to_pil(img)

        sh, sw = self.size
        w, h = img_pil.size

        target_ratio = sh / sw
        actual_ratio = h / w

        if not self.preserve_aspect_ratio or (target_ratio == actual_ratio):
            img = np.array(img_pil.resize((sw, sh), resample=self.interpolation))

            if resize_mask and mask is not None:
                mask = self._resize_mask(mask, self.size)
            elif self.return_padding_mask:
                # No padding was added: every pixel is valid image content
                mask = np.ones(self.size, dtype=bool)

            return sample.replace(image=img, mask=mask, target=target)

        # Resize
        if actual_ratio > target_ratio:
            tmp_size = (sh, max(int(sh / actual_ratio), 1))
        else:
            tmp_size = (max(int(sw * actual_ratio), 1), sw)

        # Scale image
        img_resized_pil = img_pil.resize((tmp_size[1], tmp_size[0]), resample=self.interpolation)
        raw_shape = tmp_size

        if resize_mask and mask is not None:
            mask = self._resize_mask(mask, tmp_size)

        delta_w = sw - tmp_size[1]
        delta_h = sh - tmp_size[0]

        if self.symmetric_pad:
            # Symmetric padding
            pad_left = math.ceil(delta_w / 2)
            pad_right = math.floor(delta_w / 2)
            pad_top = math.ceil(delta_h / 2)
            pad_bottom = math.floor(delta_h / 2)
        else:
            # Asymmetric padding
            pad_left, pad_top = 0, 0
            pad_right, pad_bottom = delta_w, delta_h

        img = np.array(
            ImageOps.expand(
                img_resized_pil,
                border=(pad_left, pad_top, pad_right, pad_bottom),
                fill=0,
            )
        )

        if resize_mask and mask is not None:
            mask = np.pad(mask, ((pad_top, pad_bottom), (pad_left, pad_right)))
        elif self.return_padding_mask:
            # True on valid (image content) pixels, False on the padded borders
            mask = np.zeros(self.size, dtype=bool)
            mask[pad_top : sh - pad_bottom, pad_left : sw - pad_right] = True

        # In case boxes are provided, resize boxes if needed (for detection task if preserve aspect ratio)
        if target is not None:
            offset = (pad_left / sw, pad_top / sh) if self.symmetric_pad else (0.0, 0.0)
            final_shape = (sh, sw)
            if isinstance(target, dict):
                target = {
                    cls_name: self._resize_target(
                        arr, raw_shape, final_shape, symmetric_pad=self.symmetric_pad, offset=offset
                    )
                    for cls_name, arr in target.items()
                }
            else:
                target = self._resize_target(
                    target, raw_shape, final_shape, symmetric_pad=self.symmetric_pad, offset=offset
                )

        return sample.replace(image=img, mask=mask, target=target)

    def __repr__(self) -> str:
        interpolate_str = self.interpolation
        _repr = f"output_size={self.size}, interpolation='{interpolate_str}'"
        if self.preserve_aspect_ratio:
            _repr += f", preserve_aspect_ratio={self.preserve_aspect_ratio}, symmetric_pad={self.symmetric_pad}"
        if self.return_padding_mask:
            _repr += f", return_padding_mask={self.return_padding_mask}"
        return f"{self.__class__.__name__}({_repr})"


class Normalize:
    """Normalize the input image

    Args:
        mean: mean values to subtract
        std: standard deviation values to divide
    """

    def __init__(
        self,
        mean: float | tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: float | tuple[float, float, float] = (0.229, 0.224, 0.225),
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
