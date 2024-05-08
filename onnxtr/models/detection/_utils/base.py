# Copyright (C) 2021-2024, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import cv2
import numpy as np

__all__ = ["erode", "dilate"]


def erode(x: np.ndarray, kernel_size: int) -> np.ndarray:
    """Performs erosion on a given tensor

    Args:
    ----
        x: boolean tensor of shape (N, H, W, C)
        kernel_size: the size of the kernel to use for erosion

    Returns:
    -------
        the eroded tensor
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return 1 - cv2.erode(1 - x.astype(np.uint8), kernel, iterations=1)


def dilate(x: np.ndarray, kernel_size: int) -> np.ndarray:
    """Performs dilation on a given tensor

    Args:
    ----
        x: boolean tensor of shape (N, H, W, C)
        kernel_size: the size of the kernel to use for dilation

    Returns:
    -------
        the dilated tensor
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(x.astype(np.uint8), kernel, iterations=1)
