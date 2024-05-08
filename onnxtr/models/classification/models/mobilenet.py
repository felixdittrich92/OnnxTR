# Copyright (C) 2021-2024, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Greatly inspired by https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv3.py

from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np

from ...engine import Engine

__all__ = [
    "mobilenet_v3_small_crop_orientation",
    "mobilenet_v3_small_page_orientation",
]

default_cfgs: Dict[str, Dict[str, Any]] = {
    "mobilenet_v3_small_crop_orientation": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 256, 256),
        "classes": [0, -90, 180, 90],
        "url": "https://github.com/felixdittrich92/OnnxTR/releases/download/v0.0.1/mobilenet_v3_small_crop_orientation-5620cf7e.onnx",
    },
    "mobilenet_v3_small_page_orientation": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 512, 512),
        "classes": [0, -90, 180, 90],
        "url": "https://github.com/felixdittrich92/OnnxTR/releases/download/v0.0.1/mobilenet_v3_small_page_orientation-d3f76d79.onnx",
    },
}


class MobileNetV3(Engine):
    """MobileNetV3 Onnx loader

    Args:
    ----
        model_path: path or url to onnx model file
        cfg: configuration dictionary
    """

    def __init__(
        self,
        model_path: str,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(url=model_path)
        self.cfg = cfg

    def __call__(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        return self.run(x)


def _mobilenet_v3(
    arch: str,
    model_path: str,
    **kwargs: Any,
) -> MobileNetV3:
    _cfg = deepcopy(default_cfgs[arch])
    return MobileNetV3(model_path, cfg=_cfg, **kwargs)


def mobilenet_v3_small_crop_orientation(
    model_path: str = default_cfgs["mobilenet_v3_small_crop_orientation"]["url"], **kwargs: Any
) -> MobileNetV3:
    """MobileNetV3-Small architecture as described in
    `"Searching for MobileNetV3",
    <https://arxiv.org/pdf/1905.02244.pdf>`_.

    >>> import numpy as np
    >>> from onnxtr.models import mobilenet_v3_small_crop_orientation
    >>> model = mobilenet_v3_small_crop_orientation()
    >>> input_tensor = np.random.rand((1, 3, 256, 256))
    >>> out = model(input_tensor)

    Args:
    ----
        model_path: path to onnx model file, defaults to url in default_cfgs
        **kwargs: keyword arguments of the MobileNetV3 architecture

    Returns:
    -------
        MobileNetV3
    """
    return _mobilenet_v3("mobilenet_v3_small_crop_orientation", model_path, **kwargs)


def mobilenet_v3_small_page_orientation(
    model_path: str = default_cfgs["mobilenet_v3_small_page_orientation"]["url"], **kwargs: Any
) -> MobileNetV3:
    """MobileNetV3-Small architecture as described in
    `"Searching for MobileNetV3",
    <https://arxiv.org/pdf/1905.02244.pdf>`_.

    >>> import numpy as np
    >>> from onnxtr.models import mobilenet_v3_small_page_orientation
    >>> model = mobilenet_v3_small_page_orientation()
    >>> input_tensor = np.random.rand((1, 3, 512, 512))
    >>> out = model(input_tensor)

    Args:
    ----
        model_path: path to onnx model file, defaults to url in default_cfgs
        **kwargs: keyword arguments of the MobileNetV3 architecture

    Returns:
    -------
        MobileNetV3
    """
    return _mobilenet_v3("mobilenet_v3_small_page_orientation", model_path, **kwargs)
