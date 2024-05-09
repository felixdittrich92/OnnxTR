# Copyright (C) 2021-2024, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, Dict, Optional

import numpy as np
from scipy.special import expit

from ...engine import Engine
from ..postprocessor.base import GeneralDetectionPostProcessor

__all__ = ["DBNet", "db_resnet50", "db_resnet34", "db_mobilenet_v3_large"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "db_resnet50": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://github.com/felixdittrich92/OnnxTR/releases/download/v0.0.1/db_resnet50-69ba0015.onnx",
    },
    "db_resnet34": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://github.com/felixdittrich92/OnnxTR/releases/download/v0.0.1/db_resnet34-b4873198.onnx",
    },
    "db_mobilenet_v3_large": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://github.com/felixdittrich92/OnnxTR/releases/download/v0.0.1/db_mobilenet_v3_large-1866973f.onnx",
    },
}


class DBNet(Engine):
    """DBNet Onnx loader

    Args:
    ----
        model_path: path or url to onnx model file
        bin_thresh: threshold for binarization of the output feature map
        box_thresh: minimal objectness score to consider a box
        assume_straight_pages: if True, fit straight bounding boxes only
        cfg: the configuration dict of the model
    """

    def __init__(
        self,
        model_path,
        bin_thresh: float = 0.3,
        box_thresh: float = 0.1,
        assume_straight_pages: bool = True,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(url=model_path)
        self.cfg = cfg
        self.assume_straight_pages = assume_straight_pages
        self.postprocessor = GeneralDetectionPostProcessor(
            assume_straight_pages=self.assume_straight_pages, bin_thresh=bin_thresh, box_thresh=box_thresh
        )

    def __call__(
        self,
        x: np.ndarray,
        return_model_output: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        logits = self.run(x)

        out: Dict[str, Any] = {}

        prob_map = expit(logits)
        if return_model_output:
            out["out_map"] = prob_map

        out["preds"] = [dict(zip(["words"], preds)) for preds in self.postprocessor(prob_map)]

        return out


def _dbnet(
    arch: str,
    model_path: str,
    **kwargs: Any,
) -> DBNet:
    # Build the model
    return DBNet(model_path, cfg=default_cfgs[arch], **kwargs)


def db_resnet34(model_path: str = default_cfgs["db_resnet34"]["url"], **kwargs: Any) -> DBNet:
    """DBNet as described in `"Real-time Scene Text Detection with Differentiable Binarization"
    <https://arxiv.org/pdf/1911.08947.pdf>`_, using a ResNet-34 backbone.

    >>> import numpy as np
    >>> from onnxtr.models import db_resnet34
    >>> model = db_resnet34()
    >>> input_tensor = np.random.rand(1, 3, 1024, 1024)
    >>> out = model(input_tensor)

    Args:
    ----
        model_path: path to onnx model file, defaults to url in default_cfgs
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
    -------
        text detection architecture
    """
    return _dbnet("db_resnet34", model_path, **kwargs)


def db_resnet50(model_path: str = default_cfgs["db_resnet50"]["url"], **kwargs: Any) -> DBNet:
    """DBNet as described in `"Real-time Scene Text Detection with Differentiable Binarization"
    <https://arxiv.org/pdf/1911.08947.pdf>`_, using a ResNet-50 backbone.

    >>> import numpy as np
    >>> from onnxtr.models import db_resnet50
    >>> model = db_resnet50()
    >>> input_tensor = np.random.rand(1, 3, 1024, 1024)
    >>> out = model(input_tensor)

    Args:
    ----
        model_path: path to onnx model file, defaults to url in default_cfgs
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
    -------
        text detection architecture
    """
    return _dbnet("db_resnet50", model_path, **kwargs)


def db_mobilenet_v3_large(model_path: str = default_cfgs["db_mobilenet_v3_large"]["url"], **kwargs: Any) -> DBNet:
    """DBNet as described in `"Real-time Scene Text Detection with Differentiable Binarization"
    <https://arxiv.org/pdf/1911.08947.pdf>`_, using a MobileNet V3 Large backbone.

    >>> import numpy as np
    >>> from onnxtr.models import db_mobilenet_v3_large
    >>> model = db_mobilenet_v3_large()
    >>> input_tensor = np.random.rand(1, 3, 1024, 1024)
    >>> out = model(input_tensor)

    Args:
    ----
        model_path: path to onnx model file, defaults to url in default_cfgs
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
    -------
        text detection architecture
    """
    return _dbnet("db_mobilenet_v3_large", model_path, **kwargs)
