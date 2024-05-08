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
        "url": "https://doctr-static.mindee.com/models?id=v0.7.0/db_resnet50-79bd7d70.pt&src=0",
    },
    "db_resnet34": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://doctr-static.mindee.com/models?id=v0.7.0/db_resnet34-cb6aed9e.pt&src=0",
    },
    "db_mobilenet_v3_large": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://doctr-static.mindee.com/models?id=v0.7.0/db_mobilenet_v3_large-81e9b152.pt&src=0",
    },
}


class DBNet(Engine):
    """DBNet Onnx loader

    Args:
    ----
        bin_thresh: threshold for binarization of the output feature map
        box_thresh: minimal objectness score to consider a box
        assume_straight_pages: if True, fit straight bounding boxes only
        cfg: the configuration dict of the model
    """

    def __init__(
        self,
        bin_thresh: float = 0.1,
        box_thresh: float = 0.1,
        assume_straight_pages: bool = True,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
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
        logits = self.session.run(x)

        out: Dict[str, Any] = {}

        prob_map = expit(logits)
        if return_model_output:
            out["out_map"] = prob_map

        out["preds"] = [
            dict(zip(self.class_names, preds)) for preds in self.postprocessor(np.transpose(prob_map, (0, 2, 3, 1)))
        ]

        return out


def _dbnet(
    arch: str,
    model_path: str,
    **kwargs: Any,
) -> DBNet:
    # Build the model
    return DBNet(model_path, cfg=default_cfgs[arch], **kwargs)


def db_resnet34(model_path: str = default_cfgs["db_resnet34"], **kwargs: Any) -> DBNet:
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


def db_resnet50(model_path: str = default_cfgs["db_resnet50"], **kwargs: Any) -> DBNet:
    """DBNet as described in `"Real-time Scene Text Detection with Differentiable Binarization"
    <https://arxiv.org/pdf/1911.08947.pdf>`_, using a ResNet-50 backbone.

    >>> import numpy as np
    >>> from onnxtr.models import db_resnet50
    >>> model = db_resnet50(pretrained=True)
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


def db_mobilenet_v3_large(model_path: str = default_cfgs["db_mobilenet_v3_large"], **kwargs: Any) -> DBNet:
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
    return _dbnet("db_mobilenet_v3_large", model_path**kwargs)
