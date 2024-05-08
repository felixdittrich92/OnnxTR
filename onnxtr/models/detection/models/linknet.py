# Copyright (C) 2021-2024, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, Dict, Optional

import numpy as np
from scipy.special import expit

from ...engine import Engine
from ..postprocessor.base import GeneralDetectionPostProcessor

__all__ = ["LinkNet", "linknet_resnet18", "linknet_resnet34", "linknet_resnet50"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "linknet_resnet18": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://doctr-static.mindee.com/models?id=v0.7.0/linknet_resnet18-e47a14dc.pt&src=0",
    },
    "linknet_resnet34": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://doctr-static.mindee.com/models?id=v0.7.0/linknet_resnet34-9ca2df3e.pt&src=0",
    },
    "linknet_resnet50": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://doctr-static.mindee.com/models?id=v0.7.0/linknet_resnet50-6cf565c1.pt&src=0",
    },
}


class LinkNet(Engine):
    """LinkNet Onnx loader

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


def _linknet(
    arch: str,
    model_path: str,
    **kwargs: Any,
) -> LinkNet:
    # Build the model
    return LinkNet(model_path, cfg=default_cfgs[arch], **kwargs)


def linknet_resnet18(model_path: str = default_cfgs["linknet_resnet18"], **kwargs: Any) -> LinkNet:
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    >>> import numpy as np
    >>> from onnxtr.models import linknet_resnet18
    >>> model = linknet_resnet18()
    >>> input_tensor = np.random.rand(1, 3, 1024, 1024)
    >>> out = model(input_tensor)

    Args:
    ----
        model_path: path to onnx model file, defaults to url in default_cfgs
        **kwargs: keyword arguments of the LinkNet architecture

    Returns:
    -------
        text detection architecture
    """
    return _linknet("linknet_resnet18", model_path, **kwargs)


def linknet_resnet34(model_path: str = default_cfgs["linknet_resnet34"], **kwargs: Any) -> LinkNet:
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    >>> import numpy as np
    >>> from onnxtr.models import linknet_resnet34
    >>> model = linknet_resnet34()
    >>> input_tensor = np.random.rand(1, 3, 1024, 1024)
    >>> out = model(input_tensor)

    Args:
    ----
        model_path: path to onnx model file, defaults to url in default_cfgs
        **kwargs: keyword arguments of the LinkNet architecture

    Returns:
    -------
        text detection architecture
    """
    return _linknet("linknet_resnet34", model_path, **kwargs)


def linknet_resnet50(model_path: str = default_cfgs["linknet_resnet50"], **kwargs: Any) -> LinkNet:
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    >>> import numpy as np
    >>> from onnxtr.models import linknet_resnet50
    >>> model = linknet_resnet50()
    >>> input_tensor = np.random.rand(1, 3, 1024, 1024)
    >>> out = model(input_tensor)

    Args:
    ----
        model_path: path to onnx model file, defaults to url in default_cfgs
        **kwargs: keyword arguments of the LinkNet architecture

    Returns:
    -------
        text detection architecture
    """
    return _linknet("linknet_resnet50", model_path, **kwargs)
