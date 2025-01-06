# Copyright (C) 2021-2025, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any

import numpy as np
from scipy.special import expit

from ...engine import Engine, EngineConfig
from ..postprocessor.base import GeneralDetectionPostProcessor

__all__ = ["LinkNet", "linknet_resnet18", "linknet_resnet34", "linknet_resnet50"]


default_cfgs: dict[str, dict[str, Any]] = {
    "linknet_resnet18": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://github.com/felixdittrich92/OnnxTR/releases/download/v0.0.1/linknet_resnet18-e0e0b9dc.onnx",
        "url_8_bit": "https://github.com/felixdittrich92/OnnxTR/releases/download/v0.1.2/linknet_resnet18_static_8_bit-3b3a37dd.onnx",
    },
    "linknet_resnet34": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://github.com/felixdittrich92/OnnxTR/releases/download/v0.0.1/linknet_resnet34-93e39a39.onnx",
        "url_8_bit": "https://github.com/felixdittrich92/OnnxTR/releases/download/v0.1.2/linknet_resnet34_static_8_bit-2824329d.onnx",
    },
    "linknet_resnet50": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://github.com/felixdittrich92/OnnxTR/releases/download/v0.0.1/linknet_resnet50-15d8c4ec.onnx",
        "url_8_bit": "https://github.com/felixdittrich92/OnnxTR/releases/download/v0.1.2/linknet_resnet50_static_8_bit-65d6b0b8.onnx",
    },
}


class LinkNet(Engine):
    """LinkNet Onnx loader

    Args:
        model_path: path or url to onnx model file
        engine_cfg: configuration for the inference engine
        bin_thresh: threshold for binarization of the output feature map
        box_thresh: minimal objectness score to consider a box
        assume_straight_pages: if True, fit straight bounding boxes only
        cfg: the configuration dict of the model
        **kwargs: additional arguments to be passed to `Engine`
    """

    def __init__(
        self,
        model_path: str,
        engine_cfg: EngineConfig | None = None,
        bin_thresh: float = 0.1,
        box_thresh: float = 0.1,
        assume_straight_pages: bool = True,
        cfg: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(url=model_path, engine_cfg=engine_cfg, **kwargs)

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
    ) -> dict[str, Any]:
        logits = self.run(x)

        out: dict[str, Any] = {}

        prob_map = expit(logits)
        if return_model_output:
            out["out_map"] = prob_map

        out["preds"] = self.postprocessor(prob_map)

        return out


def _linknet(
    arch: str,
    model_path: str,
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> LinkNet:
    # Patch the url
    model_path = default_cfgs[arch]["url_8_bit"] if load_in_8_bit and "http" in model_path else model_path
    # Build the model
    return LinkNet(model_path, cfg=default_cfgs[arch], engine_cfg=engine_cfg, **kwargs)


def linknet_resnet18(
    model_path: str = default_cfgs["linknet_resnet18"]["url"],
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> LinkNet:
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    >>> import numpy as np
    >>> from onnxtr.models import linknet_resnet18
    >>> model = linknet_resnet18()
    >>> input_tensor = np.random.rand(1, 3, 1024, 1024)
    >>> out = model(input_tensor)

    Args:
        model_path: path to onnx model file, defaults to url in default_cfgs
        load_in_8_bit: whether to load the the 8-bit quantized model, defaults to False
        engine_cfg: configuration for the inference engine
        **kwargs: keyword arguments of the LinkNet architecture

    Returns:
        text detection architecture
    """
    return _linknet("linknet_resnet18", model_path, load_in_8_bit, engine_cfg, **kwargs)


def linknet_resnet34(
    model_path: str = default_cfgs["linknet_resnet34"]["url"],
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> LinkNet:
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    >>> import numpy as np
    >>> from onnxtr.models import linknet_resnet34
    >>> model = linknet_resnet34()
    >>> input_tensor = np.random.rand(1, 3, 1024, 1024)
    >>> out = model(input_tensor)

    Args:
        model_path: path to onnx model file, defaults to url in default_cfgs
        load_in_8_bit: whether to load the the 8-bit quantized model, defaults to False
        engine_cfg: configuration for the inference engine
        **kwargs: keyword arguments of the LinkNet architecture

    Returns:
        text detection architecture
    """
    return _linknet("linknet_resnet34", model_path, load_in_8_bit, engine_cfg, **kwargs)


def linknet_resnet50(
    model_path: str = default_cfgs["linknet_resnet50"]["url"],
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> LinkNet:
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    >>> import numpy as np
    >>> from onnxtr.models import linknet_resnet50
    >>> model = linknet_resnet50()
    >>> input_tensor = np.random.rand(1, 3, 1024, 1024)
    >>> out = model(input_tensor)

    Args:
        model_path: path to onnx model file, defaults to url in default_cfgs
        load_in_8_bit: whether to load the the 8-bit quantized model, defaults to False
        engine_cfg: configuration for the inference engine
        **kwargs: keyword arguments of the LinkNet architecture

    Returns:
        text detection architecture
    """
    return _linknet("linknet_resnet50", model_path, load_in_8_bit, engine_cfg, **kwargs)
