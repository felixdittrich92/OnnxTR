# Copyright (C) 2021-2025, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
from typing import Any

import numpy as np
from scipy.special import expit

from ...engine import Engine, EngineConfig
from ..postprocessor.base import GeneralDetectionPostProcessor

__all__ = ["FAST", "fast_tiny", "fast_small", "fast_base"]


default_cfgs: dict[str, dict[str, Any]] = {
    "fast_tiny": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://github.com/felixdittrich92/OnnxTR/releases/download/v0.0.1/rep_fast_tiny-28867779.onnx",
    },
    "fast_small": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://github.com/felixdittrich92/OnnxTR/releases/download/v0.0.1/rep_fast_small-10428b70.onnx",
    },
    "fast_base": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://github.com/felixdittrich92/OnnxTR/releases/download/v0.0.1/rep_fast_base-1b89ebf9.onnx",
    },
}


class FAST(Engine):
    """FAST Onnx loader

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


def _fast(
    arch: str,
    model_path: str,
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> FAST:
    if load_in_8_bit:
        logging.warning("FAST models do not support 8-bit quantization yet. Loading full precision model...")
    # Build the model
    return FAST(model_path, cfg=default_cfgs[arch], engine_cfg=engine_cfg, **kwargs)


def fast_tiny(
    model_path: str = default_cfgs["fast_tiny"]["url"],
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> FAST:
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_, using a tiny TextNet backbone.

    >>> import numpy as np
    >>> from onnxtr.models import fast_tiny
    >>> model = fast_tiny()
    >>> input_tensor = np.random.rand(1, 3, 1024, 1024)
    >>> out = model(input_tensor)

    Args:
        model_path: path to onnx model file, defaults to url in default_cfgs
        load_in_8_bit: whether to load the the 8-bit quantized model, defaults to False
        engine_cfg: configuration for the inference engine
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
        text detection architecture
    """
    return _fast("fast_tiny", model_path, load_in_8_bit, engine_cfg, **kwargs)


def fast_small(
    model_path: str = default_cfgs["fast_small"]["url"],
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> FAST:
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_, using a small TextNet backbone.

    >>> import numpy as np
    >>> from onnxtr.models import fast_small
    >>> model = fast_small()
    >>> input_tensor = np.random.rand(1, 3, 1024, 1024)
    >>> out = model(input_tensor)

    Args:
        model_path: path to onnx model file, defaults to url in default_cfgs
        load_in_8_bit: whether to load the the 8-bit quantized model, defaults to False
        engine_cfg: configuration for the inference engine
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
        text detection architecture
    """
    return _fast("fast_small", model_path, load_in_8_bit, engine_cfg, **kwargs)


def fast_base(
    model_path: str = default_cfgs["fast_base"]["url"],
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> FAST:
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_, using a base TextNet backbone.

    >>> import numpy as np
    >>> from onnxtr.models import fast_base
    >>> model = fast_base()
    >>> input_tensor = np.random.rand(1, 3, 1024, 1024)
    >>> out = model(input_tensor)

    Args:
        model_path: path to onnx model file, defaults to url in default_cfgs
        load_in_8_bit: whether to load the the 8-bit quantized model, defaults to False
        engine_cfg: configuration for the inference engine
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
        text detection architecture
    """
    return _fast("fast_base", model_path, load_in_8_bit, engine_cfg, **kwargs)
