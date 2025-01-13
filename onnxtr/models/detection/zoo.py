# Copyright (C) 2021-2025, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any

from .. import detection
from ..engine import EngineConfig
from ..preprocessor import PreProcessor
from .predictor import DetectionPredictor

__all__ = ["detection_predictor"]

ARCHS = [
    "db_resnet34",
    "db_resnet50",
    "db_mobilenet_v3_large",
    "linknet_resnet18",
    "linknet_resnet34",
    "linknet_resnet50",
    "fast_tiny",
    "fast_small",
    "fast_base",
]


def _predictor(
    arch: Any,
    assume_straight_pages: bool = True,
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> DetectionPredictor:
    if isinstance(arch, str):
        if arch not in ARCHS:
            raise ValueError(f"unknown architecture '{arch}'")

        _model = detection.__dict__[arch](
            assume_straight_pages=assume_straight_pages, load_in_8_bit=load_in_8_bit, engine_cfg=engine_cfg
        )
    else:
        if not isinstance(arch, (detection.DBNet, detection.LinkNet, detection.FAST)):
            raise ValueError(f"unknown architecture: {type(arch)}")

        _model = arch
        _model.assume_straight_pages = assume_straight_pages
        _model.postprocessor.assume_straight_pages = assume_straight_pages

    kwargs["mean"] = kwargs.get("mean", _model.cfg["mean"])
    kwargs["std"] = kwargs.get("std", _model.cfg["std"])
    kwargs["batch_size"] = kwargs.get("batch_size", 2)
    predictor = DetectionPredictor(
        PreProcessor(_model.cfg["input_shape"][1:], **kwargs),
        _model,
    )
    return predictor


def detection_predictor(
    arch: Any = "fast_base",
    assume_straight_pages: bool = True,
    preserve_aspect_ratio: bool = True,
    symmetric_pad: bool = True,
    batch_size: int = 2,
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> DetectionPredictor:
    """Text detection architecture.

    >>> import numpy as np
    >>> from onnxtr.models import detection_predictor
    >>> model = detection_predictor(arch='db_resnet50')
    >>> input_page = (255 * np.random.rand(600, 800, 3)).astype(np.uint8)
    >>> out = model([input_page])

    Args:
        arch: name of the architecture or model itself to use (e.g. 'db_resnet50')
        assume_straight_pages: If True, fit straight boxes to the page
        preserve_aspect_ratio: If True, pad the input document image to preserve the aspect ratio before
            running the detection model on it
        symmetric_pad: if True, pad the image symmetrically instead of padding at the bottom-right
        batch_size: number of samples the model processes in parallel
        load_in_8_bit: whether to load the the 8-bit quantized model, defaults to False
        engine_cfg: configuration for the inference engine
        **kwargs: optional keyword arguments passed to the architecture

    Returns:
        Detection predictor
    """
    return _predictor(
        arch=arch,
        assume_straight_pages=assume_straight_pages,
        preserve_aspect_ratio=preserve_aspect_ratio,
        symmetric_pad=symmetric_pad,
        batch_size=batch_size,
        load_in_8_bit=load_in_8_bit,
        engine_cfg=engine_cfg,
        **kwargs,
    )
