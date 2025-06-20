# Copyright (C) 2021-2025, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any

from .. import recognition
from ..engine import EngineConfig
from ..preprocessor import PreProcessor
from .predictor import RecognitionPredictor

__all__ = ["recognition_predictor"]


ARCHS: list[str] = [
    "crnn_vgg16_bn",
    "crnn_mobilenet_v3_small",
    "crnn_mobilenet_v3_large",
    "sar_resnet31",
    "master",
    "vitstr_small",
    "vitstr_base",
    "parseq",
    "viptr_tiny",
]


def _predictor(
    arch: Any, load_in_8_bit: bool = False, engine_cfg: EngineConfig | None = None, **kwargs: Any
) -> RecognitionPredictor:
    if isinstance(arch, str):
        if arch not in ARCHS:
            raise ValueError(f"unknown architecture '{arch}'")

        _model = recognition.__dict__[arch](load_in_8_bit=load_in_8_bit, engine_cfg=engine_cfg)
    else:
        if not isinstance(
            arch,
            (
                recognition.CRNN,
                recognition.SAR,
                recognition.MASTER,
                recognition.ViTSTR,
                recognition.PARSeq,
                recognition.VIPTR,
            ),
        ):
            raise ValueError(f"unknown architecture: {type(arch)}")
        _model = arch

    kwargs["mean"] = kwargs.get("mean", _model.cfg["mean"])
    kwargs["std"] = kwargs.get("std", _model.cfg["std"])
    kwargs["batch_size"] = kwargs.get("batch_size", 1024)
    input_shape = _model.cfg["input_shape"][1:]
    predictor = RecognitionPredictor(PreProcessor(input_shape, preserve_aspect_ratio=True, **kwargs), _model)

    return predictor


def recognition_predictor(
    arch: Any = "crnn_vgg16_bn",
    symmetric_pad: bool = False,
    batch_size: int = 128,
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> RecognitionPredictor:
    """Text recognition architecture.

    Example::
        >>> import numpy as np
        >>> from onnxtr.models import recognition_predictor
        >>> model = recognition_predictor()
        >>> input_page = (255 * np.random.rand(32, 128, 3)).astype(np.uint8)
        >>> out = model([input_page])

    Args:
        arch: name of the architecture or model itself to use (e.g. 'crnn_vgg16_bn')
        symmetric_pad: if True, pad the image symmetrically instead of padding at the bottom-right
        batch_size: number of samples the model processes in parallel
        load_in_8_bit: whether to load the the 8-bit quantized model, defaults to False
        engine_cfg: configuration of inference engine
        **kwargs: optional parameters to be passed to the architecture

    Returns:
        Recognition predictor
    """
    return _predictor(
        arch=arch,
        symmetric_pad=symmetric_pad,
        batch_size=batch_size,
        load_in_8_bit=load_in_8_bit,
        engine_cfg=engine_cfg,
        **kwargs,
    )
