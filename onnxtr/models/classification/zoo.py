# Copyright (C) 2021-2025, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any

from onnxtr.models.engine import EngineConfig

from .. import classification
from ..preprocessor import PreProcessor
from .predictor import OrientationPredictor

__all__ = ["crop_orientation_predictor", "page_orientation_predictor"]

ORIENTATION_ARCHS: list[str] = ["mobilenet_v3_small_crop_orientation", "mobilenet_v3_small_page_orientation"]


def _orientation_predictor(
    arch: Any,
    model_type: str,
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    disabled: bool = False,
    **kwargs: Any,
) -> OrientationPredictor:
    if disabled:
        # Case where the orientation predictor is disabled
        return OrientationPredictor(None, None)

    if isinstance(arch, str):
        if arch not in ORIENTATION_ARCHS:
            raise ValueError(f"unknown architecture '{arch}'")
        # Load directly classifier from backbone
        _model = classification.__dict__[arch](load_in_8_bit=load_in_8_bit, engine_cfg=engine_cfg)
    else:
        if not isinstance(arch, classification.MobileNetV3):
            raise ValueError(f"unknown architecture: {type(arch)}")
        _model = arch

    kwargs["mean"] = kwargs.get("mean", _model.cfg["mean"])
    kwargs["std"] = kwargs.get("std", _model.cfg["std"])
    kwargs["batch_size"] = kwargs.get("batch_size", 512 if model_type == "crop" else 2)
    input_shape = _model.cfg["input_shape"][1:]
    predictor = OrientationPredictor(
        PreProcessor(input_shape, preserve_aspect_ratio=True, symmetric_pad=True, **kwargs),
        _model,
    )
    return predictor


def crop_orientation_predictor(
    arch: Any = "mobilenet_v3_small_crop_orientation",
    batch_size: int = 512,
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> OrientationPredictor:
    """Crop orientation classification architecture.

    >>> import numpy as np
    >>> from onnxtr.models import crop_orientation_predictor
    >>> model = crop_orientation_predictor(arch='mobilenet_v3_small_crop_orientation')
    >>> input_crop = (255 * np.random.rand(256, 256, 3)).astype(np.uint8)
    >>> out = model([input_crop])

    Args:
        arch: name of the architecture to use (e.g. 'mobilenet_v3_small_crop_orientation')
        batch_size: number of samples the model processes in parallel
        load_in_8_bit: load the 8-bit quantized version of the model
        engine_cfg: configuration of inference engine
        **kwargs: keyword arguments to be passed to the OrientationPredictor

    Returns:
        OrientationPredictor
    """
    model_type = "crop"
    return _orientation_predictor(
        arch=arch,
        batch_size=batch_size,
        model_type=model_type,
        load_in_8_bit=load_in_8_bit,
        engine_cfg=engine_cfg,
        **kwargs,
    )


def page_orientation_predictor(
    arch: Any = "mobilenet_v3_small_page_orientation",
    batch_size: int = 2,
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> OrientationPredictor:
    """Page orientation classification architecture.

    >>> import numpy as np
    >>> from onnxtr.models import page_orientation_predictor
    >>> model = page_orientation_predictor(arch='mobilenet_v3_small_page_orientation')
    >>> input_page = (255 * np.random.rand(512, 512, 3)).astype(np.uint8)
    >>> out = model([input_page])

    Args:
        arch: name of the architecture to use (e.g. 'mobilenet_v3_small_page_orientation')
        batch_size: number of samples the model processes in parallel
        load_in_8_bit: whether to load the the 8-bit quantized model, defaults to False
        engine_cfg: configuration for the inference engine
        **kwargs: keyword arguments to be passed to the OrientationPredictor

    Returns:
        OrientationPredictor
    """
    model_type = "page"
    return _orientation_predictor(
        arch=arch,
        batch_size=batch_size,
        model_type=model_type,
        load_in_8_bit=load_in_8_bit,
        engine_cfg=engine_cfg,
        **kwargs,
    )
