# Copyright (C) 2021-2024, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, List

from .. import classification
from ..preprocessor import PreProcessor
from .predictor import OrientationPredictor

__all__ = ["crop_orientation_predictor", "page_orientation_predictor"]

ORIENTATION_ARCHS: List[str] = ["mobilenet_v3_small_crop_orientation", "mobilenet_v3_small_page_orientation"]


def _orientation_predictor(arch: str, load_in_8_bit: bool = False, **kwargs: Any) -> OrientationPredictor:
    if arch not in ORIENTATION_ARCHS:
        raise ValueError(f"unknown architecture '{arch}'")

    # Load directly classifier from backbone
    _model = classification.__dict__[arch](load_in_8_bit=load_in_8_bit)
    kwargs["mean"] = kwargs.get("mean", _model.cfg["mean"])
    kwargs["std"] = kwargs.get("std", _model.cfg["std"])
    kwargs["batch_size"] = kwargs.get("batch_size", 128 if "crop" in arch else 4)
    input_shape = _model.cfg["input_shape"][1:]
    predictor = OrientationPredictor(
        PreProcessor(input_shape, preserve_aspect_ratio=True, symmetric_pad=True, **kwargs),
        _model,
    )
    return predictor


def crop_orientation_predictor(
    arch: Any = "mobilenet_v3_small_crop_orientation", load_in_8_bit: bool = False, **kwargs: Any
) -> OrientationPredictor:
    """Crop orientation classification architecture.

    >>> import numpy as np
    >>> from onnxtr.models import crop_orientation_predictor
    >>> model = crop_orientation_predictor(arch='mobilenet_v3_small_crop_orientation')
    >>> input_crop = (255 * np.random.rand(256, 256, 3)).astype(np.uint8)
    >>> out = model([input_crop])

    Args:
    ----
        arch: name of the architecture to use (e.g. 'mobilenet_v3_small_crop_orientation')
        load_in_8_bit: load the 8-bit quantized version of the model
        **kwargs: keyword arguments to be passed to the OrientationPredictor

    Returns:
    -------
        OrientationPredictor
    """
    return _orientation_predictor(arch, load_in_8_bit, **kwargs)


def page_orientation_predictor(
    arch: Any = "mobilenet_v3_small_page_orientation", load_in_8_bit: bool = False, **kwargs: Any
) -> OrientationPredictor:
    """Page orientation classification architecture.

    >>> import numpy as np
    >>> from onnxtr.models import page_orientation_predictor
    >>> model = page_orientation_predictor(arch='mobilenet_v3_small_page_orientation')
    >>> input_page = (255 * np.random.rand(512, 512, 3)).astype(np.uint8)
    >>> out = model([input_page])

    Args:
    ----
        arch: name of the architecture to use (e.g. 'mobilenet_v3_small_page_orientation')
        load_in_8_bit: whether to load the the 8-bit quantized model, defaults to False
        **kwargs: keyword arguments to be passed to the OrientationPredictor

    Returns:
    -------
        OrientationPredictor
    """
    return _orientation_predictor(arch, load_in_8_bit, **kwargs)
