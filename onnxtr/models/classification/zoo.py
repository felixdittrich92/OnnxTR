# Copyright (C) 2021-2024, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, Dict, List

from .. import classification
from ..preprocessor import PreProcessor
from .predictor import OrientationPredictor

__all__ = ["crop_orientation_predictor", "page_orientation_predictor"]

ORIENTATION_ARCHS: List[str] = ["mobilenet_v3_small_crop_orientation", "mobilenet_v3_small_page_orientation"]

default_cfgs: Dict[str, Dict[str, Any]] = {
    "mobilenet_v3_small_crop_orientation": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 256, 256),
        "classes": [0, -90, 180, 90],
        "url": "https://doctr-static.mindee.com/models?id=v0.8.1/mobilenet_v3_small_crop_orientation-f0847a18.pt&src=0",  # TODO
    },
    "mobilenet_v3_small_page_orientation": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 512, 512),
        "classes": [0, -90, 180, 90],
        "url": "https://doctr-static.mindee.com/models?id=v0.8.1/mobilenet_v3_small_page_orientation-8e60325c.pt&src=0",  # TODO
    },
}


def _orientation_predictor(arch: str, **kwargs: Any) -> OrientationPredictor:
    if arch not in ORIENTATION_ARCHS:
        raise ValueError(f"unknown architecture '{arch}'")

    # Load directly classifier from backbone
    _model = classification.__dict__[arch]()
    kwargs["mean"] = kwargs.get("mean", _model.cfg["mean"])
    kwargs["std"] = kwargs.get("std", _model.cfg["std"])
    kwargs["batch_size"] = kwargs.get("batch_size", 128 if "crop" in arch else 4)
    input_shape = _model.cfg["input_shape"][1:]
    predictor = OrientationPredictor(
        PreProcessor(input_shape, preserve_aspect_ratio=True, symmetric_pad=True, **kwargs), _model
    )
    return predictor


def crop_orientation_predictor(
    arch: Any = "mobilenet_v3_small_crop_orientation", **kwargs: Any
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
        **kwargs: keyword arguments to be passed to the OrientationPredictor

    Returns:
    -------
        OrientationPredictor
    """
    return _orientation_predictor(arch, **kwargs)


def page_orientation_predictor(
    arch: Any = "mobilenet_v3_small_page_orientation", **kwargs: Any
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
        **kwargs: keyword arguments to be passed to the OrientationPredictor

    Returns:
    -------
        OrientationPredictor
    """
    return _orientation_predictor(arch, **kwargs)
