# Copyright (C) 2021-2026, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any

from .. import layout
from ..engine import EngineConfig
from ..preprocessor import PreProcessor
from .predictor import LayoutPredictor

__all__ = ["layout_predictor"]

ARCHS: list[str] = ["lw_detr_s"]  # , "lw_detr_m"] --- IGNORE ---


def _predictor(
    arch: Any,
    assume_straight_pages: bool = True,
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> LayoutPredictor:
    if isinstance(arch, str):
        if arch not in ARCHS:
            raise ValueError(f"unknown architecture '{arch}'")

        _model = layout.__dict__[arch](
            assume_straight_pages=assume_straight_pages, load_in_8_bit=load_in_8_bit, engine_cfg=engine_cfg
        )
    else:
        if not isinstance(arch, layout.LWDETR):
            raise ValueError(f"unknown architecture: {type(arch)}")

        _model = arch
        _model.assume_straight_pages = assume_straight_pages
        _model.postprocessor.assume_straight_pages = assume_straight_pages

    kwargs["mean"] = kwargs.get("mean", _model.cfg["mean"])
    kwargs["std"] = kwargs.get("std", _model.cfg["std"])
    kwargs["batch_size"] = kwargs.get("batch_size", 2)
    # The model consumes an attention mask built from the padding
    kwargs["return_padding_mask"] = True
    predictor = LayoutPredictor(
        PreProcessor(_model.cfg["input_shape"][1:], **kwargs),
        _model,
    )
    return predictor


def layout_predictor(
    arch: Any = "lw_detr_s",
    assume_straight_pages: bool = True,
    preserve_aspect_ratio: bool = True,
    symmetric_pad: bool = True,
    batch_size: int = 2,
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> LayoutPredictor:
    """Layout prediction architecture.

    >>> import numpy as np
    >>> from onnxtr.models import layout_predictor
    >>> model = layout_predictor(arch='lw_detr_s')
    >>> input_page = (255 * np.random.rand(600, 800, 3)).astype(np.uint8)
    >>> out = model([input_page])

    Args:
        arch: name of the architecture or model itself to use (e.g. 'lw_detr_s')
        assume_straight_pages: If True, fit straight boxes to the page
        preserve_aspect_ratio: If True, pad the input document image to preserve the aspect ratio before
            running the layout model on it
        symmetric_pad: if True, pad the image symmetrically instead of padding at the bottom-right
        batch_size: number of samples the model processes in parallel
        load_in_8_bit: whether to load the the 8-bit quantized model, defaults to False
        engine_cfg: configuration for the inference engine
        **kwargs: optional keyword arguments passed to the architecture

    Returns:
        Layout predictor
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
