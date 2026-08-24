# Copyright (C) 2021-2026, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
from typing import Any

import numpy as np

from onnxtr.utils.geometry import shape_translate

from ...engine import Engine, EngineConfig
from ..postprocessor.base import LWDETRPostProcessor

__all__ = ["LWDETR", "lw_detr_s"]  # , "lw_detr_m"] --- IGNORE ---

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
]

default_cfgs: dict[str, dict[str, Any]] = {
    "lw_detr_s": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "class_names": CLASS_NAMES,
        "url": "https://github.com/felixdittrich92/OnnxTR/releases/download/v0.8.1/lw_detr_s-e2df565e.onnx",
        "url_8_bit": None,
    },
    "lw_detr_m": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "class_names": CLASS_NAMES,
        "url": None,
        "url_8_bit": None,
    },
}


class LWDETR(Engine):
    """LW-DETR Onnx loader

    Args:
        model_path: path or url to onnx model file
        engine_cfg: configuration for the inference engine
        class_names: list of class names the model was trained on
        score_thresh: confidence threshold for filtering predictions
        iou_thresh: IoU threshold for the class-wise NMS
        topk: number of top (query, class) pairs kept before NMS
        assume_straight_pages: if True, fit straight bounding boxes only
        cfg: the configuration dict of the model
        **kwargs: additional arguments to be passed to `Engine`
    """

    def __init__(
        self,
        model_path: str,
        engine_cfg: EngineConfig | None = None,
        class_names: list[str] | None = None,
        score_thresh: float = 0.5,
        iou_thresh: float = 0.5,
        topk: int = 300,
        assume_straight_pages: bool = True,
        cfg: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(url=model_path, engine_cfg=engine_cfg, **kwargs)

        self.cfg = cfg
        self.class_names = class_names or (cfg or {}).get("class_names", CLASS_NAMES)
        self.assume_straight_pages = assume_straight_pages

        self.postprocessor = LWDETRPostProcessor(
            num_classes=len(self.class_names),
            score_thresh=score_thresh,
            iou_thresh=iou_thresh,
            topk=topk,
            assume_straight_pages=assume_straight_pages,
        )

    def __call__(
        self,
        x: np.ndarray,
        mask: np.ndarray | None = None,
        return_model_output: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run the model on a batch of pages

        Args:
            x: batched and normalized pages of shape (N, C, H, W) or (N, H, W, C)
            mask: boolean padding mask of shape (N, H, W), True on valid (non-padded) pixels
            return_model_output: whether to return the raw logits and boxes
            **kwargs: unused, kept for API consistency

        Returns:
            dict with the postprocessed predictions (and optionally the raw model output)
        """
        # `run_multi` does not translate the layout, so do it here
        x = shape_translate(x, format="BHWC" if self.tf_exported else "BCHW")
        inputs: dict[str, np.ndarray] = {self.input_names[0]: x}
        if len(self.input_names) > 1:
            if mask is None:  # pragma: no cover
                # No mask provided: consider every pixel as valid image content
                mask = np.ones((x.shape[0], *x.shape[-2:]), dtype=bool)
            inputs[self.input_names[1]] = mask

        outputs = self.run_multi(inputs)
        # Prefer name based lookup, fall back to the export order (logits, pred_boxes)
        logits = outputs.get("logits", outputs[self.output_name[0]])
        pred_boxes = outputs.get("pred_boxes", outputs[self.output_name[-1]])

        out: dict[str, Any] = {}

        if return_model_output:
            out["logits"] = logits
            out["pred_boxes"] = pred_boxes

        out["preds"] = self.postprocessor(logits, pred_boxes)

        return out


def _lw_detr(
    arch: str,
    model_path: str,
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> LWDETR:
    if model_path is None:  # pragma: no cover
        raise ValueError(f"no pretrained ONNX export is available for '{arch}' yet")
    if load_in_8_bit:
        if default_cfgs[arch]["url_8_bit"] is None:
            logger.warning(f"No 8-bit quantized export available for '{arch}'. Loading full precision model...")
        elif "http" in model_path:
            model_path = default_cfgs[arch]["url_8_bit"]
    # Build the model
    return LWDETR(model_path, cfg=default_cfgs[arch], engine_cfg=engine_cfg, **kwargs)


def lw_detr_s(
    model_path: str = default_cfgs["lw_detr_s"]["url"],
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> LWDETR:
    """LW-DETR as described in `"LW-DETR: A Transformer Replacement to YOLO for Real-Time Detection"
    <https://arxiv.org/pdf/2406.03459v1>`_, using a small ViT-Det backbone.

    >>> import numpy as np
    >>> from onnxtr.models import lw_detr_s
    >>> model = lw_detr_s()
    >>> input_tensor = np.random.rand(1, 3, 1024, 1024)
    >>> mask = np.ones((1, 1024, 1024), dtype=bool)
    >>> out = model(input_tensor, mask)

    Args:
        model_path: path to onnx model file, defaults to url in default_cfgs
        load_in_8_bit: whether to load the the 8-bit quantized model, defaults to False
        engine_cfg: configuration for the inference engine
        **kwargs: keyword arguments of the LWDETR architecture

    Returns:
        layout detection architecture
    """
    return _lw_detr("lw_detr_s", model_path, load_in_8_bit, engine_cfg, **kwargs)


# NOTE: Model is not yet available, so the function is commented out to avoid confusion.
# Uncomment and implement when the model is released.

# def lw_detr_m(
#    model_path: str = default_cfgs["lw_detr_m"]["url"],
#    load_in_8_bit: bool = False,
#    engine_cfg: EngineConfig | None = None,
#    **kwargs: Any,
# ) -> LWDETR:
#    """LW-DETR as described in `"LW-DETR: A Transformer Replacement to YOLO for Real-Time Detection"
#    <https://arxiv.org/pdf/2406.03459v1>`_, using a medium ViT-Det backbone.

#    >>> import numpy as np
#    >>> from onnxtr.models import lw_detr_m
#    >>> model = lw_detr_m()
#    >>> input_tensor = np.random.rand(1, 3, 1024, 1024)
#    >>> mask = np.ones((1, 1024, 1024), dtype=bool)
#    >>> out = model(input_tensor, mask)

#    Args:
#        model_path: path to onnx model file, defaults to url in default_cfgs
#        load_in_8_bit: whether to load the the 8-bit quantized model, defaults to False
#        engine_cfg: configuration for the inference engine
#        **kwargs: keyword arguments of the LWDETR architecture

#    Returns:
#        layout detection architecture
#    """
#    return _lw_detr("lw_detr_m", model_path, load_in_8_bit, engine_cfg, **kwargs)
