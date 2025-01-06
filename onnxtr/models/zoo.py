# Copyright (C) 2021-2025, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any

from .detection.zoo import detection_predictor
from .engine import EngineConfig
from .predictor import OCRPredictor
from .recognition.zoo import recognition_predictor

__all__ = ["ocr_predictor"]


def _predictor(
    det_arch: Any,
    reco_arch: Any,
    assume_straight_pages: bool = True,
    preserve_aspect_ratio: bool = True,
    symmetric_pad: bool = True,
    det_bs: int = 2,
    reco_bs: int = 512,
    detect_orientation: bool = False,
    straighten_pages: bool = False,
    detect_language: bool = False,
    load_in_8_bit: bool = False,
    det_engine_cfg: EngineConfig | None = None,
    reco_engine_cfg: EngineConfig | None = None,
    clf_engine_cfg: EngineConfig | None = None,
    **kwargs,
) -> OCRPredictor:
    # Detection
    det_predictor = detection_predictor(
        det_arch,
        batch_size=det_bs,
        assume_straight_pages=assume_straight_pages,
        preserve_aspect_ratio=preserve_aspect_ratio,
        symmetric_pad=symmetric_pad,
        load_in_8_bit=load_in_8_bit,
        engine_cfg=det_engine_cfg,
    )

    # Recognition
    reco_predictor = recognition_predictor(
        reco_arch,
        batch_size=reco_bs,
        load_in_8_bit=load_in_8_bit,
        engine_cfg=reco_engine_cfg,
    )

    return OCRPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=assume_straight_pages,
        preserve_aspect_ratio=preserve_aspect_ratio,
        symmetric_pad=symmetric_pad,
        detect_orientation=detect_orientation,
        straighten_pages=straighten_pages,
        detect_language=detect_language,
        clf_engine_cfg=clf_engine_cfg,
        **kwargs,
    )


def ocr_predictor(
    det_arch: Any = "fast_base",
    reco_arch: Any = "crnn_vgg16_bn",
    assume_straight_pages: bool = True,
    preserve_aspect_ratio: bool = True,
    symmetric_pad: bool = True,
    export_as_straight_boxes: bool = False,
    detect_orientation: bool = False,
    straighten_pages: bool = False,
    detect_language: bool = False,
    load_in_8_bit: bool = False,
    det_engine_cfg: EngineConfig | None = None,
    reco_engine_cfg: EngineConfig | None = None,
    clf_engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> OCRPredictor:
    """End-to-end OCR architecture using one model for localization, and another for text recognition.

    >>> import numpy as np
    >>> from onnxtr.models import ocr_predictor
    >>> model = ocr_predictor('db_resnet50', 'crnn_vgg16_bn')
    >>> input_page = (255 * np.random.rand(600, 800, 3)).astype(np.uint8)
    >>> out = model([input_page])

    Args:
        det_arch: name of the detection architecture or the model itself to use
            (e.g. 'db_resnet50', 'db_mobilenet_v3_large')
        reco_arch: name of the recognition architecture or the model itself to use
            (e.g. 'crnn_vgg16_bn', 'sar_resnet31')
        assume_straight_pages: if True, speeds up the inference by assuming you only pass straight pages
            without rotated textual elements.
        preserve_aspect_ratio: If True, pad the input document image to preserve the aspect ratio before
            running the detection model on it.
        symmetric_pad: if True, pad the image symmetrically instead of padding at the bottom-right.
        export_as_straight_boxes: when assume_straight_pages is set to False, export final predictions
            (potentially rotated) as straight bounding boxes.
        detect_orientation: if True, the estimated general page orientation will be added to the predictions for each
            page. Doing so will slightly deteriorate the overall latency.
        straighten_pages: if True, estimates the page general orientation
            based on the segmentation map median line orientation.
            Then, rotates page before passing it again to the deep learning detection module.
            Doing so will improve performances for documents with page-uniform rotations.
        detect_language: if True, the language prediction will be added to the predictions for each
            page. Doing so will slightly deteriorate the overall latency.
        load_in_8_bit: whether to load the the 8-bit quantized model, defaults to False
        det_engine_cfg: configuration of the detection engine
        reco_engine_cfg: configuration of the recognition engine
        clf_engine_cfg: configuration of the orientation classification engine
        kwargs: keyword args of `OCRPredictor`

    Returns:
        OCR predictor
    """
    return _predictor(
        det_arch,
        reco_arch,
        assume_straight_pages=assume_straight_pages,
        preserve_aspect_ratio=preserve_aspect_ratio,
        symmetric_pad=symmetric_pad,
        export_as_straight_boxes=export_as_straight_boxes,
        detect_orientation=detect_orientation,
        straighten_pages=straighten_pages,
        detect_language=detect_language,
        load_in_8_bit=load_in_8_bit,
        det_engine_cfg=det_engine_cfg,
        reco_engine_cfg=reco_engine_cfg,
        clf_engine_cfg=clf_engine_cfg,
        **kwargs,
    )
