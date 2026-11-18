import logging
import pickle
import threading
import time

import cv2
import numpy as np
import pytest

from onnxtr import models
from onnxtr.io import ConcatSource, Document, DocumentFile, FigureEncoder, PageDecodeError, crop_layout_region
from onnxtr.io.elements import LayoutElement, Table
from onnxtr.models import detection, recognition
from onnxtr.models.classification import mobilenet_v3_small_crop_orientation, mobilenet_v3_small_page_orientation
from onnxtr.models.classification.zoo import crop_orientation_predictor, page_orientation_predictor
from onnxtr.models.detection.predictor import DetectionPredictor
from onnxtr.models.detection.zoo import ARCHS as DET_ARCHS
from onnxtr.models.detection.zoo import detection_predictor
from onnxtr.models.formula.zoo import ARCHS as FORMULA_ARCHS
from onnxtr.models.layout.predictor import LayoutPredictor
from onnxtr.models.layout.zoo import ARCHS as LAYOUT_ARCHS
from onnxtr.models.layout.zoo import layout_predictor
from onnxtr.models.predictor import OCRPredictor
from onnxtr.models.preprocessor import PreProcessor
from onnxtr.models.recognition.predictor import RecognitionPredictor
from onnxtr.models.recognition.zoo import ARCHS as RECO_ARCHS
from onnxtr.models.recognition.zoo import recognition_predictor
from onnxtr.models.table_structure.predictor import TablePredictor
from onnxtr.models.table_structure.zoo import ARCHS as TABLE_ARCHS
from onnxtr.models.table_structure.zoo import table_predictor
from onnxtr.models.zoo import ocr_predictor
from onnxtr.utils.repr import NestedObject


# Create a dummy callback
class _DummyCallback:
    def __call__(self, loc_preds):
        return loc_preds


@pytest.mark.parametrize(
    "assume_straight_pages, straighten_pages, disable_page_orientation, disable_crop_orientation",
    [
        [True, False, False, False],
        [False, False, True, True],
        [True, True, False, False],
        [False, True, True, True],
        [True, False, True, False],
    ],
)
def test_ocrpredictor(
    mock_pdf, assume_straight_pages, straighten_pages, disable_page_orientation, disable_crop_orientation
):
    det_bsize = 4
    det_predictor = DetectionPredictor(
        PreProcessor(output_size=(1024, 1024), batch_size=det_bsize),
        detection.db_mobilenet_v3_large(assume_straight_pages=assume_straight_pages),
    )

    reco_bsize = 16
    reco_predictor = RecognitionPredictor(
        PreProcessor(output_size=(32, 128), batch_size=reco_bsize, preserve_aspect_ratio=True),
        recognition.crnn_vgg16_bn(),
    )

    doc = DocumentFile.from_pdf(mock_pdf)

    predictor = OCRPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=assume_straight_pages,
        straighten_pages=straighten_pages,
        detect_orientation=True,
        detect_language=True,
        resolve_lines=True,
        resolve_blocks=True,
        disable_page_orientation=disable_page_orientation,
        disable_crop_orientation=disable_crop_orientation,
    )

    assert (
        predictor._page_orientation_disabled if disable_page_orientation else not predictor._page_orientation_disabled
    )
    assert (
        predictor._crop_orientation_disabled if disable_crop_orientation else not predictor._crop_orientation_disabled
    )

    if assume_straight_pages:
        assert predictor.crop_orientation_predictor is None
        if predictor.detect_orientation or predictor.straighten_pages:
            assert isinstance(predictor.page_orientation_predictor, NestedObject)
        else:
            assert predictor.page_orientation_predictor is None
    else:
        assert isinstance(predictor.crop_orientation_predictor, NestedObject)
        assert isinstance(predictor.page_orientation_predictor, NestedObject)

    out = predictor(doc)
    assert isinstance(out, Document)
    assert len(out.pages) == 2
    # Dimension check
    with pytest.raises(ValueError):
        input_page = (255 * np.random.rand(1, 256, 512, 3)).astype(np.uint8)
        _ = predictor([input_page])

    assert out.pages[0].orientation["value"] in range(-2, 3)
    assert isinstance(out.pages[0].language["value"], str)
    assert isinstance(out.render(), str)
    assert isinstance(out.pages[0].render(), str)
    assert isinstance(out.export(), dict)
    assert isinstance(out.pages[0].export(), dict)

    with pytest.raises(ValueError):
        _ = ocr_predictor("unknown_arch")

    # Test with custom orientation models
    custom_crop_orientation_model = mobilenet_v3_small_crop_orientation()
    custom_page_orientation_model = mobilenet_v3_small_page_orientation()

    if assume_straight_pages:
        if predictor.detect_orientation or predictor.straighten_pages:
            # Overwrite the default orientation models
            predictor.crop_orientation_predictor = crop_orientation_predictor(custom_crop_orientation_model)
            predictor.page_orientation_predictor = page_orientation_predictor(custom_page_orientation_model)
    else:
        # Overwrite the default orientation models
        predictor.crop_orientation_predictor = crop_orientation_predictor(custom_crop_orientation_model)
        predictor.page_orientation_predictor = page_orientation_predictor(custom_page_orientation_model)

    out = predictor(doc)
    orientation = 0
    assert out.pages[0].orientation["value"] == orientation


def test_trained_ocr_predictor(mock_payslip):
    doc = DocumentFile.from_images(mock_payslip)

    det_predictor = detection_predictor(
        "db_resnet50",
        batch_size=2,
        assume_straight_pages=True,
        symmetric_pad=True,
        preserve_aspect_ratio=False,
    )
    reco_predictor = recognition_predictor("crnn_vgg16_bn", batch_size=128)

    predictor = OCRPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=True,
        straighten_pages=True,
        preserve_aspect_ratio=False,
        resolve_lines=True,
        resolve_blocks=True,
    )
    # test hooks
    predictor.add_hook(_DummyCallback())

    out = predictor(doc)

    assert out.pages[0].blocks[0].lines[0].words[0].value == "Mr."
    geometry_mr = np.array([[0.1083984375, 0.0634765625], [0.1494140625, 0.0859375]])
    assert np.allclose(np.array(out.pages[0].blocks[0].lines[0].words[0].geometry), geometry_mr, rtol=0.05)

    assert out.pages[0].blocks[1].lines[0].words[-1].value == "revised"
    geometry_revised = np.array([[0.7548828125, 0.126953125], [0.8388671875, 0.1484375]])
    assert np.allclose(np.array(out.pages[0].blocks[1].lines[0].words[-1].geometry), geometry_revised, rtol=0.05)

    det_predictor = detection_predictor(
        "db_resnet50",
        batch_size=2,
        assume_straight_pages=True,
        preserve_aspect_ratio=True,
        symmetric_pad=True,
    )

    predictor = OCRPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=True,
        straighten_pages=True,
        preserve_aspect_ratio=True,
        symmetric_pad=True,
        resolve_lines=True,
        resolve_blocks=True,
    )

    out = predictor(doc)

    assert "Mr" in out.pages[0].blocks[0].lines[0].words[0].value

    # test list archs
    archs = predictor.list_archs()
    assert isinstance(archs, dict)
    assert archs["recognition_archs"] == RECO_ARCHS
    assert archs["detection_archs"] == DET_ARCHS
    assert archs["layout_archs"] == LAYOUT_ARCHS
    assert archs["table_structure_archs"] == TABLE_ARCHS
    assert archs["formula_archs"] == FORMULA_ARCHS


def _test_predictor(predictor):
    # Output checks
    assert isinstance(predictor, OCRPredictor)

    doc = [np.zeros((1024, 1024, 3), dtype=np.uint8)]
    out = predictor(doc)
    # Document
    assert isinstance(out, Document)

    # The input doc has 1 page
    assert len(out.pages) == 1
    # Dimension check
    with pytest.raises(ValueError):
        input_page = (255 * np.random.rand(1, 256, 512, 3)).astype(np.uint8)
        _ = predictor([input_page])


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize(
    "det_arch, reco_arch",
    [[det_arch, reco_arch] for det_arch, reco_arch in zip(detection.zoo.ARCHS, recognition.zoo.ARCHS)],
)
def test_zoo_models(det_arch, reco_arch, quantized):
    # Model
    predictor = models.ocr_predictor(det_arch, reco_arch, load_in_8_bit=quantized)
    _test_predictor(predictor)

    # passing model instance directly
    det_model = detection.__dict__[det_arch]()
    reco_model = recognition.__dict__[reco_arch]()
    predictor = models.ocr_predictor(det_model, reco_model)
    _test_predictor(predictor)

    # passing recognition model as detection model
    with pytest.raises(ValueError):
        models.ocr_predictor(det_arch=reco_model)

    # passing detection model as recognition model
    with pytest.raises(ValueError):
        models.ocr_predictor(reco_arch=det_model)


def test_ocrpredictor_keeps_objectness_scores_aligned():
    # The middle box lies on the right border of the page: its crop is empty, so it is dropped before recognition
    boxes = np.array(
        [[0.1, 0.1, 0.3, 0.2, 0.9], [1.0, 0.5, 1.0, 0.6, 0.1], [0.5, 0.1, 0.7, 0.2, 0.8]],
        dtype=np.float32,
    )

    class _FixedDetPredictor(NestedObject):
        def __call__(self, pages, **kwargs):
            return [boxes.copy() for _ in pages]

    class _DummyRecoPredictor(NestedObject):
        def __call__(self, crops, **kwargs):
            return [(f"word{idx}", 1.0) for idx in range(len(crops))]

    predictor = OCRPredictor(_FixedDetPredictor(), _DummyRecoPredictor(), assume_straight_pages=True)
    page = predictor([np.full((100, 200, 3), 255, dtype=np.uint8)]).pages[0]
    words = [word for block in page.blocks for line in block.lines for word in line.words]

    # Each remaining word keeps the objectness score of its own box
    assert [word.geometry[0][0] for word in words] == pytest.approx([0.1, 0.5])
    assert [word.objectness_score for word in words] == pytest.approx([0.9, 0.8])


def test_ocrpredictor_layout(mock_pdf, mock_payslip):
    det_predictor = DetectionPredictor(
        PreProcessor(output_size=(1024, 1024), batch_size=2),
        detection.db_mobilenet_v3_large(assume_straight_pages=True),
    )
    reco_predictor = RecognitionPredictor(
        PreProcessor(output_size=(32, 128), batch_size=16, preserve_aspect_ratio=True),
        recognition.crnn_vgg16_bn(),
    )
    layout_pred = layout_predictor("lw_detr_s")

    doc = DocumentFile.from_pdf(mock_pdf)

    # Without a layout predictor -> pages carry an empty layout
    predictor = OCRPredictor(det_predictor, reco_predictor, ignore_regions=["Picture", "Formula"])
    assert predictor.layout_predictor is None
    out = predictor(doc)
    assert all(page.layout == [] for page in out.pages)
    assert all(page.export()["layout"] == [] for page in out.pages)

    # With a layout predictor -> detected regions are attached to every page
    predictor = OCRPredictor(
        det_predictor, reco_predictor, layout_predictor=layout_pred, ignore_regions=["Picture", "Formula"]
    )
    assert isinstance(predictor.layout_predictor, LayoutPredictor)
    out = predictor(doc)
    assert isinstance(out, Document)
    for page in out.pages:
        assert isinstance(page.layout, list)
        assert all(isinstance(region, LayoutElement) for region in page.layout)
        # the layout is exported alongside the page
        exported = page.export()
        assert "layout" in exported
        assert exported["layout"] == [region.export() for region in page.layout]

    doc = DocumentFile.from_images(mock_payslip)

    det_predictor = detection_predictor(
        "fast_base",
        batch_size=2,
        assume_straight_pages=True,
        symmetric_pad=True,
        preserve_aspect_ratio=False,
    )
    reco_predictor = recognition_predictor("crnn_vgg16_bn", batch_size=128)

    predictor = OCRPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=True,
        straighten_pages=True,
        preserve_aspect_ratio=False,
        resolve_blocks=True,
        resolve_lines=True,
    )

    out = predictor(doc)

    assert out.pages[0].blocks[0].lines[0].words[0].value == "Mr."
    geometry_mr = np.array([[0.1083984375, 0.0634765625], [0.1494140625, 0.0859375]])
    assert np.allclose(np.array(out.pages[0].blocks[0].lines[0].words[0].geometry), geometry_mr, rtol=0.05)

    assert out.pages[0].blocks[1].lines[0].words[-1].value == "revised"
    geometry_revised = np.array([[0.7548828125, 0.126953125], [0.8388671875, 0.1484375]])
    assert np.allclose(np.array(out.pages[0].blocks[1].lines[0].words[-1].geometry), geometry_revised, rtol=0.05)

    det_predictor = detection_predictor(
        "fast_base",
        batch_size=2,
        assume_straight_pages=True,
        preserve_aspect_ratio=True,
        symmetric_pad=True,
    )

    predictor = OCRPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=True,
        straighten_pages=True,
        preserve_aspect_ratio=True,
        symmetric_pad=True,
        resolve_blocks=True,
        resolve_lines=True,
        ignore_regions=["Picture", "Formula"],
    )
    # test hooks
    predictor.add_hook(_DummyCallback())

    out = predictor(doc)

    assert out.pages[0].blocks[0].lines[0].words[0].value == "Mr."


def test_ocrpredictor_tables(mock_pdf):
    det_predictor = DetectionPredictor(
        PreProcessor(output_size=(1024, 1024), batch_size=2),
        detection.db_mobilenet_v3_large(assume_straight_pages=True),
    )
    reco_predictor = RecognitionPredictor(
        PreProcessor(output_size=(32, 128), batch_size=16, preserve_aspect_ratio=True),
        recognition.crnn_vgg16_bn(),
    )
    layout_pred = layout_predictor("lw_detr_s")
    table_pred = table_predictor("tablecenternet")

    # A table predictor requires a layout predictor (tables are located with the layout model)
    with pytest.raises(ValueError):
        OCRPredictor(det_predictor, reco_predictor, table_predictor=table_pred)

    doc = DocumentFile.from_pdf(mock_pdf)

    # Without a table predictor -> pages carry an empty list of tables
    predictor = OCRPredictor(det_predictor, reco_predictor)
    assert predictor.table_predictor is None
    out = predictor(doc)
    assert all(page.tables == [] for page in out.pages)
    assert all(page.export()["tables"] == [] for page in out.pages)

    # With layout + table predictors -> structured tables are attached and exported
    predictor = OCRPredictor(det_predictor, reco_predictor, layout_predictor=layout_pred, table_predictor=table_pred)
    assert isinstance(predictor.layout_predictor, LayoutPredictor)
    assert isinstance(predictor.table_predictor, TablePredictor)
    out = predictor(doc)
    assert isinstance(out, Document)
    for page in out.pages:
        assert isinstance(page.tables, list)
        assert all(isinstance(t, Table) for t in page.tables)
        exported = page.export()
        assert "tables" in exported
        assert exported["tables"] == [t.export() for t in page.tables]


def test_ocrpredictor_tables_factory():
    # The factory exposes a single `detect_tables` flag, which also enables the layout model
    predictor = ocr_predictor("db_mobilenet_v3_large", "crnn_vgg16_bn", detect_tables=True)
    assert isinstance(predictor.table_predictor, TablePredictor)
    assert isinstance(predictor.layout_predictor, LayoutPredictor)

    # No tables by default
    predictor = ocr_predictor("db_mobilenet_v3_large", "crnn_vgg16_bn")
    assert predictor.table_predictor is None


def _custom_aggregation(scores):
    return float(np.percentile(scores, 25))


@pytest.mark.parametrize(
    "confidence_aggregation, expected",
    [
        # Default method of the recognition model
        [None, "min"],
        ["mean", "mean"],
        ["geometric_mean", "geometric_mean"],
        [_custom_aggregation, _custom_aggregation],
    ],
)
def test_ocr_predictor_confidence_aggregation(confidence_aggregation, expected):
    predictor = models.ocr_predictor(
        "db_mobilenet_v3_large",
        "crnn_mobilenet_v3_small",
        confidence_aggregation=confidence_aggregation,
    )
    assert predictor.reco_predictor.model.postprocessor.confidence_aggregation == expected
    # The split crops keep their own aggregation
    assert predictor.reco_predictor.split_confidence_aggregation == "min"


def test_ocr_predictor_confidence_aggregation_model_instances(mock_payslip):
    reco_model = recognition.parseq()
    predictor = models.ocr_predictor(detection.db_mobilenet_v3_large(), reco_model, confidence_aggregation="min")
    assert reco_model.postprocessor.confidence_aggregation == "min"
    # None keeps the method of the model
    assert models.ocr_predictor("db_mobilenet_v3_large", reco_model).reco_predictor.model is reco_model
    assert reco_model.postprocessor.confidence_aggregation == "min"

    # End-to-end: the word confidences follow the aggregation method set on the recognition model
    doc = DocumentFile.from_images(mock_payslip)

    def word_confidences(method):
        reco_model.postprocessor.confidence_aggregation = method
        out = predictor(doc)
        return np.array([w.confidence for b in out.pages[0].blocks for line in b.lines for w in line.words])

    min_confs, max_confs = word_confidences("min"), word_confidences("max")
    mean_confs = word_confidences("mean")
    assert min_confs.size > 0
    assert np.all((min_confs >= 0) & (max_confs <= 1))
    assert np.all(min_confs <= mean_confs + 1e-6) and np.all(mean_confs <= max_confs + 1e-6)
    # The method is applied: at least one word has an uncertain character
    assert np.any(min_confs < max_confs - 1e-3)


def test_recognition_predictor_confidence_aggregation():
    # Architecture names
    reco_predictor = recognition_predictor("parseq", confidence_aggregation="min")
    assert reco_predictor.model.postprocessor.confidence_aggregation == "min"
    # Model instances are modified
    reco_model = recognition.crnn_mobilenet_v3_small()
    assert recognition_predictor(reco_model, confidence_aggregation=np.median).model is reco_model
    assert reco_model.postprocessor.confidence_aggregation is np.median
    # None keeps the method of the model
    assert recognition_predictor(reco_model).model.postprocessor.confidence_aggregation is np.median
    with pytest.raises(ValueError, match="Unknown confidence aggregation"):
        recognition_predictor("crnn_mobilenet_v3_small", confidence_aggregation="average")


@pytest.mark.parametrize(
    "detect_layout, detect_tables",
    [
        [False, False],
        [True, False],
        [False, True],
        [True, True],
    ],
)
def test_ocr_predictor_figures(mock_figure_page, tmp_path, detect_layout, detect_tables):
    predictor = models.ocr_predictor(detect_layout=detect_layout, detect_tables=detect_tables)
    page = predictor(DocumentFile.from_images(mock_figure_page)).pages[0]
    exports = {images: page.export_as_markdown(images=images) for images in ("none", "placeholder", "embedded")}
    encoder = FigureEncoder("referenced", image_dir=tmp_path, path_prefix="assets")
    referenced = page.export_as_markdown(images=encoder)
    xml = page.export_as_xml()[0].decode()

    if not (detect_layout or detect_tables):
        # Without layout, the image modes change nothing
        assert page.layout == []
        assert len({*exports.values(), referenced}) == 1
        assert encoder.written == [] and 'class="ocr_photo"' not in xml
        return

    # The layout model (also run for the tables) finds the photograph
    figures = [item for item in page.items_in_reading_order(include_figures=True) if isinstance(item, LayoutElement)]
    assert [figure.type for figure in figures] == ["Picture"]
    assert page.tables == []
    assert xml.count('class="ocr_photo"') == 1
    assert exports["placeholder"].count("<!-- image -->") == 1
    assert exports["placeholder"].replace("<!-- image -->\n\n", "") == exports["none"]
    # With its pixels, the caption is the alt text and a line below the image
    # (matched on its last words: the default ONNX recognition model can misread the 'g' of "Figure")
    caption = next(part for part in exports["none"].split("\n\n") if "sample photograph" in part)
    assert exports["embedded"].count("](data:image/png;base64,") == 1
    assert f"![{caption}](data:image/png;base64," in exports["embedded"]
    assert exports["embedded"].count(caption) == 2
    assert f"![{caption}](assets/{encoder.written[0].name})\n\n*{caption}*" in referenced
    crop = cv2.imread(str(encoder.written[0]))
    assert abs(crop.shape[0] - 500) < 25 and abs(crop.shape[1] - 800) < 40
    assert exports["embedded"].split("\n\n")[:2] == exports["none"].split("\n\n")[:2]
    assert exports["embedded"].split("\n\n")[-1] == exports["none"].split("\n\n")[-1]


def test_ocr_predictor_figures_ignore_regions(mock_figure_page):
    # Ignored regions are only masked for the text detection: the figure keeps its pixels
    predictor = models.ocr_predictor(detect_layout=True, ignore_regions=["Picture"])
    page = predictor(DocumentFile.from_images(mock_figure_page)).pages[0]
    figure = next(region for region in page.layout if region.type == "Picture")
    crop = crop_layout_region(page.page, figure.geometry)
    assert crop is not None and crop.any()
    assert page.export_as_markdown(images="embedded").count("](data:image/png;base64,") == 1


def test_ocr_predictor_straighten_with_preserve_original_coords(mock_tilted_payslip):
    doc = DocumentFile.from_images(mock_tilted_payslip)
    det_predictor = detection_predictor(
        "fast_base",
        batch_size=2,
        assume_straight_pages=False,
        symmetric_pad=True,
        preserve_aspect_ratio=False,
    )
    reco_predictor = recognition_predictor("crnn_vgg16_bn", batch_size=128)
    predictor_on = OCRPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=False,
        straighten_pages=True,
        detect_orientation=True,
        preserve_aspect_ratio=False,
        resolve_blocks=True,
        resolve_lines=True,
        preserve_original_coords=True,
    )
    predictor_off = OCRPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=False,
        straighten_pages=True,
        detect_orientation=True,
        preserve_aspect_ratio=False,
        resolve_blocks=True,
        resolve_lines=True,
        preserve_original_coords=False,
    )
    out_on = predictor_on(doc)
    out_off = predictor_off(doc)
    assert len(out_on.pages[0].blocks) > 0
    assert len(out_off.pages[0].blocks) > 0
    geoms_on = [
        np.array(w.geometry).reshape(-1, 2).tolist()
        for block in out_on.pages[0].blocks
        for line in block.lines
        for w in line.words
    ]
    geoms_off = [
        np.array(w.geometry).reshape(-1, 2).tolist()
        for block in out_off.pages[0].blocks
        for line in block.lines
        for w in line.words
    ]
    assert geoms_on != geoms_off


# --- Chunked / streaming predictors ----------------------------------------------------------------------------------


def _value_pages(num_pages):
    return [np.full((60 + idx, 80, 3), 10 * idx, dtype=np.uint8) for idx in range(num_pages)]


class _ValueDetPredictor(NestedObject):
    """Fake detection: one box per page, and records the number of pages of each call"""

    def __init__(self):
        self.calls = []

    def __call__(self, pages, **kwargs):
        self.calls.append(len(pages))
        return [np.array([[0.1, 0.1, 0.9, 0.9, 0.9]], dtype=np.float32) for _ in pages]


class _ValueRecoPredictor(NestedObject):
    """Fake recognition: reads the gray level of the crop, which identifies the page"""

    def __call__(self, crops, **kwargs):
        return [(str(round(float(crop.mean()))), 1.0) for crop in crops]


def _value_predictor(**kwargs):
    return OCRPredictor(_ValueDetPredictor(), _ValueRecoPredictor(), assume_straight_pages=True, **kwargs)


def _page_words(page):
    return [word.value for block in page.blocks for line in block.lines for word in line.words]


def _standalone_predictor(name, **kwargs):
    """Standalone predictors with small batches, so that chunks span several of them"""
    if name == "detection":
        return detection_predictor("fast_tiny", batch_size=2, **kwargs)
    if name == "recognition":
        return recognition_predictor("crnn_mobilenet_v3_small", batch_size=2, **kwargs)
    if name == "layout":
        return layout_predictor("lw_detr_s", batch_size=2, **kwargs)
    if name == "table":
        return table_predictor("tablecenternet", batch_size=2, **kwargs)
    if name == "page_orientation":
        return page_orientation_predictor(mobilenet_v3_small_page_orientation(), batch_size=2, **kwargs)
    return crop_orientation_predictor(mobilenet_v3_small_crop_orientation(), batch_size=2, **kwargs)


def _same_output(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same_output(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same_output(x, y) for x, y in zip(a, b))
    return a == b


def test_predictors_on_empty_batch():
    det_predictor = detection_predictor("db_mobilenet_v3_large", batch_size=2)
    reco_predictor = recognition_predictor("crnn_mobilenet_v3_small", batch_size=32)

    # Detection keeps the shape its `return_maps` contract promises
    assert det_predictor([]) == []
    assert det_predictor([], return_maps=True) == ([], [])
    assert reco_predictor([]) == []

    out = OCRPredictor(det_predictor, reco_predictor, assume_straight_pages=True)([])
    assert type(out) is Document
    assert out.pages == []
    assert out.export() == {"pages": []}


def test_recognition_predictor_rejects_empty_crops():
    predictor = recognition_predictor("crnn_mobilenet_v3_small")
    with pytest.raises(ValueError, match="non-empty"):
        predictor([np.zeros((0, 10, 3), dtype=np.uint8)])


def test_predictors_chunk_long_inputs():
    pages = _value_pages(7)
    predictor = _value_predictor(page_chunk_size=3)
    hook_calls = []

    def _count_pages(loc_preds):
        hook_calls.append(len(loc_preds))
        return loc_preds

    predictor.add_hook(_count_pages)

    # lists & tuples of pages are in memory already: they are processed in a single pass, so that hooks receive all
    # the pages at once
    out = predictor(pages)
    assert type(out) is Document
    assert predictor.det_predictor.calls == [7]
    assert hook_calls == [7]
    assert [page.page_idx for page in out.pages] == list(range(7))
    assert [_page_words(page) for page in out.pages] == [[str(10 * idx)] for idx in range(7)]
    assert [page.dimensions for page in out.pages] == [page.shape[:2] for page in pages]
    # in-memory pages are referenced as-is
    assert all(page.page is image for page, image in zip(out.pages, pages))
    assert predictor(tuple(pages)).export() == out.export()
    assert predictor(np.stack([np.zeros((60, 80, 3), dtype=np.uint8)] * 4)).pages[3].page_idx == 3

    # iterators are processed by chunks of at most 3 pages, the results being identical to a single pass
    predictor.det_predictor.calls.clear()
    hook_calls.clear()
    assert predictor(page for page in pages).export() == out.export()
    assert predictor.det_predictor.calls == [3, 3, 1]
    assert hook_calls == [3, 3, 1]
    # and so are lists passed to `stream`
    hook_calls.clear()
    assert [page.export() for page in predictor.stream(pages)] == [page.export() for page in out.pages]
    assert hook_calls == [3, 3, 1]


def test_predictors_invalid_page_chunk_size():
    with pytest.raises(ValueError):
        _value_predictor(page_chunk_size=0)


def test_predictors_stream():
    pages = _value_pages(5)
    predictor = _value_predictor(page_chunk_size=4)

    stream = predictor.stream(pages, page_chunk_size=2, prefetch_pages=False)
    first = next(stream)
    # results are yielded as soon as their chunk is processed
    assert predictor.det_predictor.calls == [2]
    assert (first.page_idx, _page_words(first)) == (0, ["0"])
    rest = list(stream)
    assert [page.page_idx for page in rest] == [1, 2, 3, 4]
    assert predictor.det_predictor.calls == [2, 2, 1]

    # same results with prefetching, and the predictor default chunk size
    predictor.det_predictor.calls.clear()
    streamed = list(predictor.stream(iter(pages)))
    assert predictor.det_predictor.calls == [4, 1]
    assert [page.export() for page in streamed] == [page.export() for page in [first, *rest]]
    assert list(predictor.stream([])) == []


def test_predictors_stream_early_exit_stops_decoding():
    """Stopping a stream early does not wait for the next chunk to be fully decoded in the background"""
    pages = _value_pages(4)
    decoded = []
    closed = threading.Event()

    def _slow_pages():
        try:
            for idx in range(1000):
                time.sleep(0.02)
                decoded.append(idx)
                yield pages[idx % len(pages)]
        finally:
            closed.set()

    predictor = _value_predictor(page_chunk_size=8)
    stream = predictor.stream(_slow_pages())
    first = next(stream)
    assert first.page_idx == 0
    stream.close()
    # the first chunk (8 pages), plus at most the pages decoded ahead before the stop
    assert len(decoded) < 16
    # no page is decoded after the stream is closed, and the input is closed
    count = len(decoded)
    time.sleep(0.1)
    assert len(decoded) == count
    assert closed.is_set()
    assert not [thread for thread in threading.enumerate() if thread.name.startswith("onnxtr-prefetch")]


def test_predictors_lazy_source(tmp_path, caplog):
    pages = _value_pages(5)
    for idx, page in enumerate(pages):
        cv2.imwrite(str(tmp_path / f"page_{idx}.png"), page)
    source = DocumentFile.from_images(tmp_path, lazy=True)

    predictor = _value_predictor(page_chunk_size=2)
    out = predictor(source)
    assert type(out) is Document
    assert predictor.det_predictor.calls == [2, 2, 1]
    assert out.export() == _value_predictor()(pages).export()

    # the predictions only keep a reference to their page, decoded again on access
    assert all(type(page._page).__name__ == "_SourcePage" for page in out.pages)
    assert all(np.array_equal(page.page, image) for page, image in zip(out.pages, pages))
    assert type(predictor(source[:0])) is Document
    # the reference points to the file of the page only: a prediction pickles without the rest of the document
    assert [page._page.source.files for page in out.pages] == [[file] for file in source.files]
    restored = pickle.loads(pickle.dumps(out.pages[3]))
    assert np.array_equal(restored.page, pages[3])

    # in a concatenation, the pages of an in-memory list keep their image, the others reference their own input
    in_memory = _value_pages(2)
    out = predictor(ConcatSource([in_memory, source]))
    assert [page.page_idx for page in out.pages] == list(range(7))
    assert all(page.page is image for page, image in zip(out.pages[:2], in_memory))
    assert [page._page.source.files for page in out.pages[2:]] == [[file] for file in source.files]
    assert len(pickle.dumps(out.pages[2])) < len(pickle.dumps(in_memory[0]))

    # a page that cannot be decoded (valid header, corrupted content) is only detected while processing
    (tmp_path / "page_2.png").write_bytes((tmp_path / "page_0.png").read_bytes()[:60])
    source = DocumentFile.from_images(tmp_path, lazy=True)
    stream = predictor.stream(source)
    # the error is raised at its position, once the pages before it are yielded (even within the same chunk)
    assert [next(stream).page_idx for _ in range(2)] == [0, 1]
    with pytest.raises(PageDecodeError, match="page_2.png"):
        next(stream)
    with pytest.raises(PageDecodeError, match="page_2.png"):
        predictor(source)
    # or skipped: the other pages keep their index in the input, and their results
    skipped = []
    out = predictor(source, on_error=lambda page_idx, error: skipped.append((page_idx, type(error))))
    assert [page.page_idx for page in out.pages] == [0, 1, 3, 4]
    assert skipped == [(2, PageDecodeError)]
    expected = _value_predictor()(pages).export()["pages"]
    assert [page.export() for page in out.pages] == [expected[idx] for idx in (0, 1, 3, 4)]
    # with a warning, in slices of the source as well
    with caplog.at_level(logging.WARNING, logger="onnxtr"):
        assert [page.page_idx for page in predictor.stream(source[1:], on_error="skip")] == [0, 2, 3]
    assert "page_2.png" in caplog.text
    with pytest.raises(ValueError, match="on_error"):
        next(predictor.stream(source, on_error="ignore"))
    with pytest.raises(ValueError, match="on_error"):
        predictor(pages, on_error="ignore")


def test_predictors_lazy_source_transformed_pages(tmp_path):
    """Pages modified by the pipeline (e.g. straightened) cannot be read again from the source: they keep their
    transformed image, exactly as for in-memory inputs"""

    class _TransformingPredictor(OCRPredictor):
        def _predict_pages(self, pages, **kwargs):
            out = super()._predict_pages(pages, **kwargs)
            for page in out.pages:
                page.page = np.ascontiguousarray(page.page[::-1])
            return out

    pages = _value_pages(3)
    pages[1][:10] = 255
    for idx, page in enumerate(pages):
        cv2.imwrite(str(tmp_path / f"page_{idx}.png"), page)

    predictor = _TransformingPredictor(_ValueDetPredictor(), _ValueRecoPredictor(), page_chunk_size=2)
    out = predictor(DocumentFile.from_images(tmp_path, lazy=True))
    assert all(type(page._page) is np.ndarray for page in out.pages)
    assert all(np.array_equal(page.page, image[::-1]) for page, image in zip(out.pages, pages))
    eager = predictor(pages)
    assert all(np.array_equal(lazy.page, ref.page) for lazy, ref in zip(out.pages, eager.pages))


def test_predictors_prefetch_only_lazy_inputs(tmp_path, monkeypatch):
    """In-memory inputs are chunked directly: only lazy sources and iterators are prefetched in a background thread"""
    from onnxtr.utils import multithreading

    calls = []
    _prefetch = multithreading.prefetch

    def _recording_prefetch(seq, enabled=True):
        calls.append(enabled)
        return _prefetch(seq, enabled=enabled)

    monkeypatch.setattr(multithreading, "prefetch", _recording_prefetch)

    pages = _value_pages(5)
    for idx, page in enumerate(pages):
        cv2.imwrite(str(tmp_path / f"page_{idx}.png"), page)
    predictor = _value_predictor(page_chunk_size=2)
    expected = _value_predictor()(pages).export()

    for in_memory in (pages, tuple(pages)):
        calls.clear()
        assert predictor(in_memory).export() == expected
        assert calls == []
        assert [page.export() for page in predictor.stream(in_memory)] == expected["pages"]
        assert calls == [False]

    for lazy in (iter(pages), DocumentFile.from_images(tmp_path, lazy=True)):
        calls.clear()
        assert predictor(lazy).export() == expected
        assert calls == [True]

    calls.clear()
    list(predictor.stream(iter(pages), prefetch_pages=False))
    assert calls == [False]


@pytest.mark.parametrize("assume_straight_pages", [True, False])
def test_predictors_hooks_run_before_cropping(assume_straight_pages):
    # Each box covers a region of the page filled with its own gray level, read back by the recognition stub
    levels = [10, 100, 200]
    page = np.full((100, 200, 3), 255, dtype=np.uint8)
    straight_boxes = [[0.1, 0.1, 0.3, 0.2], [0.4, 0.1, 0.6, 0.2], [0.7, 0.1, 0.9, 0.2]]
    for (xmin, ymin, xmax, ymax), level in zip(straight_boxes, levels):
        page[int(ymin * 100) : int(ymax * 100), int(xmin * 200) : int(xmax * 200)] = level
    scores = [0.9, 0.8, 0.7]
    if assume_straight_pages:
        boxes = np.array([[*box, score] for box, score in zip(straight_boxes, scores)], dtype=np.float32)
    else:
        boxes = np.array(
            [
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [score, score]]
                for (x0, y0, x1, y1), score in zip(straight_boxes, scores)
            ],
            dtype=np.float32,
        )

    class _FixedDetPredictor(NestedObject):
        def __call__(self, pages, **kwargs):
            return [boxes.copy() for _ in pages]

    class _GrayLevelRecoPredictor(NestedObject):
        def __call__(self, crops, **kwargs):
            return [(str(int(np.median(crop))), 1.0) for crop in crops]

    predictor = OCRPredictor(
        _FixedDetPredictor(),
        _GrayLevelRecoPredictor(),
        assume_straight_pages=assume_straight_pages,
        disable_page_orientation=True,
        disable_crop_orientation=True,
    )

    def _words(out):
        return [word for block in out.pages[0].blocks for line in block.lines for word in line.words]

    # A hook removing the first box: only the other boxes are read, each with its own text and score
    predictor.add_hook(lambda preds: [p[1:] for p in preds])
    words = sorted(_words(predictor([page])), key=lambda word: np.asarray(word.geometry)[..., 0].min())
    assert [word.value for word in words] == ["100", "200"]
    assert [word.objectness_score for word in words] == pytest.approx([0.8, 0.7])

    # Hooks run in order: the second one only keeps the box of the third region
    predictor.add_hook(lambda preds: [p[1:] for p in preds])
    assert [word.value for word in _words(predictor([page]))] == ["200"]

    # Invalid outputs are reported
    predictor.hooks = [lambda preds: preds[0]]
    with pytest.raises(ValueError):
        predictor([page])


@pytest.mark.parametrize(
    "name", ["detection", "recognition", "layout", "table", "page_orientation", "crop_orientation"]
)
def test_standalone_predictors_lazy_inputs(tmp_path, name):
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 255, (40 + 10 * idx, 60 + 15 * idx, 3), dtype=np.uint8) for idx in range(5)]
    for idx, image in enumerate(images):
        cv2.imwrite(str(tmp_path / f"{idx}.png"), image[..., ::-1])

    predictor = _standalone_predictor(name, chunk_size=3)
    assert predictor.chunk_size == 3
    reference = predictor._predict(images)

    # lazy page sources, slices, generators and long lists are processed chunk by chunk, with the same predictions
    source = DocumentFile.from_images(tmp_path, lazy=True)
    assert _same_output(predictor(source), reference)
    assert _same_output(predictor(image for image in images), reference)
    assert _same_output(predictor(images), reference)
    assert _same_output(predictor(tuple(images[:2])), predictor._predict(images[:2]))
    assert _same_output(predictor(source[1:4]), predictor._predict(images[1:4]))

    # empty inputs
    assert _same_output(predictor(DocumentFile.from_images([], lazy=True)), predictor([]))

    with pytest.raises(ValueError, match="chunk_size"):
        _standalone_predictor(name, chunk_size=0)


def test_detection_predictor_lazy_inputs_return_maps(tmp_path):
    images = [np.full((50 + 10 * idx, 70, 3), 20 * idx, dtype=np.uint8) for idx in range(5)]
    for idx, image in enumerate(images):
        cv2.imwrite(str(tmp_path / f"{idx}.png"), image)
    predictor = _standalone_predictor("detection", chunk_size=2)

    preds, maps = predictor(DocumentFile.from_images(tmp_path, lazy=True), return_maps=True)
    ref_preds, ref_maps = predictor._predict(images, return_maps=True)
    assert _same_output(preds, ref_preds)
    assert len(maps) == 5 and all(np.array_equal(out, ref) for out, ref in zip(maps, ref_maps))


@pytest.mark.parametrize("straighten_pages", [False, True])
def test_ocr_predictor_lazy_pdf(mock_pdf, straighten_pages):
    predictor = ocr_predictor(
        "db_mobilenet_v3_large",
        "crnn_mobilenet_v3_small",
        straighten_pages=straighten_pages,
        page_chunk_size=1,
    )
    lazy = predictor(DocumentFile.from_pdf(mock_pdf, lazy=True))
    # The reference is the list processed the same way (page by page)
    eager = Document(pages=list(predictor.stream(DocumentFile.from_pdf(mock_pdf))))

    assert lazy.export() == eager.export()
    assert lazy.render() == eager.render()
    assert all(np.array_equal(lazy_page.page, page.page) for lazy_page, page in zip(lazy.pages, eager.pages))
    assert len(lazy.export_as_markdown(images="embedded")) > 0
