import numpy as np
import pytest

from onnxtr import models
from onnxtr.io import Document, DocumentFile
from onnxtr.io.elements import LayoutElement, Table
from onnxtr.models import detection, recognition
from onnxtr.models.classification import mobilenet_v3_small_crop_orientation, mobilenet_v3_small_page_orientation
from onnxtr.models.classification.zoo import crop_orientation_predictor, page_orientation_predictor
from onnxtr.models.detection.predictor import DetectionPredictor
from onnxtr.models.detection.zoo import ARCHS as DET_ARCHS
from onnxtr.models.detection.zoo import detection_predictor
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
