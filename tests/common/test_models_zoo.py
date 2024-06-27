import numpy as np
import pytest

from onnxtr import models
from onnxtr.io import Document, DocumentFile
from onnxtr.models import detection, recognition
from onnxtr.models.detection.predictor import DetectionPredictor
from onnxtr.models.detection.zoo import ARCHS as DET_ARCHS
from onnxtr.models.detection.zoo import detection_predictor
from onnxtr.models.predictor import OCRPredictor
from onnxtr.models.preprocessor import PreProcessor
from onnxtr.models.recognition.predictor import RecognitionPredictor
from onnxtr.models.recognition.zoo import ARCHS as RECO_ARCHS
from onnxtr.models.recognition.zoo import recognition_predictor
from onnxtr.models.zoo import ocr_predictor
from onnxtr.utils.repr import NestedObject


# Create a dummy callback
class _DummyCallback:
    def __call__(self, loc_preds):
        return loc_preds


@pytest.mark.parametrize(
    "assume_straight_pages, straighten_pages",
    [
        [True, False],
        [False, False],
        [True, True],
    ],
)
def test_ocrpredictor(mock_pdf, assume_straight_pages, straighten_pages):
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
