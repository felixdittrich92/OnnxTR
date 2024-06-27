import numpy as np
import pytest
from onnxruntime import SessionOptions

from onnxtr import models
from onnxtr.io import Document
from onnxtr.models import EngineConfig, detection, recognition
from onnxtr.models.predictor import OCRPredictor


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


@pytest.mark.parametrize(
    "det_arch, reco_arch",
    [[det_arch, reco_arch] for det_arch, reco_arch in zip(detection.zoo.ARCHS, recognition.zoo.ARCHS)],
)
def test_engine_cfg(det_arch, reco_arch):
    session_options = SessionOptions()
    session_options.enable_cpu_mem_arena = False
    engine_cfg = EngineConfig(
        providers=["CPUExecutionProvider"],
        session_options=session_options,
    )

    assert engine_cfg.__repr__() == "EngineConfig(providers=['CPUExecutionProvider']"

    # Model
    predictor = models.ocr_predictor(
        det_arch, reco_arch, det_engine_cfg=engine_cfg, reco_engine_cfg=engine_cfg, clf_engine_cfg=engine_cfg
    )
    assert predictor.det_predictor.model.providers == ["CPUExecutionProvider"]
    assert not predictor.det_predictor.model.session_options.enable_cpu_mem_arena
    assert predictor.reco_predictor.model.providers == ["CPUExecutionProvider"]
    assert not predictor.reco_predictor.model.session_options.enable_cpu_mem_arena
    _test_predictor(predictor)

    # passing model instance directly
    det_model = detection.__dict__[det_arch](engine_cfg=engine_cfg)
    assert det_model.providers == ["CPUExecutionProvider"]
    assert not det_model.session_options.enable_cpu_mem_arena

    reco_model = recognition.__dict__[reco_arch](engine_cfg=engine_cfg)
    assert reco_model.providers == ["CPUExecutionProvider"]
    assert not reco_model.session_options.enable_cpu_mem_arena

    predictor = models.ocr_predictor(det_model, reco_model)
    assert predictor.det_predictor.model.providers == ["CPUExecutionProvider"]
    assert not predictor.det_predictor.model.session_options.enable_cpu_mem_arena
    assert predictor.reco_predictor.model.providers == ["CPUExecutionProvider"]
    assert not predictor.reco_predictor.model.session_options.enable_cpu_mem_arena
    _test_predictor(predictor)

    det_predictor = models.detection_predictor(det_arch, engine_cfg=engine_cfg)
    assert det_predictor.model.providers == ["CPUExecutionProvider"]
    assert not det_predictor.model.session_options.enable_cpu_mem_arena

    reco_predictor = models.recognition_predictor(reco_arch, engine_cfg=engine_cfg)
    assert reco_predictor.model.providers == ["CPUExecutionProvider"]
    assert not reco_predictor.model.session_options.enable_cpu_mem_arena
