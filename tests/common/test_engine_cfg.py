import gc

import numpy as np
import psutil
import pytest
from onnxruntime import RunOptions, SessionOptions

from onnxtr import models
from onnxtr.io import Document
from onnxtr.models import EngineConfig, detection, recognition
from onnxtr.models.predictor import OCRPredictor


def _get_rss_mb():
    gc.collect()
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


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

    assert engine_cfg.__repr__() == "EngineConfig(providers=['CPUExecutionProvider'])"

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


def test_cpu_memory_arena_shrinkage_enabled():
    session_options = SessionOptions()
    session_options.enable_mem_pattern = False
    session_options.enable_cpu_mem_arena = True

    enable_shrinkage = False

    providers = [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]

    def enable_arena_shrinkage(run_options: "RunOptions") -> "RunOptions":
        if enable_shrinkage:
            run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0")
            assert run_options.get_run_config_entry("memory.enable_memory_arena_shrinkage") == "cpu:0"
        return run_options

    engine_cfg = EngineConfig(
        providers=providers,
        session_options=session_options,
        run_options_provider=enable_arena_shrinkage,
    )

    predictor = models.ocr_predictor(
        det_engine_cfg=engine_cfg,
        reco_engine_cfg=engine_cfg,
        clf_engine_cfg=engine_cfg,
        detect_orientation=True,
    )

    assert predictor.det_predictor.model.providers == providers
    assert predictor.det_predictor.model.session_options.enable_cpu_mem_arena
    assert predictor.reco_predictor.model.providers == providers
    assert predictor.reco_predictor.model.session_options.enable_cpu_mem_arena

    rng = np.random.RandomState(seed=42)
    sample = rng.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)

    start_rss = _get_rss_mb()

    predictor([sample])
    increased_rss = _get_rss_mb()

    assert increased_rss > start_rss

    enable_shrinkage = True

    predictor([sample])
    decreased_rss = _get_rss_mb()

    assert increased_rss >= decreased_rss


class _FixedBatchRuntime:
    """Stub inference session whose graph only accepts batches of exactly `batch_size`, echoing them back"""

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.calls = 0

    def run(self, output_names, feed, run_options=None):
        (batch,) = feed.values()
        if batch.shape[0] != self.batch_size:
            raise ValueError(f"invalid batch size {batch.shape[0]}, expected {self.batch_size}")
        self.calls += 1
        return [batch.transpose(0, 2, 3, 1)]


@pytest.mark.parametrize("fixed_batch_size", [1, 2, 3])
@pytest.mark.parametrize("num_samples", [1, 2, 5])
def test_engine_fixed_batch_size(fixed_batch_size, num_samples):
    model = recognition.crnn_mobilenet_v3_small(engine_cfg=EngineConfig(providers=["CPUExecutionProvider"]))
    model.runtime = _FixedBatchRuntime(fixed_batch_size)
    model.fixed_batch_size = fixed_batch_size
    inputs = np.random.rand(num_samples, 3, 32, 128).astype(np.float32)

    out = model.run(inputs)

    # One output per input sample, in order, whatever the fixed batch size of the graph
    assert out.shape == (num_samples, 32, 128, 3)
    np.testing.assert_array_equal(out, inputs.transpose(0, 2, 3, 1))
    assert model.runtime.calls == -(-num_samples // fixed_batch_size)


def test_engine_warns_on_provider_fallback(caplog):
    from onnxtr.models import engine

    engine._REPORTED_FALLBACKS.clear()
    with caplog.at_level("WARNING", logger="onnxtr.models.engine"):
        missing = engine._warn_on_provider_fallback(
            [("CUDAExecutionProvider", {}), "CPUExecutionProvider"], ["CPUExecutionProvider"]
        )
        # Several models of the same pipeline falling back only warn once
        engine._warn_on_provider_fallback(["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"])
    assert missing == ["CUDAExecutionProvider"]
    assert len([r for r in caplog.records if "CUDAExecutionProvider" in r.getMessage()]) == 1

    caplog.clear()
    with caplog.at_level("WARNING", logger="onnxtr.models.engine"):
        assert engine._warn_on_provider_fallback(["CPUExecutionProvider"], ["CPUExecutionProvider"]) == []
    assert not caplog.records


def test_cuda_device_id():
    from onnxtr.models.engine import _cuda_device_id

    assert _cuda_device_id(["CPUExecutionProvider"]) is None
    assert _cuda_device_id([("CPUExecutionProvider", {})]) is None
    assert _cuda_device_id(["CUDAExecutionProvider", "CPUExecutionProvider"]) == 0
    assert _cuda_device_id([("CUDAExecutionProvider", {"device_id": 1}), "CPUExecutionProvider"]) == 1


def test_release_gpu_memory_after_run_only_on_gpu():
    from onnxtr.models import classification

    cpu_cfg = EngineConfig(providers=["CPUExecutionProvider"])
    # The occasionally called models opt in for releasing the GPU memory after each run, the main ones do not
    clf = classification.mobilenet_v3_small_crop_orientation(engine_cfg=cpu_cfg)
    reco = recognition.crnn_mobilenet_v3_small(engine_cfg=cpu_cfg)
    assert clf._release_gpu_memory_after_run and not reco._release_gpu_memory_after_run
    # ... which is a no-op on the CPU
    assert clf._arena_shrinkage is None
    with pytest.raises(RuntimeError, match="does not have configuration"):
        clf._run_options().get_run_config_entry("memory.enable_memory_arena_shrinkage")


def test_engine_cfg_uses_openvino_when_available(monkeypatch):
    from onnxtr.models import engine

    monkeypatch.setattr(
        engine, "get_available_providers", lambda: ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    )
    monkeypatch.setattr(engine, "get_device", lambda: "CPU")
    providers = [p[0] for p in EngineConfig().providers]
    assert providers == ["OpenVINOExecutionProvider", "CPUExecutionProvider"]


def test_engine_falls_back_from_openvino(monkeypatch, caplog):
    from onnxtr.models import engine

    real_session = engine.InferenceSession

    def session(path, providers, sess_options):
        if any(engine._provider_name(p) == "OpenVINOExecutionProvider" for p in providers):
            raise RuntimeError("FrontEnd API failed\nWhile validating ONNX node '<Node(GridSample): /GridSample>'")
        return real_session(path, providers=["CPUExecutionProvider"], sess_options=sess_options)

    monkeypatch.setattr(engine, "InferenceSession", session)
    cfg = EngineConfig(providers=[("OpenVINOExecutionProvider", {}), "CPUExecutionProvider"])
    with caplog.at_level("WARNING", logger="onnxtr.models.engine"):
        model = recognition.crnn_mobilenet_v3_small(engine_cfg=cfg)
    assert model.providers == ["CPUExecutionProvider"]
    assert "cannot run the model" in caplog.text and "GridSample" in caplog.text
    # Without another provider to fall back to, the error is raised
    with pytest.raises(RuntimeError, match="FrontEnd API failed"):
        recognition.crnn_mobilenet_v3_small(engine_cfg=EngineConfig(providers=["OpenVINOExecutionProvider"]))
