import numpy as np
import pytest

from onnxtr.models import layout
from onnxtr.models.engine import Engine
from onnxtr.models.layout.postprocessor.base import LWDETRPostProcessor
from onnxtr.models.layout.predictor import LayoutPredictor


def test_lwdetr_postprocessor():
    postprocessor = LWDETRPostProcessor(
        num_classes=5,
        score_thresh=0.2,
        iou_thresh=0.5,
        topk=50,
        assume_straight_pages=True,
    )

    r_postprocessor = LWDETRPostProcessor(
        num_classes=5,
        score_thresh=0.2,
        iou_thresh=0.5,
        topk=50,
        assume_straight_pages=False,
    )

    # Input validation: boxes must be in OBB format (cx, cy, w, h, sin, cos)
    with pytest.raises(Exception):
        postprocessor(np.random.rand(2, 20, 5).astype(np.float32), np.random.rand(2, 20, 5).astype(np.float32))

    # Forward pass
    batch_size, num_queries = 2, 20
    logits = np.random.randn(batch_size, num_queries, 6).astype(np.float32)
    boxes = np.random.rand(batch_size, num_queries, 6).astype(np.float32)

    out = postprocessor(logits, boxes)
    r_out = r_postprocessor(logits, boxes)

    # Batch composition
    assert isinstance(out, list)
    assert len(out) == batch_size

    assert isinstance(r_out, list)
    assert len(r_out) == batch_size

    assert all(isinstance(sample, tuple) and len(sample) == 3 for sample in out)
    assert all(isinstance(sample, tuple) and len(sample) == 3 for sample in r_out)

    labels, bboxes, scores = out[0]
    r_labels, r_bboxes, r_scores = r_out[0]

    assert isinstance(labels, list)
    assert isinstance(scores, list)
    assert isinstance(bboxes, np.ndarray)

    # straight pages: (K, 4)
    assert bboxes.ndim == 2
    assert bboxes.shape[1] == 4

    # rotated pages: (K, 4, 2)
    assert isinstance(r_bboxes, np.ndarray)
    assert r_bboxes.ndim == 3
    assert r_bboxes.shape[2] == 2

    # Relative coords
    assert np.all(bboxes >= 0) and np.all(bboxes <= 1)
    assert np.all(r_bboxes >= 0) and np.all(r_bboxes <= 1)

    # Score / label validity
    assert all(isinstance(s, float) for s in scores)
    assert all(s >= 0 for s in scores)
    assert len(labels) == len(scores)


def test_lwdetr_postprocessor_empty_predictions():
    # Nothing above the score threshold -> well-formed empty arrays, not a crash
    postprocessor = LWDETRPostProcessor(num_classes=5, score_thresh=0.99, assume_straight_pages=True)
    r_postprocessor = LWDETRPostProcessor(num_classes=5, score_thresh=0.99, assume_straight_pages=False)
    logits = np.full((1, 10, 5), -50.0, dtype=np.float32)
    boxes = np.random.rand(1, 10, 6).astype(np.float32)

    labels, bboxes, scores = postprocessor(logits, boxes)[0]
    assert labels == [] and scores == []
    assert bboxes.shape == (0, 4)

    r_labels, r_bboxes, r_scores = r_postprocessor(logits, boxes)[0]
    assert r_labels == [] and r_scores == []
    assert r_bboxes.shape == (0, 4, 2)


def test_lwdetr_postprocessor_nms():
    # Two near-identical boxes of the same class -> NMS keeps the highest scoring one only
    postprocessor = LWDETRPostProcessor(num_classes=2, score_thresh=0.2, iou_thresh=0.5, assume_straight_pages=True)
    logits = np.full((1, 3, 2), -50.0, dtype=np.float32)
    logits[0, 0, 0] = 4.0
    logits[0, 1, 0] = 3.0  # duplicate of query 0
    logits[0, 2, 1] = 3.0  # different class, same location -> kept (NMS is class-wise)
    boxes = np.array(
        [
            [[0.5, 0.5, 0.4, 0.2, 0.0, 1.0], [0.5, 0.5, 0.41, 0.21, 0.0, 1.0], [0.5, 0.5, 0.4, 0.2, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    labels, bboxes, scores = postprocessor(logits, boxes)[0]
    assert len(labels) == 2
    assert sorted(labels) == [0, 1]


@pytest.mark.parametrize("assume_straight_pages", [True, False])
def test_layout_models(assume_straight_pages):
    batch_size = 2
    model = layout.lw_detr_s(assume_straight_pages=assume_straight_pages)
    assert isinstance(model, Engine)
    assert len(model.class_names) == 11

    input_array = np.random.rand(batch_size, 3, 1024, 1024).astype(np.float32)
    masks = np.ones((batch_size, 1024, 1024), dtype=bool)

    out = model(input_array, masks, return_model_output=True)
    assert isinstance(out, dict)
    assert set(out) == {"logits", "pred_boxes", "preds"}
    assert out["logits"].shape == (batch_size, 195, 11)
    assert out["pred_boxes"].shape == (batch_size, 195, 6)

    assert len(out["preds"]) == batch_size
    for labels, boxes, scores in out["preds"]:
        assert boxes.shape[1:] == ((4,) if assume_straight_pages else (4, 2))
        assert len(labels) == len(scores) == boxes.shape[0]
        assert all(0 <= label < len(model.class_names) for label in labels)
        assert np.all(boxes >= 0) and np.all(boxes <= 1)


def test_layout_model_without_mask():
    # A mask is optional: every pixel is then treated as valid image content
    model = layout.lw_detr_s()
    out = model(np.random.rand(1, 3, 1024, 1024).astype(np.float32))
    assert isinstance(out, dict) and "preds" in out


@pytest.mark.parametrize("assume_straight_pages", [True, False])
def test_layout_zoo(assume_straight_pages):
    predictor = layout.zoo.layout_predictor("lw_detr_s", assume_straight_pages=assume_straight_pages, batch_size=1)
    assert isinstance(predictor, LayoutPredictor)
    # The predictor drives the padding mask through the pre-processor
    assert predictor.pre_processor.resize.return_padding_mask

    pages = [np.zeros((512, 384, 3), dtype=np.uint8)]
    out = predictor(pages)
    assert isinstance(out, list) and len(out) == len(pages)
    for page_out in out:
        assert set(page_out) == {"class_names", "boxes", "scores"}
        assert isinstance(page_out["boxes"], np.ndarray)
        assert page_out["boxes"].shape[1:] == ((4,) if assume_straight_pages else (4, 2))
        assert len(page_out["class_names"]) == len(page_out["scores"]) == page_out["boxes"].shape[0]
        assert all(name in predictor.model.class_names for name in page_out["class_names"])
        assert np.all(page_out["boxes"] >= 0) and np.all(page_out["boxes"] <= 1)


def test_layout_zoo_error():
    # Unsupported architecture
    with pytest.raises(ValueError):
        _ = layout.zoo.layout_predictor("unknown_arch")
    # Unsupported model type
    with pytest.raises(ValueError):
        _ = layout.zoo.layout_predictor(object())


def test_layout_predictor_dimension_check():
    predictor = layout.zoo.layout_predictor("lw_detr_s", batch_size=1)
    with pytest.raises(ValueError):
        _ = predictor([np.zeros((512, 384), dtype=np.uint8)])
