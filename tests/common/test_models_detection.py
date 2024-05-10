import numpy as np
import pytest

from onnxtr.models import detection
from onnxtr.models.detection.postprocessor.base import GeneralDetectionPostProcessor
from onnxtr.models.detection.predictor import DetectionPredictor
from onnxtr.models.engine import Engine


def test_postprocessor():
    postprocessor = GeneralDetectionPostProcessor(assume_straight_pages=True)
    r_postprocessor = GeneralDetectionPostProcessor(assume_straight_pages=False)
    with pytest.raises(AssertionError):
        postprocessor(np.random.rand(2, 512, 512).astype(np.float32))
    mock_batch = np.random.rand(2, 512, 512, 1).astype(np.float32)
    out = postprocessor(mock_batch)
    r_out = r_postprocessor(mock_batch)
    # Batch composition
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(sample, list) and all(isinstance(v, np.ndarray) for v in sample) for sample in out)
    assert all(all(v.shape[1] == 5 for v in sample) for sample in out)
    assert all(all(v.shape[1] == 4 and v.shape[2] == 2 for v in sample) for sample in r_out)
    # Relative coords
    assert all(all(np.all(np.logical_and(v[:, :4] >= 0, v[:, :4] <= 1)) for v in sample) for sample in out)
    assert all(all(np.all(np.logical_and(v[:, :4] >= 0, v[:, :4] <= 1)) for v in sample) for sample in r_out)
    # Repr
    assert repr(postprocessor) == "GeneralDetectionPostProcessor(bin_thresh=0.1, box_thresh=0.1)"
    # Edge case when the expanded points of the polygon has two lists
    issue_points = np.array(
        [
            [869, 561],
            [923, 581],
            [925, 595],
            [915, 583],
            [889, 583],
            [905, 593],
            [882, 601],
            [901, 595],
            [904, 604],
            [876, 608],
            [915, 614],
            [911, 605],
            [925, 601],
            [930, 616],
            [911, 617],
            [900, 636],
            [931, 637],
            [904, 649],
            [932, 649],
            [932, 628],
            [918, 627],
            [934, 624],
            [935, 573],
            [909, 569],
            [934, 562],
        ],
        dtype=np.int32,
    )
    out = postprocessor.polygon_to_box(issue_points)
    r_out = r_postprocessor.polygon_to_box(issue_points)
    assert isinstance(out, tuple) and len(out) == 4
    assert isinstance(r_out, np.ndarray) and r_out.shape == (4, 2)


@pytest.mark.parametrize(
    "arch_name, input_shape, output_size, out_prob",
    [
        ["db_resnet34", (1024, 1024, 3), (1024, 1024, 1), True],
        ["db_resnet50", (1024, 1024, 3), (1024, 1024, 1), True],
        ["db_mobilenet_v3_large", (1024, 1024, 3), (1024, 1024, 1), True],
        ["linknet_resnet18", (1024, 1024, 3), (1024, 1024, 1), True],
        ["linknet_resnet34", (1024, 1024, 3), (1024, 1024, 1), True],
        ["linknet_resnet50", (1024, 1024, 3), (1024, 1024, 1), True],
        ["fast_tiny", (1024, 1024, 3), (1024, 1024, 1), True],
        ["fast_small", (1024, 1024, 3), (1024, 1024, 1), True],
        ["fast_base", (1024, 1024, 3), (1024, 1024, 1), True],
    ],
)
def test_detection_models(arch_name, input_shape, output_size, out_prob):
    batch_size = 2
    model = detection.__dict__[arch_name]()
    assert isinstance(model, Engine)
    input_array = np.random.rand(batch_size, *input_shape).astype(np.float32)
    out = model(input_array, return_model_output=True)
    assert isinstance(out, dict)
    assert len(out) == 2
    # Check proba map
    assert out["out_map"].shape == (batch_size, *output_size)
    assert out["out_map"].dtype == np.float32
    if out_prob:
        assert np.all(out["out_map"] >= 0) and np.all(out["out_map"] <= 1)
    # Check boxes
    for boxes_list in out["preds"]:
        for boxes in boxes_list:
            assert boxes.shape[1] == 5
            assert np.all(boxes[:, :2] < boxes[:, 2:4])
            assert np.all(boxes[:, :4] >= 0) and np.all(boxes[:, :4] <= 1)


@pytest.mark.parametrize(
    "arch_name",
    [
        "db_resnet34",
        "db_resnet50",
        "db_mobilenet_v3_large",
        "linknet_resnet18",
        "linknet_resnet34",
        "linknet_resnet50",
        "fast_tiny",
        "fast_small",
        "fast_base",
    ],
)
def test_detection_zoo(arch_name):
    # Model
    predictor = detection.zoo.detection_predictor(arch_name)
    # object check
    assert isinstance(predictor, DetectionPredictor)
    input_array = np.random.rand(2, 3, 1024, 1024).astype(np.float32)

    out, seq_maps = predictor(input_array, return_maps=True)
    assert all(isinstance(boxes, list) for boxes in out)
    for boxes in out:
        for box in boxes:
            assert isinstance(box, np.ndarray)
            assert box.shape[1] == 5
            assert np.all(box[:, :2] < box[:, 2:4])
            assert np.all(box[:, :4] >= 0) and np.all(box[:, :4] <= 1)
    assert all(isinstance(seq_map, np.ndarray) for seq_map in seq_maps)
    assert all(seq_map.shape[:2] == (1024, 1024) for seq_map in seq_maps)
    # check that all values in the seq_maps are between 0 and 1
    assert all((seq_map >= 0).all() and (seq_map <= 1).all() for seq_map in seq_maps)
