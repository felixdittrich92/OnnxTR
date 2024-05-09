import numpy as np
import pytest

from onnxtr.models import recognition
from onnxtr.models.engine import Engine
from onnxtr.models.recognition.predictor import RecognitionPredictor
from onnxtr.models.recognition.predictor._utils import remap_preds, split_crops


@pytest.mark.parametrize(
    "crops, max_ratio, target_ratio, dilation, channels_last, num_crops",
    [
        # No split required
        [[np.zeros((32, 128, 3), dtype=np.uint8)], 8, 4, 1.4, True, 1],
        [[np.zeros((3, 32, 128), dtype=np.uint8)], 8, 4, 1.4, False, 1],
        # Split required
        [[np.zeros((32, 1024, 3), dtype=np.uint8)], 8, 6, 1.4, True, 5],
        [[np.zeros((3, 32, 1024), dtype=np.uint8)], 8, 6, 1.4, False, 5],
    ],
)
def test_split_crops(crops, max_ratio, target_ratio, dilation, channels_last, num_crops):
    new_crops, crop_map, should_remap = split_crops(crops, max_ratio, target_ratio, dilation, channels_last)
    assert len(new_crops) == num_crops
    assert len(crop_map) == len(crops)
    assert should_remap == (len(crops) != len(new_crops))


@pytest.mark.parametrize(
    "preds, crop_map, dilation, pred",
    [
        # Nothing to remap
        [[("hello", 0.5)], [0], 1.4, [("hello", 0.5)]],
        # Merge
        [[("hellowo", 0.5), ("loworld", 0.6)], [(0, 2)], 1.4, [("helloworld", 0.5)]],
    ],
)
def test_remap_preds(preds, crop_map, dilation, pred):
    preds = remap_preds(preds, crop_map, dilation)
    assert len(preds) == len(pred)
    assert preds == pred
    assert all(isinstance(pred, tuple) for pred in preds)
    assert all(isinstance(pred[0], str) and isinstance(pred[1], float) for pred in preds)


@pytest.mark.parametrize(
    "arch_name, input_shape",
    [
        ["crnn_vgg16_bn", (32, 128, 3)],
        ["crnn_mobilenet_v3_small", (32, 128, 3)],
        ["crnn_mobilenet_v3_large", (32, 128, 3)],
        ["sar_resnet31", (32, 128, 3)],
        ["master", (32, 128, 3)],
        ["vitstr_small", (32, 128, 3)],
        ["vitstr_base", (32, 128, 3)],
        ["parseq", (32, 128, 3)],
    ],
)
def test_recognition_models(arch_name, input_shape, mock_vocab):
    batch_size = 4
    model = recognition.__dict__[arch_name]()
    assert isinstance(model, Engine)
    input_array = np.random.rand(batch_size, *input_shape).astype(np.float32)

    out = model(input_array)
    assert isinstance(out, dict)
    assert len(out) == 1
    assert isinstance(out["preds"], list)
    assert len(out["preds"]) == batch_size
    assert all(isinstance(word, str) and isinstance(conf, float) and 0 <= conf <= 1 for word, conf in out["preds"])

    # test model post processor
    post_processor = model.postprocessor
    decoded = post_processor(np.random.rand(2, len(mock_vocab), 30).astype(np.float32))
    assert isinstance(decoded, list)
    assert all(isinstance(word, str) and isinstance(conf, float) and 0 <= conf <= 1 for word, conf in decoded)
    assert len(decoded) == 2
    assert all(char in mock_vocab for word, _ in decoded for char in word)
    # Repr
    assert repr(post_processor) == f"{post_processor.__name__}(vocab_size={len(mock_vocab)})"


@pytest.mark.parametrize(
    "arch_name",
    [
        "crnn_vgg16_bn",
        "crnn_mobilenet_v3_small",
        "crnn_mobilenet_v3_large",
        "sar_resnet31",
        "master",
        "vitstr_small",
        "vitstr_base",
        "parseq",
    ],
)
def test_recognition_zoo(arch_name):
    batch_size = 2
    # Model
    predictor = recognition.zoo.recognition_predictor(arch_name)
    # object check
    assert isinstance(predictor, RecognitionPredictor)
    input_array = np.random.rand(batch_size, 3, 128, 128).astype(np.float32)
    out = predictor(input_array)
    assert isinstance(out, list) and len(out) == batch_size
    assert all(isinstance(word, str) and isinstance(conf, float) for word, conf in out)
