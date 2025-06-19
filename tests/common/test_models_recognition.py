import numpy as np
import pytest

from onnxtr.models import recognition
from onnxtr.models.engine import Engine
from onnxtr.models.recognition.core import RecognitionPostProcessor
from onnxtr.models.recognition.predictor import RecognitionPredictor
from onnxtr.models.recognition.predictor._utils import remap_preds, split_crops
from onnxtr.utils.vocabs import VOCABS


def test_recognition_postprocessor():
    mock_vocab = VOCABS["french"]
    post_processor = RecognitionPostProcessor(mock_vocab)
    assert post_processor.extra_repr() == f"vocab_size={len(mock_vocab)}"
    assert post_processor.vocab == mock_vocab
    assert post_processor._embedding == list(mock_vocab) + ["<eos>"]


@pytest.mark.parametrize(
    "crops, max_ratio, target_ratio, target_overlap_ratio, channels_last, num_crops",
    [
        # No split required
        [[np.zeros((32, 128, 3), dtype=np.uint8)], 8, 4, 0.5, True, 1],
        [[np.zeros((3, 32, 128), dtype=np.uint8)], 8, 4, 0.5, False, 1],
        # Split required
        [[np.zeros((32, 1024, 3), dtype=np.uint8)], 8, 6, 0.5, True, 10],
        [[np.zeros((3, 32, 1024), dtype=np.uint8)], 8, 6, 0.5, False, 10],
    ],
)
def test_split_crops(crops, max_ratio, target_ratio, target_overlap_ratio, channels_last, num_crops):
    new_crops, crop_map, should_remap = split_crops(crops, max_ratio, target_ratio, target_overlap_ratio, channels_last)
    assert len(new_crops) == num_crops
    assert len(crop_map) == len(crops)
    assert should_remap == (len(crops) != len(new_crops))


@pytest.mark.parametrize(
    "preds, crop_map, split_overlap_ratio, pred",
    [
        # Nothing to remap
        ([("hello", 0.5)], [0], 0.5, [("hello", 0.5)]),
        # Merge
        ([("hellowo", 0.5), ("loworld", 0.6)], [(0, 2, 0.5)], 0.5, [("helloworld", 0.55)]),
    ],
)
def test_remap_preds(preds, crop_map, split_overlap_ratio, pred):
    preds = remap_preds(preds, crop_map, split_overlap_ratio)
    assert len(preds) == len(pred)
    assert preds == pred
    assert all(isinstance(pred, tuple) for pred in preds)
    assert all(isinstance(pred[0], str) and isinstance(pred[1], float) for pred in preds)


@pytest.mark.parametrize(
    "inputs, max_ratio, target_ratio, target_overlap_ratio, expected_remap_required, expected_len, expected_shape, "
    "expected_crop_map, channels_last",
    [
        # Don't split
        ([np.zeros((32, 32 * 4, 3))], 4, 4, 0.5, False, 1, (32, 128, 3), 0, True),
        # Split needed
        ([np.zeros((32, 32 * 4 + 1, 3))], 4, 4, 0.5, True, 2, (32, 128, 3), (0, 2, 0.9921875), True),
        # Larger max ratio prevents split
        ([np.zeros((32, 32 * 8, 3))], 8, 4, 0.5, False, 1, (32, 256, 3), 0, True),
        # Half-overlap, two crops
        ([np.zeros((32, 128 + 64, 3))], 4, 4, 0.5, True, 2, (32, 128, 3), (0, 2, 0.5), True),
        # Half-overlap, two crops, channels first
        ([np.zeros((3, 32, 128 + 64))], 4, 4, 0.5, True, 2, (3, 32, 128), (0, 2, 0.5), False),
        # Half-overlap with small max_ratio forces split
        ([np.zeros((32, 128 + 64, 3))], 2, 4, 0.5, True, 2, (32, 128, 3), (0, 2, 0.5), True),
        # > half last overlap ratio
        ([np.zeros((32, 128 + 32, 3))], 4, 4, 0.5, True, 2, (32, 128, 3), (0, 2, 0.75), True),
        # 3 crops, half last overlap
        ([np.zeros((32, 128 + 128, 3))], 4, 4, 0.5, True, 3, (32, 128, 3), (0, 3, 0.5), True),
        # 3 crops, > half last overlap
        ([np.zeros((32, 128 + 64 + 32, 3))], 4, 4, 0.5, True, 3, (32, 128, 3), (0, 3, 0.75), True),
        # Split into larger crops
        ([np.zeros((32, 192 * 2, 3))], 4, 6, 0.5, True, 3, (32, 192, 3), (0, 3, 0.5), True),
        # Test fallback for empty splits
        ([np.empty((1, 0, 3))], -1, 4, 0.5, False, 1, (1, 0, 3), (0), True),
    ],
)
def test_split_crops_cases(
    inputs,
    max_ratio,
    target_ratio,
    target_overlap_ratio,
    expected_remap_required,
    expected_len,
    expected_shape,
    expected_crop_map,
    channels_last,
):
    new_crops, crop_map, _remap_required = split_crops(
        inputs,
        max_ratio=max_ratio,
        target_ratio=target_ratio,
        split_overlap_ratio=target_overlap_ratio,
        channels_last=channels_last,
    )

    assert _remap_required == expected_remap_required
    assert len(new_crops) == expected_len
    assert len(crop_map) == 1

    if expected_remap_required:
        assert isinstance(crop_map[0], tuple)

    assert crop_map[0] == expected_crop_map

    for crop in new_crops:
        assert crop.shape == expected_shape


@pytest.mark.parametrize(
    "split_overlap_ratio",
    [
        # lower bound
        0.0,
        # upper bound
        1.0,
    ],
)
def test_invalid_split_overlap_ratio(split_overlap_ratio):
    with pytest.raises(ValueError):
        split_crops(
            [np.zeros((32, 32 * 4, 3))],
            max_ratio=4,
            target_ratio=4,
            split_overlap_ratio=split_overlap_ratio,
        )


@pytest.mark.parametrize("quantized", [False, True])
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
        ["viptr_tiny", (32, 128, 3)],
    ],
)
def test_recognition_models(arch_name, input_shape, quantized):
    mock_vocab = VOCABS["french"]
    batch_size = 4
    model = recognition.__dict__[arch_name](load_in_8_bit=quantized)
    assert isinstance(model, Engine)
    input_array = np.random.rand(batch_size, *input_shape).astype(np.float32)

    out = model(input_array, return_model_output=True)
    assert isinstance(out, dict)
    assert len(out) == 2
    assert isinstance(out["preds"], list)
    assert len(out["preds"]) == batch_size
    assert all(isinstance(word, str) and isinstance(conf, float) and 0 <= conf <= 1 for word, conf in out["preds"])

    assert isinstance(out["out_map"], np.ndarray)
    assert out["out_map"].shape[0] == 4

    # test model post processor
    post_processor = model.postprocessor
    decoded = post_processor(np.random.rand(2, len(mock_vocab), 30).astype(np.float32))
    assert isinstance(decoded, list)
    assert all(isinstance(word, str) and isinstance(conf, float) and 0 <= conf <= 1 for word, conf in decoded)
    assert len(decoded) == 2
    assert all(char in mock_vocab for word, _ in decoded for char in word)

    # Testing with a fixed batch size
    model = recognition.__dict__[arch_name]()
    model.fixed_batch_size = 1
    assert isinstance(model, Engine)
    input_array = np.random.rand(batch_size, *input_shape).astype(np.float32)

    out = model(input_array, return_model_output=True)
    assert isinstance(out, dict)
    assert len(out) == 2
    assert isinstance(out["preds"], list)
    assert len(out["preds"]) == batch_size
    assert all(isinstance(word, str) and isinstance(conf, float) and 0 <= conf <= 1 for word, conf in out["preds"])

    assert isinstance(out["out_map"], np.ndarray)
    assert out["out_map"].shape[0] == 4


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize(
    "input_shape",
    [
        (128, 128, 3),
        (32, 1024, 3),  # test case split wide crops
    ],
)
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
        "viptr_tiny",
    ],
)
def test_recognition_zoo(arch_name, input_shape, quantized):
    batch_size = 2
    # Model
    predictor = recognition.zoo.recognition_predictor(arch_name, load_in_8_bit=quantized)
    # object check
    assert isinstance(predictor, RecognitionPredictor)
    input_array = np.random.rand(batch_size, *input_shape).astype(np.float32)
    out = predictor(input_array)
    assert isinstance(out, list) and len(out) == batch_size
    assert all(isinstance(word, str) and isinstance(conf, float) for word, conf in out)

    with pytest.raises(ValueError):
        _ = recognition.zoo.recognition_predictor(arch="wrong_model")
