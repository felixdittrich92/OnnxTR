import numpy as np
import pytest

from onnxtr.models import recognition
from onnxtr.models.engine import Engine
from onnxtr.models.preprocessor import PreProcessor
from onnxtr.models.recognition.core import RecognitionPostProcessor
from onnxtr.models.recognition.models.crnn import CRNNPostProcessor
from onnxtr.models.recognition.models.master import MASTERPostProcessor
from onnxtr.models.recognition.models.parseq import PARSeqPostProcessor
from onnxtr.models.recognition.models.sar import SARPostProcessor
from onnxtr.models.recognition.models.viptr import VIPTRPostProcessor
from onnxtr.models.recognition.models.vitstr import ViTSTRPostProcessor
from onnxtr.models.recognition.predictor import RecognitionPredictor
from onnxtr.models.recognition.predictor._utils import remap_preds, split_crops
from onnxtr.utils.vocabs import VOCABS


def test_recognition_postprocessor():
    mock_vocab = VOCABS["french"]
    post_processor = RecognitionPostProcessor(mock_vocab)
    assert post_processor.extra_repr() == f"vocab_size={len(mock_vocab)}, confidence_aggregation='mean'"
    assert post_processor.vocab == mock_vocab
    assert post_processor._embedding == list(mock_vocab) + ["<eos>"]


@pytest.mark.parametrize(
    "post_processor, input_shape",
    [
        [CRNNPostProcessor, [2, 119, 30]],
        [SARPostProcessor, [2, 119, 30]],
        [ViTSTRPostProcessor, [2, 119, 30]],
        [MASTERPostProcessor, [2, 119, 30]],
        [PARSeqPostProcessor, [2, 119, 30]],
        [VIPTRPostProcessor, [2, 119, 30]],
    ],
)
def test_reco_postprocessors(post_processor, input_shape, mock_vocab):
    processor = post_processor(mock_vocab)
    decoded = processor(np.random.rand(*input_shape).astype(np.float32))
    assert isinstance(decoded, list)
    assert all(isinstance(word, str) and isinstance(conf, float) and 0 <= conf <= 1 for word, conf in decoded)
    assert len(decoded) == input_shape[0]
    assert all(char in mock_vocab for word, _ in decoded for char in word)
    # Repr
    default = "mean" if post_processor in (ViTSTRPostProcessor, PARSeqPostProcessor) else "min"
    assert repr(processor) == (
        f"{post_processor.__name__}(vocab_size={len(mock_vocab)}, confidence_aggregation={default!r})"
    )
    assert "confidence_aggregation=<lambda>" in repr(post_processor(mock_vocab, confidence_aggregation=lambda p: 1.0))


def _logits(probs: list[list[float]], num_classes: int) -> np.ndarray:
    probs_ = np.zeros((1, len(probs), num_classes), dtype=np.float32)
    probs_[0, :, : len(probs[0])] = np.asarray(probs, dtype=np.float32)
    return np.log(np.clip(probs_, 1e-9, None))


@pytest.mark.parametrize(
    "confidence_aggregation, ctc_conf, attention_conf",
    [
        ("mean", 0.75, 0.7),
        ("min", 0.6, 0.5),
        ("geometric_mean", 0.54**0.5, 0.45**0.5),
        (lambda probs: 1.0, 1.0, 1.0),
    ],
)
@pytest.mark.parametrize(
    "post_processor, num_classes",
    [
        [CRNNPostProcessor, 4],
        [VIPTRPostProcessor, 4],
        [SARPostProcessor, 4],
        [ViTSTRPostProcessor, 5],
        [MASTERPostProcessor, 6],
        [PARSeqPostProcessor, 6],
    ],
)
def test_reco_postprocessors_confidence_aggregation(
    post_processor, num_classes, confidence_aggregation, ctc_conf, attention_conf
):
    processor = post_processor("abc", confidence_aggregation=confidence_aggregation)
    if post_processor in (CRNNPostProcessor, VIPTRPostProcessor):
        # "a a <blank> b b <blank>": a character probability is the highest one within its run, blanks are ignored
        probs = [[0.5, 0.2, 0.1, 0.2], [0.9, 0.05, 0.0, 0.05], [0.1, 0.1, 0.1, 0.7], [0.1, 0.4, 0.2, 0.3]]
        probs += [[0.2, 0.6, 0.1, 0.1], [0.0, 0.0, 0.0, 1.0]]
        word, conf = processor(_logits(probs, num_classes))[0]
        assert (word, conf) == ("ab", pytest.approx(ctc_conf, abs=1e-5))
    else:
        # "a b <eos> a": the probabilities after the <eos> token are ignored
        probs = [[0.9, 0.05, 0.03, 0.02], [0.1, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.7], [0.4, 0.2, 0.2, 0.2]]
        word, conf = processor(_logits(probs, num_classes))[0]
        assert (word, conf) == ("ab", pytest.approx(attention_conf, abs=1e-5))
    # Empty word
    assert processor(_logits([[0.1, 0.1, 0.1, 0.7]] * 3, num_classes)) == [("", 0.0)]
    for invalid in ["average", ["mean"], None]:
        with pytest.raises(ValueError, match="Unknown confidence aggregation"):
            post_processor("abc", confidence_aggregation=invalid)


class _MockRecoModel:
    """Recognition model returning a predefined confidence for each crop it receives"""

    def __init__(self, confidences: list[float], postprocessor=None) -> None:
        self.confidences = confidences
        if postprocessor is not None:
            self.postprocessor = postprocessor

    def __call__(self, x: np.ndarray, **kwargs):
        return {"preds": [("ab", conf) for conf in self.confidences[: x.shape[0]]]}


@pytest.mark.parametrize(
    "postprocessor",
    [
        # The aggregation of the split parts is independent from the one of the character probabilities
        CRNNPostProcessor("abc", confidence_aggregation="max"),
        PARSeqPostProcessor("abc"),
        # A custom model does not have to provide a postprocessor with a confidence aggregation
        None,
    ],
)
def test_recognition_predictor_split_confidence_aggregation(postprocessor):
    confidences = [0.9, 0.2, 0.7, 0.8, 0.6, 0.5]
    predictor = RecognitionPredictor(
        PreProcessor(output_size=(32, 128), batch_size=32, preserve_aspect_ratio=True),
        _MockRecoModel(confidences, postprocessor),
    )
    # A wide crop split into several parts
    wide_crop = np.zeros((32, 32 * 20, 3), dtype=np.uint8)
    num_parts = len(split_crops([wide_crop], predictor.critical_ar, predictor.target_ar, predictor.overlap_ratio)[0])
    assert 2 <= num_parts <= len(confidences)
    confidences = confidences[:num_parts]

    # The lowest confidence of the parts by default
    assert predictor.split_confidence_aggregation == "min"
    assert predictor([wide_crop])[0][1] == pytest.approx(min(confidences))
    predictor.split_confidence_aggregation = "mean"
    assert predictor([wide_crop])[0][1] == pytest.approx(np.mean(confidences))
    predictor.split_confidence_aggregation = lambda confs: float(confs.max())
    assert predictor([wide_crop])[0][1] == pytest.approx(max(confidences))
    # A crop which is not split keeps the confidence of the model
    assert predictor([np.zeros((32, 128, 3), dtype=np.uint8)]) == [("ab", confidences[0])]
    # An invalid method is rejected at the first call, even without a crop to split
    predictor = RecognitionPredictor(
        PreProcessor(output_size=(32, 128), batch_size=32, preserve_aspect_ratio=True),
        _MockRecoModel(confidences, postprocessor),
    )
    predictor.split_confidence_aggregation = "average"
    with pytest.raises(ValueError, match="Unknown confidence aggregation"):
        predictor([np.zeros((32, 128, 3), dtype=np.uint8)])


@pytest.mark.parametrize(
    "arch_name", ["crnn_mobilenet_v3_small", "sar_resnet31", "master", "vitstr_small", "parseq", "viptr_tiny"]
)
def test_recognition_models_confidence_aggregation(arch_name):
    model = recognition.__dict__[arch_name](confidence_aggregation="max")
    assert model.postprocessor.confidence_aggregation == "max"


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
        # Merge: the confidence of the merged word is the one of its weakest part
        ([("hellowo", 0.5), ("loworld", 0.6)], [(0, 2, 0.5)], 0.5, [("helloworld", 0.5)]),
        # A weak part must not be averaged away
        ([("hello wor", 0.99), ("orld", 0.1)], [(0, 2, 0.5)], 0.5, [("hello world", 0.1)]),
        # Parts without text (e.g. blank padding) are ignored for the confidence
        ([("hello wor", 0.95), ("", 0.0)], [(0, 2, 0.99)], 0.5, [("hello wor", 0.95)]),
        # Only empty parts
        ([("", 0.2), ("", 0.4)], [(0, 2, 0.5)], 0.5, [("", 0.2)]),
        # Mixed: unsplit and split crops
        (
            [("single", 0.9), ("hellowo", 0.8), ("loworld", 0.7)],
            [0, (1, 3, 0.5)],
            0.5,
            [("single", 0.9), ("helloworld", 0.7)],
        ),
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
        # Zero-height crops are passed through instead of raising ZeroDivisionError
        ([np.empty((0, 64, 3))], 4, 4, 0.5, False, 1, (0, 64, 3), (0), True),
        ([np.empty((3, 0, 64))], 4, 4, 0.5, False, 1, (3, 0, 64), (0), False),
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


@pytest.mark.parametrize(
    "confidence_aggregation, conf",
    [("min", 0.1), ("mean", 0.5), ("geometric_mean", 0.3), (lambda confs: 0.42, 0.42)],
)
def test_remap_preds_confidence_aggregation(confidence_aggregation, conf):
    # The empty part is ignored for the confidence
    preds = [("hellowo", 0.9), ("", 0.0), ("loworld", 0.1)]
    assert remap_preds(preds, [(0, 3, 0.5)], 0.5, confidence_aggregation) == [("helloworld", pytest.approx(conf))]


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
