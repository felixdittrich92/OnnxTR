import numpy as np
import pytest

from onnxtr.models import ocr_predictor, recognition
from onnxtr.models.preprocessor import PreProcessor
from onnxtr.models.recognition.predictor import RecognitionPredictor
from onnxtr.models.utils import (
    WhitelistHandle,
    _anyascii_nearest_map,
    _keep_and_reassign,
    _logits_width,
    _projection_weights,
    _weights_nearest_map,
    add_whitelist,
)


def _predictor(arch_name):
    model = recognition.__dict__[arch_name]()
    return RecognitionPredictor(PreProcessor((32, 128), batch_size=4, preserve_aspect_ratio=True), model), model


class _LogitRecorder:
    """Stands in for the real post-processor so the constrained logits can be inspected."""

    def __init__(self, postprocessor):
        self.postprocessor = postprocessor
        self.logits = None

    def __call__(self, logits, *args, **kwargs):
        self.logits = logits
        return self.postprocessor(logits, *args, **kwargs)


@pytest.mark.parametrize(
    "arch_name",
    [
        "crnn_mobilenet_v3_small",
        "vitstr_small",
        "viptr_tiny",
        "master",  # 3 special tokens instead of 1
    ],
)
def test_add_whitelist(arch_name):
    predictor, model = _predictor(arch_name)
    vocab = model.cfg["vocab"]
    # a whitelist that is a strict subset of the model's own vocabulary
    _whitelist = "abcdefghijklmnopqrstuvwxyz "
    allowed = {char for char in _whitelist if char in vocab}
    assert allowed  # sanity: the model knows these characters

    forbidden_idx = [i for i, c in enumerate(vocab) if c not in allowed]
    allowed_idx = [i for i, c in enumerate(vocab) if c in allowed]
    terminator_idx = len(vocab)
    assert len(forbidden_idx) > 0

    samples = [(255 * np.random.rand(32, 128, 3)).astype(np.uint8) for _ in range(4)]

    # record what the post-processor actually receives once the whitelist is installed
    recorder = _LogitRecorder(model.postprocessor)
    model.postprocessor = recorder

    handle = add_whitelist(predictor, _whitelist)
    assert isinstance(handle, WhitelistHandle)
    preds = predictor(samples)
    logits = recorder.logits

    # forbidden characters are masked out, while whitelisted characters and the terminator stay finite
    assert np.isneginf(logits[..., forbidden_idx]).all()
    assert np.isfinite(logits[..., allowed_idx]).all()
    assert np.isfinite(logits[..., terminator_idx]).all()
    # every column past the vocabulary is a special token and must stay finite so decoding ends
    assert np.isfinite(logits[..., len(vocab) :]).all()
    # the decoded output only contains whitelisted characters (and no leaked special tokens)
    for word, _ in preds:
        assert all(char in allowed for char in word)

    # remove() restores the original, unconstrained decoding
    handle.remove()
    _ = predictor(samples)
    assert np.isfinite(recorder.logits).all()


@pytest.mark.parametrize("arch_name", ["crnn_mobilenet_v3_small", "vitstr_small"])
def test_add_whitelist_changes_prediction(arch_name):
    predictor, model = _predictor(arch_name)
    from conftest import synthesize_text_img

    img = np.asarray(synthesize_text_img("12345", background_color=(255, 255, 255), text_color=(0, 0, 0)))
    baseline = predictor([img])[0][0]
    assert "1" in baseline

    with add_whitelist(predictor, "023456789"):  # '1' deliberately absent
        constrained = predictor([img])[0][0]
    assert "1" not in constrained
    # the context manager restored the original behaviour
    assert predictor([img])[0][0] == baseline


def test_add_whitelist_is_removable_via_context_manager():
    predictor, model = _predictor("crnn_mobilenet_v3_small")
    original = model.postprocessor
    with add_whitelist(predictor, "abc"):
        assert model.postprocessor is not original
    assert model.postprocessor is original


def test_add_whitelist_nearest_folds_to_base():
    _, model = _predictor("crnn_mobilenet_v3_small")
    vocab = model.cfg["vocab"]
    width = _logits_width(model, len(vocab))
    allowed = set("abcou")

    char_map = _anyascii_nearest_map(vocab, allowed)
    assert char_map.get("A") == "a"

    keep, src, dst = _keep_and_reassign(vocab, allowed, width, char_map)
    from onnxtr.models.utils import _ConstrainedPostProcessor

    constrainer = _ConstrainedPostProcessor(lambda x: x, keep, src, dst)

    logits = np.zeros((1, 1, width), dtype=np.float32)
    logits[0, 0, vocab.index("A")] = 9.0  # the model strongly wants a forbidden 'A'
    out = constrainer._constrain(logits)

    assert out[0, 0, vocab.index("a")] == 9.0  # folded onto the allowed base
    assert np.isneginf(out[0, 0, vocab.index("A")])  # and the forbidden column is masked
    assert vocab[int(np.argmax(out[0, 0, : len(vocab)]))] == "a"

    # mask-only does not fold: the score is simply dropped
    keep, src, dst = _keep_and_reassign(vocab, allowed, width, {})
    out = _ConstrainedPostProcessor(lambda x: x, keep, src, dst)._constrain(logits)
    assert out[0, 0, vocab.index("a")] == 0.0


def test_add_whitelist_nearest_custom_mapping():
    _, model = _predictor("crnn_mobilenet_v3_small")
    vocab = model.cfg["vocab"]
    width = _logits_width(model, len(vocab))
    allowed = set("abcou")

    char_map = {**_anyascii_nearest_map(vocab, allowed), "A": "b"}  # override A -> b
    keep, src, dst = _keep_and_reassign(vocab, allowed, width, char_map)
    from onnxtr.models.utils import _ConstrainedPostProcessor

    logits = np.zeros((1, 1, width), dtype=np.float32)
    logits[0, 0, vocab.index("A")] = 9.0
    out = _ConstrainedPostProcessor(lambda x: x, keep, src, dst)._constrain(logits)
    assert out[0, 0, vocab.index("b")] == 9.0
    assert out[0, 0, vocab.index("a")] == 0.0


def test_add_whitelist_nearest_weights_stays_within_whitelist():
    _, model = _predictor("crnn_mobilenet_v3_small")
    vocab = model.cfg["vocab"]
    width = _logits_width(model, len(vocab))
    allowed = set("abcou")

    projection = _projection_weights(model, width)
    assert projection.shape[0] == width  # oriented as (out_features, hidden)

    char_map = _weights_nearest_map(vocab, allowed, projection)
    assert char_map  # something was mapped
    assert all(target in allowed for target in char_map.values())
    assert all(source not in allowed for source in char_map)


def test_add_whitelist_end_to_end():
    predictor = ocr_predictor(reco_arch="crnn_mobilenet_v3_small", det_bs=1)
    page = (255 * np.ones((300, 900, 3))).astype(np.uint8)
    with add_whitelist(predictor, "0123456789"):
        out = predictor([page])
    for block in out.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                assert all(char in "0123456789" for char in word.value)


def test_add_whitelist_errors():
    predictor, _ = _predictor("crnn_mobilenet_v3_small")
    # unknown strategy
    with pytest.raises(ValueError):
        add_whitelist(predictor, "abc", strategy="unknown")
    # mapping is meaningless without strategy='nearest'
    with pytest.raises(ValueError):
        add_whitelist(predictor, "abc", strategy="mask", mapping="anyascii")
    # unknown mapping
    with pytest.raises(ValueError):
        add_whitelist(predictor, "abc", strategy="nearest", mapping="unknown")
    # a whitelist sharing nothing with the model vocabulary
    with pytest.raises(ValueError):
        add_whitelist(predictor, "中文")
    # not a predictor nor a recognition model
    with pytest.raises(TypeError):
        add_whitelist(object(), "abc")


def test_add_whitelist_accepts_multiple_vocabs():
    predictor, model = _predictor("crnn_mobilenet_v3_small")
    handle = add_whitelist(predictor, ["abc", "123"])
    keep = model.postprocessor.keep
    vocab = model.cfg["vocab"]
    for char in "abc123":
        assert keep[vocab.index(char)]
    assert not keep[vocab.index("z")]
    handle.remove()
