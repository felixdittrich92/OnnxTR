import numpy as np
import pytest

from onnxtr.io import Document, DocumentFile
from onnxtr.io.elements import Formula
from onnxtr.models import formula, layout_predictor
from onnxtr.models.engine import Engine
from onnxtr.models.formula.models.formulanet import FormulaNetPostProcessor
from onnxtr.models.formula.predictor import FormulaPredictor
from onnxtr.models.formula.zoo import formula_predictor
from onnxtr.models.predictor import OCRPredictor
from onnxtr.models.zoo import ocr_predictor
from onnxtr.utils import LATEX_TOKENS

# The formula of the `mock_formula_image` fixture
_MOCK_FORMULA = r"J ( u , y ) = \int _ { 0 } ^ { 1 } \frac { | u | ^ { 2 } } { 2 } \, d x + \Psi ( y )"


def _logits(token_seqs, num_steps, vocab_size):
    """One-hot logits decoding the given token sequences, followed by EOS"""
    logits = np.full((len(token_seqs), num_steps, vocab_size + 1), -10.0, dtype=np.float32)
    for idx, tokens in enumerate(token_seqs):
        for step, token in enumerate([*tokens, vocab_size]):
            if step < num_steps:
                logits[idx, step, token] = 10.0
    return logits


def test_formulanet_postprocessor():
    vocab = ["a", "b", "\\frac", "{", "}"]
    postprocessor = FormulaNetPostProcessor(vocab)
    assert repr(postprocessor) == "FormulaNetPostProcessor(vocab_size=5, confidence_aggregation='mean')"
    # Decoding stops at the first EOS, the tokens are separated by spaces
    out = postprocessor(_logits([[2, 3, 0, 4, 3, 1, 4], [1], []], 10, len(vocab)))
    assert [latex for latex, _ in out] == ["\\frac { a } { b }", "b", ""]
    assert all(0 <= conf <= 1 for _, conf in out)
    # No EOS: the whole sequence is kept
    logits = np.zeros((1, 3, len(vocab) + 1), dtype=np.float32)
    logits[0, :, 0] = 10.0
    assert postprocessor(logits)[0][0] == "a a a"
    # Confidence aggregation
    probs = np.array([[[0.0, 2.0, 0.0, 0.0, 0.0, -10.0], [0.0, 0.0, 0.0, 0.0, 0.0, 10.0]]], dtype=np.float32)
    mean_conf = FormulaNetPostProcessor(vocab)(probs)[0][1]
    assert FormulaNetPostProcessor(vocab, confidence_aggregation="min")(probs)[0][1] == pytest.approx(mean_conf)
    with pytest.raises(ValueError):
        FormulaNetPostProcessor(vocab, confidence_aggregation="unknown")


def test_formulanet():
    model = formula.formulanet()
    assert isinstance(model, Engine)
    assert model.cfg["input_shape"] == (3, 256, 768)
    assert model.vocab == LATEX_TOKENS
    out = model(np.random.rand(2, 3, 256, 768).astype(np.float32), return_model_output=True)
    assert out["out_map"].shape[0] == 2 and out["out_map"].shape[-1] == len(LATEX_TOKENS) + 1
    assert len(out["preds"]) == 2
    assert all(isinstance(latex, str) and 0 <= conf <= 1 for latex, conf in out["preds"])


def test_formula_predictor(mock_formula_image):
    predictor = formula_predictor("formulanet", batch_size=2)
    assert isinstance(predictor, FormulaPredictor)
    # Same resizing as during training: padded at the bottom-right
    assert predictor.pre_processor.resize.symmetric_pad is False
    out = predictor(DocumentFile.from_images(mock_formula_image))
    assert len(out) == 1
    assert out[0][0] == _MOCK_FORMULA.replace(r"\frac { | u | ^ { 2 } } { 2 }", r"{ \frac { | u | ^ { 2 } } { 2 } }")
    assert out[0][1] > 0.9
    # Lazy page sources and empty inputs
    assert predictor(DocumentFile.from_images(mock_formula_image, lazy=True)) == out
    assert predictor([]) == []
    # Input checks
    with pytest.raises(ValueError):
        predictor([np.zeros((10, 10), dtype=np.uint8)])
    with pytest.raises(ValueError):
        predictor([np.zeros((0, 10, 3), dtype=np.uint8)])
    with pytest.raises(ValueError, match="chunk_size"):
        formula_predictor("formulanet", chunk_size=0)
    # Unknown architectures
    with pytest.raises(ValueError):
        formula_predictor("unknown_arch")
    with pytest.raises(ValueError):
        formula_predictor(object())
    # Confidence aggregation of a model instance
    model = formula.formulanet()
    predictor = formula_predictor(model, confidence_aggregation="min")
    assert predictor.model.postprocessor.confidence_aggregation == "min"
    with pytest.raises(ValueError):
        formula_predictor("formulanet", confidence_aggregation="unknown")


def test_formula_predictor_regions(monkeypatch):
    predictor = formula_predictor("formulanet")
    crops = []

    def fake_predict(images, **kwargs):
        crops.extend(images)
        return [(f"x_{idx}", 0.5) for idx in range(len(images))]

    monkeypatch.setattr(predictor, "_predict", fake_predict)
    page = np.full((400, 600, 3), 255, dtype=np.uint8)
    polygon = np.array([[0.5, 0.5], [0.8, 0.5], [0.8, 0.6], [0.5, 0.6]], dtype=np.float32)
    regions = [
        {
            "class_names": ["Text", "Formula", "Formula", "Formula"],
            "boxes": [
                np.array([0, 0, 0.1, 0.1]),
                np.array([0.1, 0.2, 0.4, 0.3]),
                polygon,
                np.array([0.2, 0.2, 0.2, 0.2]),
            ],
            "scores": [0.9, 0.8, 0.7, 0.6],
        },
        None,
    ]
    formulas = predictor.predict_regions([page, page], regions)
    # Only the (non-degenerate) formula regions are cropped: a straight box and a polygon
    assert len(crops) == 2
    assert crops[0].shape[:2] == (41, 181)
    assert len(formulas) == 2 and formulas[1] == []
    assert [(f["value"], f["confidence"], f["score"]) for f in formulas[0]] == [("x_0", 0.5, 0.8), ("x_1", 0.5, 0.7)]
    np.testing.assert_allclose(formulas[0][0]["geometry"], [0.1, 0.2, 0.4, 0.3])
    np.testing.assert_allclose(formulas[0][1]["geometry"], polygon)
    # No region at all
    assert predictor.predict_regions([page], [None]) == [[]]


def test_ocr_predictor_formulas(mock_formula_image):
    # A formula predictor requires a layout predictor
    det_predictor = ocr_predictor().det_predictor
    reco_predictor = ocr_predictor().reco_predictor
    with pytest.raises(ValueError, match="layout_predictor"):
        OCRPredictor(det_predictor, reco_predictor, formula_predictor=formula_predictor())

    predictor = ocr_predictor(detect_formulas=True)
    assert predictor.layout_predictor is not None and predictor.formula_predictor is not None
    # The formula regions are masked out of the text detection
    assert "Formula" in predictor._masked_classes()

    # A page holding the formula image: the layout model locates it, the formula model reads it
    image = DocumentFile.from_images(mock_formula_image)[0]
    page = np.full((image.shape[0] + 200, image.shape[1] + 200, 3), 255, dtype=np.uint8)
    page[100 : 100 + image.shape[0], 100 : 100 + image.shape[1]] = image
    out = predictor([page])
    assert isinstance(out, Document)
    formulas = out.pages[0].formulas
    assert all(isinstance(f, Formula) for f in formulas)
    for f in formulas:
        assert isinstance(f.value, str) and 0 <= f.confidence <= 1
    # Exported, and read back
    exported = out.export()
    assert "formulas" in exported["pages"][0]
    assert len(Document.from_dict(exported).pages[0].formulas) == len(formulas)
    if formulas:
        assert "$$" in out.export_as_markdown()

    # Without `detect_formulas`, no formula model is built and the pages have no formula
    assert ocr_predictor().formula_predictor is None
    assert ocr_predictor()([page]).pages[0].formulas == []


def test_layout_predictor_lw_detr_m():
    predictor = layout_predictor("lw_detr_m", batch_size=1)
    out = predictor([np.full((512, 384, 3), 255, dtype=np.uint8)])
    assert len(out) == 1 and set(out[0]) == {"class_names", "boxes", "scores"}
