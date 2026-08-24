import numpy as np
import pytest

from onnxtr.models import table_structure
from onnxtr.models.engine import Engine
from onnxtr.models.table_structure.models.tablecenternet import (
    HEAD_NAMES,
    _gather_feat,
    _topk,
    _transpose_and_gather_feat,
)
from onnxtr.models.table_structure.postprocessor.base import TableCenterNetPostProcessor
from onnxtr.models.table_structure.predictor import TablePredictor


def _decoded(kc=12, kn=16, feat=64, batch=1):
    return {
        "center_polygons": (np.random.rand(batch, kc, 8) * feat).astype(np.float32),
        "center_scores": np.random.rand(batch, kc).astype(np.float32),
        "center_spans": np.random.randint(1, 3, (batch, kc, 2)).astype(np.float32),
        "corner_polygons": (np.random.rand(batch, kn, 8) * feat).astype(np.float32),
        "corner_scores": np.random.rand(batch, kn).astype(np.float32),
        "corner_points": (np.random.rand(batch, kn, 2) * feat).astype(np.float32),
        "corner_logics": np.random.rand(batch, kn, 2).astype(np.float32),
        "lc": (np.random.rand(batch, 2, feat, feat) * 5).astype(np.float32),
        "feat_size": (feat, feat),
    }


@pytest.mark.parametrize("assume_straight_pages", [True, False])
def test_tablecenternet_postprocessor(assume_straight_pages):
    postprocessor = TableCenterNetPostProcessor(
        center_thresh=0.0,
        assume_straight_pages=assume_straight_pages,
    )
    decoded = _decoded()
    res = postprocessor(decoded)
    assert len(res) == 1
    assert res[0]["polygons"].shape[1:] == ((4,) if assume_straight_pages else (4, 2))
    assert res[0]["logical"].shape[1] == 4
    if res[0]["polygons"].size:
        assert res[0]["polygons"].max() <= 1.0
        assert res[0]["polygons"].min() >= 0.0

    # not_relocate path follows the same geometry contract
    simple_res = TableCenterNetPostProcessor(
        center_thresh=0.0,
        not_relocate=True,
        assume_straight_pages=assume_straight_pages,
    )(decoded)
    assert len(simple_res) == 1
    assert simple_res[0]["polygons"].shape[1:] == ((4,) if assume_straight_pages else (4, 2))


@pytest.mark.parametrize("assume_straight_pages", [True, False])
def test_tablecenternet_postprocessor_empty(assume_straight_pages):
    # Nothing above threshold -> well-formed empty output
    postprocessor = TableCenterNetPostProcessor(center_thresh=1.1, assume_straight_pages=assume_straight_pages)
    res = postprocessor(_decoded())
    assert len(res) == 1
    assert res[0]["polygons"].shape[0] == 0
    assert res[0]["polygons"].shape[1:] == ((4,) if assume_straight_pages else (4, 2))
    assert res[0]["scores"].shape[0] == 0
    assert res[0]["logical"].shape == (0, 4)


def test_tablecenternet_postprocessor_batched():
    res = TableCenterNetPostProcessor(center_thresh=0.0)(_decoded(batch=3))
    assert len(res) == 3


def test_tablecenternet_decode_helpers():
    # _topk mirrors torch.topk: descending and sorted over the last axis
    scores = np.random.rand(2, 50).astype(np.float32)
    values, idxs = _topk(scores, 10)
    assert values.shape == idxs.shape == (2, 10)
    assert np.all(np.diff(values, axis=-1) <= 0)
    assert np.allclose(values, np.take_along_axis(scores, idxs, axis=-1))
    # k larger than the axis is clamped
    values, idxs = _topk(scores, 500)
    assert values.shape == (2, 50)

    # _gather_feat picks rows out of (B, N, C) by (B, K) indices
    feat = np.random.rand(2, 20, 4).astype(np.float32)
    ind = np.random.randint(0, 20, (2, 7))
    gathered = _gather_feat(feat, ind)
    assert gathered.shape == (2, 7, 4)
    for b in range(2):
        for k in range(7):
            assert np.array_equal(gathered[b, k], feat[b, ind[b, k]])

    # _transpose_and_gather_feat does the same on a (B, C, H, W) head map
    head = np.random.rand(2, 3, 8, 8).astype(np.float32)
    ind = np.random.randint(0, 64, (2, 5))
    gathered = _transpose_and_gather_feat(head, ind)
    assert gathered.shape == (2, 5, 3)
    for b in range(2):
        for k in range(5):
            y, x = divmod(int(ind[b, k]), 8)
            assert np.array_equal(gathered[b, k], head[b, :, y, x])


@pytest.mark.parametrize("assume_straight_pages", [True, False])
def test_table_structure_models(assume_straight_pages):
    batch_size = 2
    model = table_structure.tablecenternet(assume_straight_pages=assume_straight_pages)
    assert isinstance(model, Engine)

    input_array = np.random.rand(batch_size, 3, 1024, 1024).astype(np.float32)
    out = model(input_array, return_model_output=True)
    assert isinstance(out, dict)
    assert set(out) == {"out_map", "preds"}
    # All six dense heads are exposed
    assert set(out["out_map"]) == set(HEAD_NAMES)
    assert out["out_map"]["hm"].shape[:2] == (batch_size, 2)
    assert out["out_map"]["ct2cn"].shape[1] == 8

    assert len(out["preds"]) == batch_size
    for pred in out["preds"]:
        assert set(pred) == {"polygons", "scores", "logical"}
        assert pred["polygons"].shape[1:] == ((4,) if assume_straight_pages else (4, 2))
        assert pred["logical"].shape[1] == 4
        assert pred["polygons"].shape[0] == pred["scores"].shape[0] == pred["logical"].shape[0]
        if pred["polygons"].size:
            assert np.all(pred["polygons"] >= 0) and np.all(pred["polygons"] <= 1)


@pytest.mark.parametrize("assume_straight_pages", [True, False])
def test_table_zoo(assume_straight_pages):
    predictor = table_structure.zoo.table_predictor(
        "tablecenternet", assume_straight_pages=assume_straight_pages, batch_size=1
    )
    assert isinstance(predictor, TablePredictor)

    pages = [np.zeros((512, 384, 3), dtype=np.uint8)]
    out = predictor(pages)
    assert isinstance(out, list) and len(out) == len(pages)
    for page_out in out:
        assert set(page_out) == {"cells", "num_rows", "num_cols"}
        assert isinstance(page_out["cells"], list)
        for cell in page_out["cells"]:
            assert set(cell) == {"geometry", "score", "row_start", "row_end", "col_start", "col_end"}
            geom = np.asarray(cell["geometry"])
            assert geom.shape == ((4,) if assume_straight_pages else (4, 2))
            assert np.all(geom >= 0) and np.all(geom <= 1)
            assert cell["row_start"] <= cell["row_end"]
            assert cell["col_start"] <= cell["col_end"]
        # Grid size is the largest logical index + 1
        if page_out["cells"]:
            assert page_out["num_rows"] == max(c["row_end"] for c in page_out["cells"]) + 1
            assert page_out["num_cols"] == max(c["col_end"] for c in page_out["cells"]) + 1
        else:
            assert page_out["num_rows"] == 0 and page_out["num_cols"] == 0


def test_table_zoo_error():
    # Unsupported architecture
    with pytest.raises(ValueError):
        _ = table_structure.zoo.table_predictor("unknown_arch")
    # Unsupported model type
    with pytest.raises(ValueError):
        _ = table_structure.zoo.table_predictor(object())


def test_table_predictor_dimension_check():
    predictor = table_structure.zoo.table_predictor("tablecenternet", batch_size=1)
    with pytest.raises(ValueError):
        _ = predictor([np.zeros((512, 384), dtype=np.uint8)])
