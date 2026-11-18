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
from onnxtr.models.table_structure.postprocessor.base import (
    TableCenterNetPostProcessor,
    _cluster_lines,
    _index_lines,
    _logic_from_geometry,
    _suppress_overlapping_cells,
)
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


def _grid_target(rows: int, cols: int, use_polygons: bool) -> dict[str, np.ndarray]:
    """A relative-coordinate `{"cells", "logic"}` target for a `rows x cols` grid."""
    xs, ys = np.linspace(0.1, 0.9, cols + 1), np.linspace(0.1, 0.9, rows + 1)
    cells, logic = [], []
    for r in range(rows):
        for c in range(cols):
            cells.append([[xs[c], ys[r]], [xs[c + 1], ys[r]], [xs[c + 1], ys[r + 1]], [xs[c], ys[r + 1]]])
            logic.append([c, c, r, r])

    cell_array = np.asarray(cells, dtype=np.float32).reshape(-1, 4, 2)
    if not use_polygons:
        cell_array = np.concatenate([cell_array.min(axis=1), cell_array.max(axis=1)], axis=1)
    return {"cells": cell_array, "logic": np.asarray(logic, dtype=np.int64).reshape(-1, 4)}


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


def _rotate(polys: np.ndarray, angle: float) -> np.ndarray:
    rad = np.deg2rad(angle)
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]], dtype=np.float32)
    return (polys - 0.5) @ rot.T + 0.5


@pytest.mark.parametrize("angle", [0.0, 15.0, -30.0])
def test_tablecenternet_logic_from_geometry(angle):
    rows, cols = 4, 3
    target = _grid_target(rows, cols, use_polygons=True)
    gt = target["logic"]
    # jitter the edges, like noisy predictions, then shuffle the cells
    rng = np.random.default_rng(0)
    polys = target["cells"] + rng.uniform(-0.01, 0.01, target["cells"].shape).astype(np.float32)
    perm = rng.permutation(len(polys))
    np.testing.assert_array_equal(_logic_from_geometry(_rotate(polys[perm], angle), gt[perm]), gt[perm])

    # The model drifts on a minority of the cells of a row / column: the cells on the same line share its index
    drifted = gt.copy()
    drifted[[2, 5]] += [0, 0, 1, 1]  # last cell of rows 0 and 1 one row too low
    drifted[[3]] += [1, 1, 0, 0]  # first cell of row 1 one column too far
    np.testing.assert_array_equal(_logic_from_geometry(_rotate(polys, angle), drifted), gt)

    # A row with no detected cell is kept: the cells below keep their index
    no_row_1 = np.flatnonzero(gt[:, 2] != 1)
    np.testing.assert_array_equal(_logic_from_geometry(_rotate(polys[no_row_1], angle), gt[no_row_1]), gt[no_row_1])

    # Edges of a row that do not line up (an extra line inside the first row) do not shift the rows below
    uneven = polys.copy()
    uneven[1, 2:, 1] -= 0.06  # the middle cell of the first row is shorter
    np.testing.assert_array_equal(_logic_from_geometry(_rotate(uneven, angle), gt), gt)

    # A cell spanning the 2 first columns of the first row
    merged = np.concatenate([polys[:1, [0]], polys[1:2, [1, 2]], polys[:1, [3]]], axis=1)
    spanned = np.concatenate([merged, polys[2:]])
    spanned_logic = np.concatenate([[[0, 1, 0, 0]], gt[2:]])
    np.testing.assert_array_equal(_logic_from_geometry(_rotate(spanned, angle), spanned_logic), spanned_logic)
    # its model coordinates missing the span are corrected by the geometry
    np.testing.assert_array_equal(
        _logic_from_geometry(_rotate(spanned, angle), np.concatenate([[[0, 0, 0, 0]], gt[2:]])), spanned_logic
    )


def test_tablecenternet_logic_from_geometry_model_drift():
    target = _grid_target(3, 3, use_polygons=True)
    polys, gt = target["cells"], target["logic"]
    drifted = gt.copy()
    drifted[2] += [0, 0, 1, 1]  # last cell of row 0 one row too low

    # An upside-down table (rows growing upwards): the axes follow the model coordinates
    flipped = polys.copy()
    flipped[..., 1] = 1 - flipped[..., 1]
    np.testing.assert_array_equal(_logic_from_geometry(flipped, drifted), gt)

    # The model counts one more column in the 1st row than in the 2nd one, on a regular grid: the cells of a
    # visual column share their index
    two_rows = _grid_target(2, 3, use_polygons=True)
    split = two_rows["logic"].copy()
    split[:3, :2] += [[0, 1], [1, 1], [1, 1]]
    np.testing.assert_array_equal(_logic_from_geometry(two_rows["cells"], split), two_rows["logic"])

    # The model drifts by one row on half of the columns (e.g. a borderless table): the rows follow the geometry
    grid = _grid_target(3, 4, use_polygons=True)
    drifted = grid["logic"].copy()
    drifted[drifted[:, 0] >= 2] += [0, 0, 1, 1]
    np.testing.assert_array_equal(_logic_from_geometry(grid["cells"], drifted), grid["logic"])


def test_tablecenternet_index_lines():
    def index(lines, cells, size=10.0):
        start, end = (np.array(v) for v in zip(*cells))
        return _index_lines(np.asarray(lines, dtype=float), start, end, size).tolist()

    # A regular grid: one index per line
    assert index([0, 10, 20, 30], [(0, 1), (1, 2), (2, 3)]) == [0, 1, 2, 3]
    # A stray line (a shorter cell) separates no cell from the line after it: both lines share an index
    assert index([0, 7, 10, 20], [(0, 1), (0, 2), (2, 3), (2, 3)]) == [0, 1, 1, 2]
    # A cell spanning 2 rows of the other cells
    assert index([0, 10, 20], [(0, 2), (0, 1), (1, 2)]) == [0, 1, 2]
    # A gap of 2 cell sizes covered by no cell counts as 2 empty indices, a narrow one (between cells that do not
    # touch) as none
    assert index([0, 10, 30, 40], [(0, 1), (2, 3)]) == [0, 1, 3, 4]
    assert index([0, 10, 12, 22], [(0, 1), (2, 3)]) == [0, 1, 1, 2]


def test_tablecenternet_cluster_lines():
    lines = np.array([0.0, 0.5, 1.0, 20.0, 20.5, 21.0])
    # stray edges between 2 lines do not chain them into a single one
    np.testing.assert_allclose(_cluster_lines(np.concatenate([lines, [7.0, 13.0]]), tol=8.0), [0.5, 20.5])
    # an isolated edge close to a line is absorbed by it, a far one makes its own line
    np.testing.assert_allclose(_cluster_lines(np.concatenate([lines, [30.0]]), tol=8.0), [0.5, 20.5])
    np.testing.assert_allclose(_cluster_lines(np.concatenate([lines, [40.0]]), tol=8.0), [0.5, 20.5, 40.0])
    # a single cell: every edge is isolated
    np.testing.assert_allclose(_cluster_lines(np.array([0.0, 10.0]), tol=1.0), [0.0, 10.0])


def test_tablecenternet_suppress_overlapping_cells():
    polys = _grid_target(2, 2, use_polygons=True)["cells"]
    # a duplicate of the 1st cell, spreading over the 3rd one
    duplicate = polys[:1].copy()
    duplicate[0, 2:, 1] = polys[2, 2, 1]
    cells = np.concatenate([polys, duplicate])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    np.testing.assert_array_equal(_suppress_overlapping_cells(cells, scores), [0, 1, 2, 3])
    # the duplicate wins when it has the higher score
    scores[-1] = 0.95
    np.testing.assert_array_equal(_suppress_overlapping_cells(cells, scores), [4, 1, 3])


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


def test_table_predictor_regions(monkeypatch):
    predictor = table_structure.zoo.table_predictor("tablecenternet")
    crops = []

    def fake_forward(images, **kwargs):
        # A fixed grid (crop-relative cells) for every crop: 2 cells, and 1 of the context margin, above the region
        crops.extend(images)
        h, w = images[0].shape[:2]
        pad = 20  # the context margin of a region smaller than 2000 px

        def cell(x0, y0, x1, y1, r, c):
            box = [(pad + x0) / w, (pad + y0) / h, (pad + x1) / w, (pad + y1) / h]
            return {"geometry": box, "score": 0.9, "row_start": r, "row_end": r, "col_start": c, "col_end": c}

        cells = [cell(0, 0, 100, 30, 1, 0), cell(100, 0, 200, 30, 1, 1), cell(0, -18, 200, -4, 0, 0)]
        return [{"cells": cells, "num_rows": 2, "num_cols": 2} for _ in images]

    monkeypatch.setattr(predictor, "_predict", fake_forward)
    page = np.full((400, 600, 3), 255, dtype=np.uint8)
    x0, y0, x1 = 0.1, 0.25, 0.1 + 200 / 600  # a 200 x 50 px table region, its cells cover its top 30 px
    regions = [{"class_names": ["Text", "Table"], "boxes": np.array([[0, 0, 0.1, 0.1], [x0, y0, x1, 0.375]])}, None]
    # Words in each cell, one in the region below the cells, and one outside of the region
    words = np.array([
        [0.12, 0.27, 0.15, 0.29],
        [0.30, 0.27, 0.33, 0.29],
        [0.12, 0.345, 0.15, 0.365],
        [0.12, 0.80, 0.15, 0.82],
    ])
    tables = predictor.predict_regions([page, page], regions, [words, words[:0]])
    # Only the table region is cropped, with a context margin
    assert len(crops) == 1 and crops[0].shape[:2] == (90, 240)
    assert len(tables) == 2 and len(tables[0]) == 1 and tables[1] == []
    table = tables[0][0]
    # The cell of the context margin is dropped, the word below the cells makes a new row
    assert (table["num_rows"], table["num_cols"]) == (2, 2)
    cells = sorted(table["cells"], key=lambda c: (c["row_start"], c["col_start"]))
    assert [(c["row_start"], c["col_start"], c["words"], c["score"]) for c in cells] == [
        (0, 0, [0], 0.9),
        (0, 1, [1], 0.9),
        (1, 0, [2], None),
    ]
    # The cells are mapped back to page-relative coordinates
    expected = [[x0, y0], [x0 + 100 / 600, y0], [x0 + 100 / 600, 0.325], [x0, 0.325]]
    np.testing.assert_allclose(cells[0]["geometry"], expected, atol=1e-5)
    # Without words, the cells are empty
    tables = predictor.predict_regions([page], regions[:1])
    assert all(cell["words"] == [] for cell in tables[0][0]["cells"])
