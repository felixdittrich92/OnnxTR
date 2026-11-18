# Copyright (C) 2021-2026, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Credits: decode logic ported from https://github.com/dreamy-xay/TableCenterNet

import math

import cv2
import numpy as np

from onnxtr.utils.geometry import order_points

__all__ = ["TableCenterNetPostProcessor"]


def _get_logic_coords(lc_logic: np.ndarray, col_span: int, row_span: int) -> tuple[int, int, int, int]:
    """Resolve a cell's logical coordinates (start/end column and row) from the per-corner logical
    predictions (`lc_logic` is a (4, 2) array of [col, row] for corners TL, TR, BR, BL) and the cell span.
    Pure numpy port of the reference `get_logic_coords`."""
    col_span = max(1, col_span)
    row_span = max(1, row_span)
    col_lc = [max(1, int(round(float(p)))) for p in lc_logic[:, 0]]
    row_lc = [max(1, int(round(float(p)))) for p in lc_logic[:, 1]]
    cols, rows = lc_logic[:, 0], lc_logic[:, 1]

    if col_lc[0] == col_lc[3]:
        start_col = col_lc[0]
        end_col = start_col + col_span - 1
    elif col_lc[1] == col_lc[2]:
        end_col = max(col_span + 1, col_lc[1]) - 1
        start_col = end_col + 1 - col_span
    elif abs(cols[0] - cols[3]) <= abs(cols[1] - cols[2]):
        start_col = max(1, int(round((cols[0] + cols[3]) / 2.0)))
        end_col = start_col + col_span - 1
    else:
        end_col = max(col_span + 1, int(round((cols[1] + cols[2]) / 2.0))) - 1
        start_col = end_col + 1 - col_span

    if row_lc[0] == row_lc[1]:
        start_row = row_lc[0]
        end_row = start_row + row_span - 1
    elif row_lc[2] == row_lc[3]:
        end_row = max(row_span + 1, row_lc[2]) - 1
        start_row = end_row + 1 - row_span
    elif abs(rows[0] - rows[1]) <= abs(rows[2] - rows[3]):
        start_row = max(1, int(round((rows[0] + rows[1]) / 2.0)))
        end_row = start_row + row_span - 1
    else:
        end_row = max(row_span + 1, int(round((rows[2] + rows[3]) / 2.0))) - 1
        start_row = end_row + 1 - row_span

    return start_col, end_col, start_row, end_row


def _bbox_overlap_query(center_polys: np.ndarray, corner_polys: np.ndarray) -> list[np.ndarray]:
    """For each center polygon, the indices of corner polygons whose axis-aligned bounding boxes overlap."""
    c_xmin, c_xmax = center_polys[:, 0::2].min(1), center_polys[:, 0::2].max(1)
    c_ymin, c_ymax = center_polys[:, 1::2].min(1), center_polys[:, 1::2].max(1)
    k_xmin, k_xmax = corner_polys[:, 0::2].min(1), corner_polys[:, 0::2].max(1)
    k_ymin, k_ymax = corner_polys[:, 1::2].min(1), corner_polys[:, 1::2].max(1)
    out = []
    for i in range(center_polys.shape[0]):
        x_ok = (k_xmin <= c_xmax[i]) & (k_xmax >= c_xmin[i])
        y_ok = (k_ymin <= c_ymax[i]) & (k_ymax >= c_ymin[i])
        out.append(np.nonzero(x_ok & y_ok)[0])
    return out


def _deskew_cells(polys: np.ndarray) -> np.ndarray:
    """Rotate (N, 4, 2) cell quadrilaterals so the rows and columns of the table become horizontal and vertical.

    The angle is the mean direction of the cell top edges modulo 90 degrees: the corner order of the cells (and a
    rotation of the table by a multiple of 90 degrees) does not matter, the axes are matched in `_logic_from_geometry`.
    """
    top_edges = polys[:, 1] - polys[:, 0]
    theta = np.arctan2(top_edges[:, 1], top_edges[:, 0])
    angle = float(np.arctan2(np.sin(4 * theta).mean(), np.cos(4 * theta).mean()) / 4)
    cos_a, sin_a = math.cos(-angle), math.sin(-angle)
    pts = polys.reshape(-1, 2) - polys.reshape(-1, 2).mean(axis=0)
    rotated = np.stack([pts[:, 0] * cos_a - pts[:, 1] * sin_a, pts[:, 0] * sin_a + pts[:, 1] * cos_a], axis=1)
    return rotated.reshape(polys.shape)


def _cell_edges(polys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Left, right, top and bottom edges of deskewed (N, 4, 2) cells."""
    return polys[..., 0].min(axis=1), polys[..., 0].max(axis=1), polys[..., 1].min(axis=1), polys[..., 1].max(axis=1)


def _suppress_overlapping_cells(polys: np.ndarray, scores: np.ndarray, thresh: float = 0.7) -> np.ndarray:
    """Indices of the cells to keep (by decreasing score) once the duplicates are removed.

    The cells of a table do not overlap: a cell covering more than `thresh` of a cell with a higher score, or the
    other way around, is a duplicate detection of the same cell.
    """
    left, right, top, bottom = _cell_edges(_deskew_cells(polys))
    areas = np.maximum(right - left, 0) * np.maximum(bottom - top, 0)
    keep: list[int] = []
    for i in np.argsort(-scores, kind="stable"):
        if keep:
            k = np.asarray(keep)
            inter_w = np.minimum(right[i], right[k]) - np.maximum(left[i], left[k])
            inter_h = np.minimum(bottom[i], bottom[k]) - np.maximum(top[i], top[k])
            inter = np.maximum(inter_w, 0) * np.maximum(inter_h, 0)
            if np.any(inter > thresh * np.maximum(np.minimum(areas[i], areas[k]), 1e-9)):
                continue
        keep.append(int(i))
    return np.asarray(keep, dtype=np.int64)


def _cluster_lines(values: np.ndarray, tol: float) -> np.ndarray:
    """Group 1D cell-edge positions into the sorted positions of the grid lines.

    The positions shared by the most edges become lines first, and absorb every edge closer than `tol`: unlike a
    single-linkage grouping, a few stray edges between two lines can not chain them into a single one. An isolated
    edge (shared by no other edge) only makes a line of its own when no line lies within `1.5 * tol`.
    """
    values = np.sort(values)
    half = tol / 2

    def window(lo: np.ndarray | float, hi: np.ndarray | float):
        return np.searchsorted(values, lo, "left"), np.searchsorted(values, hi, "right")

    lo, hi = window(values - half, values + half)
    support = hi - lo  # number of edges around each edge (itself included)
    taken = np.zeros(values.size, dtype=bool)
    lines: list[float] = []
    order = np.argsort(-support, kind="stable")
    # 1st pass: the lines shared by several edges, 2nd pass: the isolated edges
    for shared in (True, False):
        for i in order:
            if taken[i] or (shared and support[i] < 2):
                continue
            if not shared and lines and np.min(np.abs(np.asarray(lines) - values[i])) <= 1.5 * tol:
                taken[i] = True  # a stray edge of a cell, close to an existing line
                continue
            near = values[lo[i] : hi[i]][~taken[lo[i] : hi[i]]]
            center = float(np.median(near))
            start, end = window(center - tol, center + tol)
            taken[start:end] = True
            taken[i] = True
            lines.append(center)
    return np.sort(np.asarray(lines))


def _index_lines(lines: np.ndarray, start: np.ndarray, end: np.ndarray, cell_size: float) -> np.ndarray:
    """Logical index of each grid line of an axis, from the cells lying between the lines.

    Every cell spans at least one index, and the lines no cell separates share an index. A gap covered by no cell
    counts as many indices as cells of the typical size fit in it.

    Args:
        lines: the sorted positions of the grid lines, shape (L,)
        start: index of the line each cell starts on, shape (N,)
        end: index of the line each cell ends on (after its start line), shape (N,)
        cell_size: the typical size of a cell along the axis

    Returns:
        the (non-decreasing) logical index of each line, shape (L,)
    """
    values = np.zeros(lines.size, dtype=np.int64)
    for k in range(1, lines.size):
        value = values[k - 1]
        if not np.any((start <= k - 1) & (end >= k)):  # no cell covers the gap between the 2 lines
            value += int(round((lines[k] - lines[k - 1]) / max(cell_size, 1e-9)))
        ending = start[end == k]
        values[k] = max(value, int(values[ending].max()) + 1) if ending.size else value
    return values


def _align_table_logic(edges: tuple[np.ndarray, ...], logic: np.ndarray) -> np.ndarray:
    """Align the (N, 4) logical coordinates of the cells of a table on their deskewed edges."""
    left, right, top, bottom = edges
    logic = logic.copy()
    # The table may be rotated by any multiple of 90 degrees (or the corners ordered by the image axes): match each
    # logical axis (columns, rows) with the geometric axis and direction its model coordinates follow
    geo_axes = [(left, right), (top, bottom)]
    corr = np.zeros((2, 2))
    for g, (lo, hi) in enumerate(geo_axes):
        for a, offset in enumerate((0, 2)):
            pos, idx = lo + hi, logic[:, offset] + logic[:, offset + 1]
            if pos.std() > 0 and idx.std() > 0:
                corr[g, a] = float(np.corrcoef(pos, idx)[0, 1])
    swap = abs(corr[1, 0]) + abs(corr[0, 1]) > abs(corr[0, 0]) + abs(corr[1, 1])
    for a, offset in enumerate((0, 2)):
        g = 1 - a if swap else a
        start_edge, end_edge = geo_axes[g]
        if corr[g, a] < 0:  # the logical indices grow against the geometric axis
            start_edge, end_edge = -end_edge, -start_edge
        # Tolerance on the alignment of the edges, from the narrowest cells along this axis: merging the 2 lines of a
        # narrow row / column would lose it, while the lines split from a line that no cell separates share an index
        tol = max(0.4 * float(np.percentile(end_edge - start_edge, 10)), 1e-6)
        lines = _cluster_lines(np.concatenate([start_edge, end_edge]), tol)
        if lines.size < 2:  # degenerate (flat) cells: nothing to align
            continue
        # Index of the line each cell starts / ends on (a cell spans at least one row / column)
        start = np.abs(start_edge[:, None] - lines[None]).argmin(axis=1)
        end = np.abs(end_edge[:, None] - lines[None]).argmin(axis=1)
        end = np.clip(end, 1, lines.size - 1)
        start = np.minimum(start, end - 1)
        values = _index_lines(lines, start, end, float(np.median(end_edge - start_edge)))
        # Keep the absolute position the model predicts for most of the cells
        values += max(int(round(float(np.median(logic[:, offset] - values[start])))), 0)
        logic[:, offset] = values[start]
        logic[:, offset + 1] = values[end] - 1
    return logic


def _logic_from_geometry(polys: np.ndarray, model_logic: np.ndarray) -> np.ndarray:
    """Make the logical coordinates `[start_col, end_col, start_row, end_row]` (0-indexed) of the (N, 4, 2) cells of a
    table consistent with their geometry.

    The cell edges are grouped into the row and column lines of the grid, indexed from the geometry (cf.
    `_index_lines`). The model coordinates only give the direction of the axes and the absolute position of the table.

    Args:
        polys: the cells, of shape (N, 4, 2)
        model_logic: the logical coordinates predicted by the model, of shape (N, 4)

    Returns:
        the logical coordinates, of shape (N, 4)
    """
    logic = np.asarray(model_logic, dtype=np.int64)
    if logic.shape[0] == 0:
        return logic.copy()
    return _align_table_logic(_cell_edges(_deskew_cells(polys)), logic)


def _lookup_logic(lc_map: np.ndarray, x: float, y: float) -> np.ndarray:
    """Sample the (2, H, W) logical-coordinate map at a clamped pixel location."""
    h, w = lc_map.shape[1:]
    xi = 0 if x < 0 else (w - 1 if x >= w else int(x))
    yi = 0 if y < 0 else (h - 1 if y >= h else int(y))
    return lc_map[:, yi, xi]


def _any_point_inside(poly: np.ndarray, points: np.ndarray) -> bool:
    """Whether any of the given points lies strictly inside the polygon.

    NOTE: docTR uses `shapely.contains_xy` here. To avoid pulling shapely into OnnxTR's
    dependency set, this uses OpenCV's point-in-contour test, which is likewise strict
    (points on the boundary are excluded).
    """
    contour = np.ascontiguousarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    return any(cv2.pointPolygonTest(contour, (float(x), float(y)), False) > 0 for x, y in points)


class TableCenterNetPostProcessor:
    """TableCenterNet post-processor turning the model's *decoded* key-points into table cells.

    The cell geometry is returned in **relative** coordinates ([0, 1] w.r.t. the model input), so the
    predictor can undo the pre-processor's padding/resize like the other OnnxTR predictors. When
    `assume_straight_pages=True`, geometries are axis-aligned boxes of shape `(N, 4)`; otherwise they
    are quadrilaterals of shape `(N, 4, 2)`.

    Args:
        center_thresh: minimum score for a cell center to be kept
        corner_thresh: minimum score for a corner to be used during relocation
        not_relocate: if True, skip the corner-relocation step (faster, less accurate)
        assume_straight_pages: whether the pages are assumed to be straight (i.e., no rotation)
        geometric_logic: if True, remove the duplicate (overlapping) cells. `TablePredictor.predict_regions` then
            indexes the rows and columns from the cell edges instead of the logical coordinates of the model
    """

    def __init__(
        self,
        center_thresh: float = 0.3,
        corner_thresh: float = 0.3,
        not_relocate: bool = False,
        assume_straight_pages: bool = True,
        geometric_logic: bool = True,
    ) -> None:
        self.center_thresh = center_thresh
        self.corner_thresh = corner_thresh
        self.not_relocate = not_relocate
        self.assume_straight_pages = assume_straight_pages
        self.geometric_logic = geometric_logic
        # Cell score decay: cells optimised on <= 2 corners get their score scaled.
        self.cell_min_optimize_count = 2
        self.cell_decay_thresh = 0.4

    def _relocate(self, decoded: dict[str, np.ndarray], b: int):
        cp = decoded["center_polygons"][b].copy()  # (Kc, 8)
        cs = decoded["center_scores"][b].copy()  # (Kc,)
        spans = decoded["center_spans"][b]  # (Kc, 2)
        corner_polys = decoded["corner_polygons"][b]  # (Kn, 8)
        corner_scores = decoded["corner_scores"][b]  # (Kn,)
        corner_pts = decoded["corner_points"][b]  # (Kn, 2)
        corner_logics = decoded["corner_logics"][b]  # (Kn, 2)
        lc_map = decoded["lc"][b]  # (2, H, W)

        valid_c = np.nonzero(cs >= self.center_thresh)[0]
        valid_k = np.nonzero(corner_scores >= self.corner_thresh)[0]
        queries = (
            _bbox_overlap_query(cp[valid_c], corner_polys[valid_k])
            if valid_k.size
            else [np.array([], int)] * valid_c.size
        )

        logic = np.zeros((cp.shape[0], 4), dtype=np.int32)
        corner_count = np.zeros(cp.shape[0], dtype=np.int32)
        for qi, i in enumerate(valid_c):
            center_poly = cp[i].reshape(4, 2)
            cell = cp[i].reshape(4, 2)
            origin = decoded["center_polygons"][b][i].reshape(4, 2)
            lc_logic: list[np.ndarray | None] = [None, None, None, None]
            n_used = n_repeat = 0
            for j in valid_k[queries[qi]]:
                cx, cy = corner_pts[j]
                candidate_pts = corner_polys[j].reshape(4, 2)
                if not _any_point_inside(center_poly, candidate_pts):
                    continue
                # nearest corner index is computed on the ORIGINAL polygon
                idx = int(np.argmin(((origin - [cx, cy]) ** 2).sum(1)))
                ox, oy = origin[idx]
                px, py = cell[idx]
                if px == ox and py == oy:
                    n_used += 1
                    cell[idx] = [cx, cy]
                    lc_logic[idx] = corner_logics[j]
                elif (ox - px) ** 2 + (oy - py) ** 2 >= (ox - cx) ** 2 + (oy - cy) ** 2:
                    n_repeat += 1
                    cell[idx] = [cx, cy]
                    lc_logic[idx] = corner_logics[j]
            corner_count[i] = n_used + n_repeat
            for k in range(4):
                if lc_logic[k] is None:
                    lc_logic[k] = _lookup_logic(lc_map, cell[k][0], cell[k][1])
            col_span, row_span = int(round(float(spans[i][0]))), int(round(float(spans[i][1])))
            logic[i] = _get_logic_coords(np.stack(lc_logic), col_span, row_span)  # type: ignore[arg-type]
            cp[i] = cell.reshape(8)

        # Score decay for under-optimised cells, then re-sort
        keep_high = cs >= self.center_thresh
        decay = keep_high & (corner_count <= self.cell_min_optimize_count)
        cs[decay] *= self.cell_decay_thresh
        order = np.argsort(-cs)
        return cp[order], cs[order], logic[order]

    def _simple(self, decoded: dict[str, np.ndarray], b: int):
        cp = decoded["center_polygons"][b]
        cs = decoded["center_scores"][b]
        spans = decoded["center_spans"][b]
        lc_map = decoded["lc"][b]
        logic = np.zeros((cp.shape[0], 4), dtype=np.int32)
        for i in np.nonzero(cs >= self.center_thresh)[0]:
            cell = cp[i].reshape(4, 2)
            lc_logic = np.stack([_lookup_logic(lc_map, cell[k][0], cell[k][1]) for k in range(4)])
            col_span, row_span = int(round(float(spans[i][0]))), int(round(float(spans[i][1])))
            logic[i] = _get_logic_coords(lc_logic, col_span, row_span)
        return cp, cs, logic

    def __call__(self, decoded: dict[str, np.ndarray]) -> list[dict[str, np.ndarray]]:
        feat_h, feat_w = decoded["feat_size"]
        scale = np.array([feat_w, feat_h], dtype=np.float32)
        results: list[dict[str, np.ndarray]] = []
        for b in range(decoded["center_polygons"].shape[0]):
            cp, cs, logic = self._simple(decoded, b) if self.not_relocate else self._relocate(decoded, b)
            keep = cs >= self.center_thresh
            polys = cp[keep].reshape(-1, 4, 2) / scale  # relative coordinates
            polys = np.clip(polys.astype(np.float32), 0, 1)
            scores = cs[keep].astype(np.float32)
            logic = (logic[keep] - 1).astype(np.int32)  # start_col, end_col, start_row, end_row (0-indexed)
            if self.geometric_logic and polys.shape[0] > 0:
                # The logical coordinates stay the model's: `TablePredictor._region_table` indexes them from geometry
                kept = _suppress_overlapping_cells(polys, scores)
                polys, scores, logic = polys[kept], scores[kept], logic[kept]
            if self.assume_straight_pages:
                cells = np.concatenate([polys.min(axis=1), polys.max(axis=1)], axis=1).astype(np.float32)
            else:
                cells = (
                    np.stack([order_points(poly) for poly in polys]).astype(np.float32)
                    if polys.shape[0]
                    else polys.reshape(0, 4, 2).astype(np.float32)
                )
            results.append({
                "polygons": cells,  # (N, 4) boxes or (N, 4, 2) quads in relative coordinates
                "scores": scores,
                "logical": logic,  # start_col, end_col, start_row, end_row (0-indexed)
            })
        return results
