# Copyright (C) 2021-2026, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Credits: decode logic ported from https://github.com/dreamy-xay/TableCenterNet

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
    """

    def __init__(
        self,
        center_thresh: float = 0.3,
        corner_thresh: float = 0.3,
        not_relocate: bool = False,
        assume_straight_pages: bool = True,
    ) -> None:
        self.center_thresh = center_thresh
        self.corner_thresh = corner_thresh
        self.not_relocate = not_relocate
        self.assume_straight_pages = assume_straight_pages
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
                "scores": cs[keep].astype(np.float32),
                "logical": (logic[keep] - 1).astype(np.int32),  # start_col, end_col, start_row, end_row (0-indexed)
            })
        return results
