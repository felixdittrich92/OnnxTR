# Copyright (C) 2021-2026, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Credits: decode logic ported from https://github.com/dreamy-xay/TableCenterNet

import logging
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.special import expit

from onnxtr.utils.geometry import shape_translate

from ...engine import Engine, EngineConfig
from ..postprocessor.base import TableCenterNetPostProcessor

__all__ = ["TableCenterNet", "tablecenternet"]

logger = logging.getLogger(__name__)

# The order the heads are written by docTR's ONNX export
HEAD_NAMES = ["hm", "reg", "ct2cn", "cn2ct", "lc", "sp"]

default_cfgs: dict[str, dict[str, Any]] = {
    "tablecenternet": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://github.com/felixdittrich92/OnnxTR/releases/download/v0.8.1/tablecenternet-2c8de407.onnx",
        "url_8_bit": None,
    },
}


def _topk(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Numpy equivalent of `torch.topk` over the last axis (descending, sorted).

    Args:
        scores: (B, N) array of scores
        k: number of elements to keep

    Returns:
        tuple of (topk scores, topk indices), both of shape (B, k)
    """
    k = min(k, scores.shape[-1])
    idxs = np.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    part = np.take_along_axis(scores, idxs, axis=-1)
    order = np.argsort(-part, axis=-1, kind="stable")
    idxs = np.take_along_axis(idxs, order, axis=-1)
    return np.take_along_axis(scores, idxs, axis=-1), idxs


def _gather_feat(feat: np.ndarray, ind: np.ndarray) -> np.ndarray:
    """Gather features at specific indices.

    Args:
        feat: (B, N, C) array of features
        ind: (B, K) array of indices

    Returns:
        (B, K, C) array of gathered features
    """
    dim = feat.shape[2]
    ind = np.repeat(ind[..., None], dim, axis=2)
    return np.take_along_axis(feat, ind, axis=1)


def _transpose_and_gather_feat(feat: np.ndarray, ind: np.ndarray) -> np.ndarray:
    """Transpose and gather features at specific indices.

    Args:
        feat: (B, C, H, W) head map
        ind: (B, K) array of flat spatial indices

    Returns:
        (B, K, C) array of gathered features
    """
    feat = feat.transpose(0, 2, 3, 1).reshape(feat.shape[0], -1, feat.shape[1])
    return _gather_feat(feat, ind)


class TableCenterNet(Engine):
    """TableCenterNet Onnx loader

    A StarNet backbone feeds a deformable-convolution DLA decoder, followed by six dense heads
    (`hm`, `reg`, `ct2cn`, `cn2ct`, `lc`, `sp`) describing cell centers, corners and their
    logical coordinates. This class decodes those raw head maps and post-processes them into cells.

    Args:
        model_path: path or url to onnx model file
        engine_cfg: configuration for the inference engine
        center_thresh: minimum score for a cell center to be kept
        corner_thresh: minimum score for a corner to be used during relocation
        center_k: maximum number of cell centers
        corner_k: maximum number of corners
        not_relocate: if True, skip the corner-relocation step
        assume_straight_pages: if True, fit straight boxes to the detected cells
        cfg: the configuration dict of the model
        **kwargs: additional arguments to be passed to `Engine`
    """

    def __init__(
        self,
        model_path: str,
        engine_cfg: EngineConfig | None = None,
        center_thresh: float = 0.3,
        corner_thresh: float = 0.3,
        center_k: int = 3000,
        corner_k: int = 5000,
        not_relocate: bool = False,
        assume_straight_pages: bool = True,
        cfg: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(url=model_path, engine_cfg=engine_cfg, **kwargs)

        self.cfg = cfg
        self.center_k = center_k
        self.corner_k = corner_k
        self.assume_straight_pages = assume_straight_pages

        self.postprocessor = TableCenterNetPostProcessor(
            center_thresh=center_thresh,
            corner_thresh=corner_thresh,
            not_relocate=not_relocate,
            assume_straight_pages=assume_straight_pages,
        )

    def _polygons_decode(self, heatmap: np.ndarray, vec: np.ndarray, reg: np.ndarray, k: int):
        """Decode key-points (cell centers or corners) into the four points of a quadrilateral."""
        batch, cat, height, width = heatmap.shape
        # NMS on heatmaps (3x3 max pooling with stride 1, padding 1)
        hmax = maximum_filter(heatmap, size=(1, 1, 3, 3), mode="constant", cval=-np.inf)
        heatmap = heatmap * (hmax == heatmap).astype(heatmap.dtype)
        # Top-K key-points
        k = min(k, height * width)  # never request more points than there are locations
        topk_scores, topk_inds = _topk(heatmap.reshape(batch * cat, -1), k)
        topk_scores = topk_scores.reshape(batch, cat, k)
        topk_inds = topk_inds.reshape(batch, cat, k) % (height * width)
        topk_ys = (topk_inds // width).astype(np.float32)
        topk_xs = (topk_inds % width).astype(np.float32)
        scores, topk_ind = _topk(topk_scores.reshape(batch, -1), k)
        indexes = _gather_feat(topk_inds.reshape(batch, -1, 1), topk_ind).reshape(batch, k)
        ys = _gather_feat(topk_ys.reshape(batch, -1, 1), topk_ind).reshape(batch, k)
        xs = _gather_feat(topk_xs.reshape(batch, -1, 1), topk_ind).reshape(batch, k)

        scores = scores.reshape(batch, k, 1)
        offset = _transpose_and_gather_feat(reg, indexes)
        xs = xs.reshape(batch, k, 1) + offset[:, :, 0:1]
        ys = ys.reshape(batch, k, 1) + offset[:, :, 1:2]
        v = _transpose_and_gather_feat(vec, indexes)
        polygons = np.concatenate(
            [
                xs - v[..., 0:1],
                ys - v[..., 1:2],
                xs - v[..., 2:3],
                ys - v[..., 3:4],
                xs - v[..., 4:5],
                ys - v[..., 5:6],
                xs - v[..., 6:7],
                ys - v[..., 7:8],
            ],
            axis=2,
        )
        return scores, indexes, xs, ys, polygons

    def _decode(self, heads: dict[str, np.ndarray]) -> dict[str, Any]:
        """Decode the raw head maps into cell polygons, scores, logical coordinates and corner points."""
        hm = expit(heads["hm"])
        reg = heads["reg"]
        c_scores, c_ind, _, _, c_poly = self._polygons_decode(hm[:, 0:1], heads["ct2cn"], reg, self.center_k)
        k_scores, k_ind, k_xs, k_ys, k_poly = self._polygons_decode(hm[:, 1:2], heads["cn2ct"], reg, self.corner_k)
        spans = _transpose_and_gather_feat(heads["sp"], c_ind)
        corner_logics = _transpose_and_gather_feat(heads["lc"], k_ind)
        feat_h, feat_w = hm.shape[2], hm.shape[3]

        return {
            "center_polygons": c_poly.astype(np.float32),
            "center_scores": c_scores.squeeze(-1).astype(np.float32),
            "center_spans": spans.astype(np.float32),
            "corner_polygons": k_poly.astype(np.float32),
            "corner_scores": k_scores.squeeze(-1).astype(np.float32),
            "corner_points": np.concatenate([k_xs, k_ys], axis=2).astype(np.float32),
            "corner_logics": corner_logics.astype(np.float32),
            "lc": heads["lc"].astype(np.float32),
            "feat_size": (feat_h, feat_w),
        }

    def _resolve_heads(self, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Map the graph outputs to the head names, by name when available, by export order otherwise"""
        if all(head in outputs for head in HEAD_NAMES):
            return {head: outputs[head] for head in HEAD_NAMES}
        if len(self.output_name) != len(HEAD_NAMES):  # pragma: no cover
            raise ValueError(
                f"expected {len(HEAD_NAMES)} outputs {HEAD_NAMES}, got {len(self.output_name)}: {self.output_name}"
            )
        return {head: outputs[name] for head, name in zip(HEAD_NAMES, self.output_name)}

    def __call__(
        self,
        x: np.ndarray,
        return_model_output: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run the model on a batch of pages

        Args:
            x: batched and normalized pages of shape (N, C, H, W) or (N, H, W, C)
            return_model_output: whether to return the raw head maps
            **kwargs: unused, kept for API consistency

        Returns:
            dict with the postprocessed predictions (and optionally the raw head maps)
        """
        # `run_multi` does not translate the layout, so do it here
        x = shape_translate(x, format="BHWC" if self.tf_exported else "BCHW")
        heads = self._resolve_heads(self.run_multi({self.input_names[0]: x}))

        out: dict[str, Any] = {}

        if return_model_output:
            out["out_map"] = heads

        out["preds"] = self.postprocessor(self._decode(heads))

        return out


def _tablecenternet(
    arch: str,
    model_path: str,
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> TableCenterNet:
    if load_in_8_bit:
        if default_cfgs[arch]["url_8_bit"] is None:
            logger.warning(f"No 8-bit quantized export available for '{arch}'. Loading full precision model...")
        elif "http" in model_path:
            model_path = default_cfgs[arch]["url_8_bit"]
    # Build the model
    return TableCenterNet(model_path, cfg=default_cfgs[arch], engine_cfg=engine_cfg, **kwargs)


def tablecenternet(
    model_path: str = default_cfgs["tablecenternet"]["url"],
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> TableCenterNet:
    """TableCenterNet for table-structure recognition, as described in the official implementation
    `<https://github.com/dreamy-xay/TableCenterNet>`_.

    >>> import numpy as np
    >>> from onnxtr.models import tablecenternet
    >>> model = tablecenternet()
    >>> input_tensor = np.random.rand(1, 3, 1024, 1024)
    >>> out = model(input_tensor)

    Args:
        model_path: path to onnx model file, defaults to url in default_cfgs
        load_in_8_bit: whether to load the the 8-bit quantized model, defaults to False
        engine_cfg: configuration for the inference engine
        **kwargs: keyword arguments of the TableCenterNet architecture

    Returns:
        table structure recognition architecture
    """
    return _tablecenternet("tablecenternet", model_path, load_in_8_bit, engine_cfg, **kwargs)
