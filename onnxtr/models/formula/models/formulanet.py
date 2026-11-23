# Copyright (C) 2021-2026, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np
from scipy.special import softmax

from onnxtr.utils import LATEX_TOKENS
from onnxtr.utils.repr import NestedObject

from ..._utils import (
    ConfidenceAggregation,
    _confidence_aggregation_repr,
    _resolve_confidence_aggregation,
    aggregate_confidence,
)
from ...engine import Engine, EngineConfig

__all__ = ["FormulaNet", "formulanet"]

logger = logging.getLogger(__name__)

default_cfgs: dict[str, dict[str, Any]] = {
    "formulanet": {
        "mean": (0.7931, 0.7931, 0.7931),
        "std": (0.1738, 0.1738, 0.1738),
        "input_shape": (3, 256, 768),
        "vocab": LATEX_TOKENS,
        "url": "https://github.com/felixdittrich92/OnnxTR/releases/download/v0.9.0/formulanet-3d2c732e.onnx",
        "url_8_bit": None,
    },
}


class FormulaNet(Engine):
    """FormulaNet Onnx loader

    The exported graph runs the HGNetV2 encoder and the greedy decoding of the transformer decoder: it takes a batch of
    formula images and returns the logits of the decoded tokens, of shape (N, L, vocab_size + 1). The number of
    decoding steps L is the one of the export.

    Args:
        model_path: path or url to onnx model file
        vocab: the list of LaTeX tokens (without EOS, SOS, PAD)
        engine_cfg: configuration for the inference engine
        cfg: dictionary containing information about the model
        confidence_aggregation: aggregation method of the token probabilities into the formula confidence
        **kwargs: additional arguments to be passed to `Engine`
    """

    _release_gpu_memory_after_run = True

    def __init__(
        self,
        model_path: str,
        vocab: Sequence[str],
        engine_cfg: EngineConfig | None = None,
        cfg: dict[str, Any] | None = None,
        confidence_aggregation: ConfidenceAggregation = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(url=model_path, engine_cfg=engine_cfg, **kwargs)

        self.vocab = list(vocab)
        self.cfg = cfg

        self.postprocessor = FormulaNetPostProcessor(vocab=self.vocab, confidence_aggregation=confidence_aggregation)

    def __call__(
        self,
        x: np.ndarray,
        return_model_output: bool = False,
    ) -> dict[str, Any]:
        """Call function

        Args:
            x: batched and normalized formula images
            return_model_output: if True, return logits

        Returns:
            A dictionnary containing eventually logits and predictions.
        """
        logits = self.run(x)
        out: dict[str, Any] = {}

        if return_model_output:
            out["out_map"] = logits

        out["preds"] = self.postprocessor(logits)

        return out


class FormulaNetPostProcessor(NestedObject):
    """Turns the token scores of a formula recognition model into LaTeX strings

    Args:
        vocab: the list of LaTeX tokens of the model
        confidence_aggregation: aggregation method of the token probabilities into the formula confidence:
            "mean", "min", "max", "median", "geometric_mean", "harmonic_mean" or a callable
    """

    def __init__(
        self,
        vocab: Sequence[str],
        confidence_aggregation: ConfidenceAggregation = "mean",
    ) -> None:
        _resolve_confidence_aggregation(confidence_aggregation)
        self.vocab = list(vocab)
        self.confidence_aggregation = confidence_aggregation

    def extra_repr(self) -> str:
        return (
            f"vocab_size={len(self.vocab)}, "
            f"confidence_aggregation={_confidence_aggregation_repr(self.confidence_aggregation)}"
        )

    def __call__(self, logits: np.ndarray) -> list[tuple[str, float]]:
        out_idxs = np.argmax(logits, axis=-1)
        probs = softmax(logits.astype(np.float32), axis=-1).max(axis=-1)
        eos = len(self.vocab)

        results: list[tuple[str, float]] = []
        for idxs, token_probs in zip(out_idxs, probs):
            ends = np.flatnonzero(idxs == eos)
            length = int(ends[0]) if ends.size > 0 else len(idxs)
            latex = " ".join(self.vocab[idx] for idx in idxs[:length])
            results.append((latex, aggregate_confidence(token_probs[:length], self.confidence_aggregation)))
        return results


def _formulanet(
    arch: str,
    model_path: str,
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> FormulaNet:
    if model_path is None:  # pragma: no cover
        raise ValueError(f"no pretrained ONNX export is available for '{arch}' yet")
    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["input_shape"] = kwargs.get("input_shape", _cfg["input_shape"])
    _cfg["vocab"] = list(kwargs.get("vocab", _cfg["vocab"]))

    kwargs["vocab"] = _cfg["vocab"]
    kwargs.pop("input_shape", None)
    # Patch the url
    if load_in_8_bit:
        if default_cfgs[arch]["url_8_bit"] is None:
            logger.warning(f"No 8-bit quantized export available for '{arch}'. Loading full precision model...")
        elif model_path.startswith(("http://", "https://")):
            model_path = default_cfgs[arch]["url_8_bit"]

    return FormulaNet(model_path, cfg=_cfg, engine_cfg=engine_cfg, **kwargs)


def formulanet(
    model_path: str = default_cfgs["formulanet"]["url"],
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> FormulaNet:
    """FormulaNet: a HGNetV2-B4 encoder with a transformer decoder predicting LaTeX tokens, following
    `"PP-FormulaNet: Bridging Accuracy and Efficiency in Advanced Formula Recognition"
    <https://arxiv.org/abs/2503.18382>`_.

    >>> import numpy as np
    >>> from onnxtr.models import formulanet
    >>> model = formulanet()
    >>> input_tensor = np.random.rand(1, 3, 256, 768)
    >>> out = model(input_tensor)

    Args:
        model_path: path to onnx model file, defaults to url in default_cfgs
        load_in_8_bit: whether to load the the 8-bit quantized model, defaults to False
        engine_cfg: configuration for the inference engine
        **kwargs: keyword arguments of the FormulaNet architecture

    Returns:
        formula recognition architecture
    """
    return _formulanet("formulanet", model_path, load_in_8_bit, engine_cfg, **kwargs)
