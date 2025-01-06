# Copyright (C) 2021-2025, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from typing import Any

import numpy as np
from scipy.special import softmax

from onnxtr.utils import VOCABS

from ...engine import Engine, EngineConfig
from ..core import RecognitionPostProcessor

__all__ = ["PARSeq", "parseq"]

default_cfgs: dict[str, dict[str, Any]] = {
    "parseq": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 128),
        "vocab": VOCABS["french"],
        "url": "https://github.com/felixdittrich92/OnnxTR/releases/download/v0.0.1/parseq-00b40714.onnx",
        "url_8_bit": "https://github.com/felixdittrich92/OnnxTR/releases/download/v0.1.2/parseq_dynamic_8_bit-5b04d9f7.onnx",
    },
}


class PARSeq(Engine):
    """PARSeq Onnx loader

    Args:
        model_path: path to onnx model file
        vocab: vocabulary used for encoding
        engine_cfg: configuration for the inference engine
        cfg: dictionary containing information about the model
        **kwargs: additional arguments to be passed to `Engine`
    """

    def __init__(
        self,
        model_path: str,
        vocab: str,
        engine_cfg: EngineConfig | None = None,
        cfg: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(url=model_path, engine_cfg=engine_cfg, **kwargs)

        self.vocab = vocab
        self.cfg = cfg

        self.postprocessor = PARSeqPostProcessor(vocab=self.vocab)

    def __call__(
        self,
        x: np.ndarray,
        return_model_output: bool = False,
    ) -> dict[str, Any]:
        logits = self.run(x)
        out: dict[str, Any] = {}

        if return_model_output:
            out["out_map"] = logits

        out["preds"] = self.postprocessor(logits)
        return out


class PARSeqPostProcessor(RecognitionPostProcessor):
    """Post processor for PARSeq architecture

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    def __init__(
        self,
        vocab: str,
    ) -> None:
        super().__init__(vocab)
        self._embedding = list(vocab) + ["<eos>", "<sos>", "<pad>"]

    def __call__(self, logits):
        # compute pred with argmax for attention models
        out_idxs = np.argmax(logits, axis=-1)
        preds_prob = softmax(logits, axis=-1).max(axis=-1)

        word_values = [
            "".join(self._embedding[idx] for idx in encoded_seq).split("<eos>")[0] for encoded_seq in out_idxs
        ]
        # compute probabilties for each word up to the EOS token
        probs = [
            preds_prob[i, : len(word)].clip(0, 1).mean().astype(float) if word else 0.0
            for i, word in enumerate(word_values)
        ]

        return list(zip(word_values, probs))


def _parseq(
    arch: str,
    model_path: str,
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> PARSeq:
    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["vocab"] = kwargs.get("vocab", _cfg["vocab"])
    _cfg["input_shape"] = kwargs.get("input_shape", _cfg["input_shape"])

    kwargs["vocab"] = _cfg["vocab"]
    # Patch the url
    model_path = default_cfgs[arch]["url_8_bit"] if load_in_8_bit and "http" in model_path else model_path

    # Build the model
    return PARSeq(model_path, cfg=_cfg, engine_cfg=engine_cfg, **kwargs)


def parseq(
    model_path: str = default_cfgs["parseq"]["url"],
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> PARSeq:
    """PARSeq architecture from
    `"Scene Text Recognition with Permuted Autoregressive Sequence Models" <https://arxiv.org/pdf/2207.06966>`_.

    >>> import numpy as np
    >>> from onnxtr.models import parseq
    >>> model = parseq()
    >>> input_tensor = np.random.rand(1, 3, 32, 128)
    >>> out = model(input_tensor)

    Args:
        model_path: path to onnx model file, defaults to url in default_cfgs
        load_in_8_bit: whether to load the the 8-bit quantized model, defaults to False
        engine_cfg: configuration for the inference engine
        **kwargs: keyword arguments of the PARSeq architecture

    Returns:
        text recognition architecture
    """
    return _parseq("parseq", model_path, load_in_8_bit, engine_cfg, **kwargs)
