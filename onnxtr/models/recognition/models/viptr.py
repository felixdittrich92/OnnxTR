# Copyright (C) 2021-2025, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
from copy import deepcopy
from itertools import groupby
from typing import Any

import numpy as np
from scipy.special import softmax

from onnxtr.utils import VOCABS

from ...engine import Engine, EngineConfig
from ..core import RecognitionPostProcessor

__all__ = ["VIPTR", "viptr_tiny"]

default_cfgs: dict[str, dict[str, Any]] = {
    "viptr_tiny": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 128),
        "vocab": VOCABS["french"],
        "url": "https://github.com/felixdittrich92/OnnxTR/releases/download/v0.6.3/viptr_tiny-499b8015.onnx",
        "url_8_bit": "https://github.com/felixdittrich92/OnnxTR/releases/download/v0.6.3/viptr_tiny-499b8015.onnx",
    },
}


class VIPTRPostProcessor(RecognitionPostProcessor):
    """Postprocess raw prediction of the model (logits) to a list of words using CTC decoding

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    def __init__(self, vocab):
        self.vocab = vocab

    def decode_sequence(self, sequence, vocab):
        return "".join([vocab[int(char)] for char in sequence])

    def ctc_best_path(
        self,
        logits,
        vocab,
        blank=0,
    ):
        """Implements best path decoding as shown by Graves (Dissertation, p63), highly inspired from
        <https://github.com/githubharald/CTCDecoder>`_.

        Args:
            logits: model output, shape: N x T x C
            vocab: vocabulary to use
            blank: index of blank label

        Returns:
            A list of tuples: (word, confidence)
        """
        # Gather the most confident characters, and assign the smallest conf among those to the sequence prob
        probs = softmax(logits, axis=-1).max(axis=-1).min(axis=1)

        # collapse best path (using itertools.groupby), map to chars, join char list to string
        words = [
            self.decode_sequence([k for k, _ in groupby(seq.tolist()) if k != blank], vocab)
            for seq in np.argmax(logits, axis=-1)
        ]

        return list(zip(words, probs.astype(float).tolist()))

    def __call__(self, logits):
        """Performs decoding of raw output with CTC and decoding of CTC predictions
        with label_to_idx mapping dictionnary

        Args:
            logits: raw output of the model, shape (N, C + 1, seq_len)

        Returns:
            A tuple of 2 lists: a list of str (words) and a list of float (probs)

        """
        # Decode CTC
        return self.ctc_best_path(logits=logits, vocab=self.vocab, blank=len(self.vocab))


class VIPTR(Engine):
    """VIPTR Onnx loader

    Args:
        model_path: path or url to onnx model file
        vocab: vocabulary used for encoding
        engine_cfg: configuration for the inference engine
        cfg: configuration dictionary
        **kwargs: additional arguments to be passed to `Engine`
    """

    _children_names: list[str] = ["postprocessor"]

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

        self.postprocessor = VIPTRPostProcessor(self.vocab)

    def __call__(
        self,
        x: np.ndarray,
        return_model_output: bool = False,
    ) -> dict[str, Any]:
        logits = self.run(x)

        out: dict[str, Any] = {}
        if return_model_output:
            out["out_map"] = logits

        # Post-process
        out["preds"] = self.postprocessor(logits)

        return out


def _viptr(
    arch: str,
    model_path: str,
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> VIPTR:
    if load_in_8_bit:
        logging.warning("VIPTR models do not support 8-bit quantization yet. Loading full precision model...")
    kwargs["vocab"] = kwargs.get("vocab", default_cfgs[arch]["vocab"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["vocab"] = kwargs["vocab"]
    _cfg["input_shape"] = kwargs.get("input_shape", default_cfgs[arch]["input_shape"])
    # Patch the url
    model_path = default_cfgs[arch]["url_8_bit"] if load_in_8_bit and "http" in model_path else model_path

    # Build the model
    return VIPTR(model_path, cfg=_cfg, engine_cfg=engine_cfg, **kwargs)


def viptr_tiny(
    model_path: str = default_cfgs["viptr_tiny"]["url"],
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> VIPTR:
    """VIPTR as described in `"A Vision Permutable Extractor for Fast and Efficient
    Scene Text Recognition" <https://arxiv.org/pdf/1507.05717.pdf>`_.

    >>> import numpy as np
    >>> from onnxtr.models import viptr_tiny
    >>> model = viptr_tiny()
    >>> input_tensor = np.random.rand(1, 3, 32, 128)
    >>> out = model(input_tensor)

    Args:
        model_path: path to onnx model file, defaults to url in default_cfgs
        load_in_8_bit: whether to load the the 8-bit quantized model, defaults to False
        engine_cfg: configuration for the inference engine
        **kwargs: keyword arguments of the VIPTR architecture

    Returns:
        text recognition architecture
    """
    return _viptr("viptr_tiny", model_path, load_in_8_bit, engine_cfg, **kwargs)
