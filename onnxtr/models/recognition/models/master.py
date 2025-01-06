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

__all__ = ["MASTER", "master"]


default_cfgs: dict[str, dict[str, Any]] = {
    "master": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 128),
        "vocab": VOCABS["french"],
        "url": "https://github.com/felixdittrich92/OnnxTR/releases/download/v0.0.1/master-b1287fcd.onnx",
        "url_8_bit": "https://github.com/felixdittrich92/OnnxTR/releases/download/v0.1.2/master_dynamic_8_bit-d8bd8206.onnx",
    },
}


class MASTER(Engine):
    """MASTER Onnx loader

    Args:
        model_path: path or url to onnx model file
        vocab: vocabulary, (without EOS, SOS, PAD)
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

        self.postprocessor = MASTERPostProcessor(vocab=self.vocab)

    def __call__(
        self,
        x: np.ndarray,
        return_model_output: bool = False,
    ) -> dict[str, Any]:
        """Call function

        Args:
            x: images
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


class MASTERPostProcessor(RecognitionPostProcessor):
    """Post-processor for the MASTER model

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    def __init__(
        self,
        vocab: str,
    ) -> None:
        super().__init__(vocab)
        self._embedding = list(vocab) + ["<eos>"] + ["<sos>"] + ["<pad>"]

    def __call__(self, logits: np.ndarray) -> list[tuple[str, float]]:
        # compute pred with argmax for attention models
        out_idxs = np.argmax(logits, axis=-1)
        # N x L
        probs = np.take_along_axis(softmax(logits, axis=-1), out_idxs[..., None], axis=-1).squeeze(-1)
        # Take the minimum confidence of the sequence
        probs = np.min(probs, axis=1)

        word_values = [
            "".join(self._embedding[idx] for idx in encoded_seq).split("<eos>")[0] for encoded_seq in out_idxs
        ]

        return list(zip(word_values, np.clip(probs, 0, 1).astype(float).tolist()))


def _master(
    arch: str,
    model_path: str,
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> MASTER:
    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["input_shape"] = kwargs.get("input_shape", _cfg["input_shape"])
    _cfg["vocab"] = kwargs.get("vocab", _cfg["vocab"])

    kwargs["vocab"] = _cfg["vocab"]
    # Patch the url
    model_path = default_cfgs[arch]["url_8_bit"] if load_in_8_bit and "http" in model_path else model_path

    return MASTER(model_path, cfg=_cfg, engine_cfg=engine_cfg, **kwargs)


def master(
    model_path: str = default_cfgs["master"]["url"],
    load_in_8_bit: bool = False,
    engine_cfg: EngineConfig | None = None,
    **kwargs: Any,
) -> MASTER:
    """MASTER as described in paper: <https://arxiv.org/pdf/1910.02562.pdf>`_.

    >>> import numpy as np
    >>> from onnxtr.models import master
    >>> model = master()
    >>> input_tensor = np.random.rand(1, 3, 32, 128)
    >>> out = model(input_tensor)

    Args:
        model_path: path to onnx model file, defaults to url in default_cfgs
        load_in_8_bit: whether to load the the 8-bit quantized model, defaults to False
        engine_cfg: configuration for the inference engine
        **kwargs: keywoard arguments passed to the MASTER architecture

    Returns:
        text recognition architecture
    """
    return _master("master", model_path, load_in_8_bit, engine_cfg, **kwargs)
