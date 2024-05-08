# Copyright (C) 2021-2024, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
from scipy.special import softmax

from onnxtr.utils import VOCABS

from ...engine import Engine
from ..core import RecognitionPostProcessor

__all__ = ["SAR", "sar_resnet31"]

default_cfgs: Dict[str, Dict[str, Any]] = {
    "sar_resnet31": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 128),
        "vocab": VOCABS["french"],
        "url": "https://doctr-static.mindee.com/models?id=v0.7.0/sar_resnet31-9a1deedf.pt&src=0",
    },
}


class SAR(Engine):
    """SAR Onnx loader

    Args:
    ----
        model_path: path to onnx model file
        vocab: vocabulary used for encoding
        cfg: dictionary containing information about the model
    """

    def __init__(
        self,
        model_path: str,
        vocab: str,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(url=model_path)
        self.vocab = vocab
        self.cfg = cfg
        self.postprocessor = SARPostProcessor(self.vocab)

    def __call__(
        self,
        x: np.ndarray,
        return_model_output: bool = False,
    ) -> Dict[str, Any]:
        logits = self.session.run(x)

        out: Dict[str, Any] = {}
        if return_model_output:
            out["out_map"] = logits

        out["preds"] = self.postprocessor(logits)

        return out


class SARPostProcessor(RecognitionPostProcessor):
    """Post processor for SAR architectures

    Args:
    ----
        embedding: string containing the ordered sequence of supported characters
    """

    def __init__(
        self,
        vocab: str,
    ) -> None:
        super().__init__(vocab)
        self._embedding = list(self.vocab) + ["<eos>"]

    def __call__(self, logits):
        # compute pred with argmax for attention models
        out_idxs = np.argmax(logits, axis=-1)
        # N x L
        probs = np.take_along_axis(softmax(logits, axis=-1), out_idxs[..., None], axis=-1).squeeze(-1)
        # Take the minimum confidence of the sequence
        probs = np.min(probs, axis=1)

        word_values = [
            "".join(self._embedding[idx] for idx in encoded_seq).split("<eos>")[0] for encoded_seq in out_idxs
        ]

        return list(zip(word_values, np.clip(probs, 0, 1).tolist()))


def _sar(
    arch: str,
    model_path: str,
    **kwargs: Any,
) -> SAR:
    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["vocab"] = kwargs.get("vocab", _cfg["vocab"])
    _cfg["input_shape"] = kwargs.get("input_shape", _cfg["input_shape"])

    kwargs["vocab"] = _cfg["vocab"]
    kwargs["input_shape"] = _cfg["input_shape"]

    # Build the model
    return SAR(model_path, cfg=_cfg, **kwargs)


def sar_resnet31(model_path: str = default_cfgs["sar_resnet31"], **kwargs: Any) -> SAR:
    """SAR with a resnet-31 feature extractor as described in `"Show, Attend and Read:A Simple and Strong
    Baseline for Irregular Text Recognition" <https://arxiv.org/pdf/1811.00751.pdf>`_.

    >>> import numpy as np
    >>> from onnxtr.models import sar_resnet31
    >>> model = sar_resnet31()
    >>> input_tensor = np.random.rand(1, 3, 32, 128)
    >>> out = model(input_tensor)

    Args:
    ----
        model_path: path to onnx model file, defaults to url in default_cfgs
        **kwargs: keyword arguments of the SAR architecture

    Returns:
    -------
        text recognition architecture
    """
    return _sar("sar_resnet31", model_path, **kwargs)
