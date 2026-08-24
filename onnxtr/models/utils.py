# Copyright (C) 2021-2026, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
from anyascii import anyascii

__all__ = ["WhitelistHandle", "add_whitelist"]

logger = logging.getLogger(__name__)

# Initializer names of the final vocabulary projection, per recognition architecture.
# Only used to disambiguate when several initializers could match; the search falls back to
# shape-based detection for anything not listed here.
_PROJECTION_HINTS: tuple[str, ...] = ("head.weight", "linear", "output_dense", "head", "fc", "classifier")


class WhitelistHandle:
    """Removable registration returned by :func:`add_whitelist`.

    Call :meth:`remove` to restore the model's original, unconstrained decoding. The
    handle can also be used as a context manager, in which case the whitelist is removed
    on exit.
    """

    def __init__(self, restore: list[tuple[Any, Any]]) -> None:
        self._restore = restore

    def remove(self) -> None:
        """Restore the original, unconstrained decoding"""
        for model, postprocessor in self._restore:
            model.postprocessor = postprocessor
        self._restore = []

    def __enter__(self) -> "WhitelistHandle":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.remove()


class _ConstrainedPostProcessor:
    """Wraps a recognition post-processor and constrains the logits before they are decoded.

    docTR enforces the whitelist with a forward hook on the model's final projection layer.
    An ONNX graph exposes no such hook, but every OnnxTR recognition architecture funnels its
    raw logits through ``self.postprocessor(logits)``, so wrapping that call is the equivalent
    single choke point - and it is just as removable.
    """

    def __init__(self, postprocessor: Any, keep: np.ndarray, src: np.ndarray, dst: np.ndarray) -> None:
        self.postprocessor = postprocessor
        self.keep = keep
        self.src = src
        self.dst = dst

    def __call__(self, logits: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        return self.postprocessor(self._constrain(logits), *args, **kwargs)

    def _constrain(self, logits: np.ndarray) -> np.ndarray:
        output = np.array(logits, dtype=np.float32, copy=True)
        if self.src.size:
            # move each forbidden character's score onto its nearest allowed character.
            # Several forbidden characters can share a target, so the maximum is taken.
            for target in np.unique(self.dst):
                sources = self.src[self.dst == target]
                output[..., target] = np.maximum(output[..., target], output[..., sources].max(axis=-1))
        output[..., ~self.keep] = -np.inf
        return output

    def __getattr__(self, name: str) -> Any:
        # stay transparent for callers reaching for e.g. `postprocessor.vocab`
        return getattr(self.__dict__["postprocessor"], name)


def _recognition_models(model: Any) -> list[Any]:
    """Collect the recognition model(s) a whitelist should be applied to

    Accepts an ``ocr_predictor``, a ``recognition_predictor`` or a bare recognition model.
    """
    # a recognition model itself
    if hasattr(model, "cfg") and isinstance(getattr(model, "cfg", None), dict) and "vocab" in (model.cfg or {}):
        return [model]
    reco_predictor = getattr(model, "reco_predictor", model)
    reco_model = getattr(reco_predictor, "model", None)
    if reco_model is None:
        raise TypeError(
            "Expected an ocr_predictor, recognition_predictor or a recognition model, but could "
            f"not find a recognition model on {type(model).__name__}."
        )
    return [reco_model]


def _anyascii_nearest_map(vocab: str, allowed: set[str]) -> dict[str, str]:
    """Map each forbidden character to the visually closest allowed one via transliteration.

    Uses `anyascii` to fold characters to their ASCII form (e.g. `ä -> a`, `ł -> l`,
    Cyrillic `а -> a`); a forbidden character is mapped to an allowed character sharing the
    same ASCII form. Forbidden characters without such a match are left unmapped (they fall
    back to plain masking).
    """
    by_translit: dict[str, str] = {}
    for char in vocab:
        if char not in allowed:
            continue
        key = anyascii(char)
        current = by_translit.get(key)
        # Prefer a pure-ASCII allowed character as the canonical target for a given form.
        if current is None or (char == anyascii(char) and current != anyascii(current)):
            by_translit[key] = char

    mapping: dict[str, str] = {}
    for char in vocab:
        if char in allowed:
            continue
        form = anyascii(char)
        target = by_translit.get(form) or by_translit.get(form.lower()) or by_translit.get(form[:1])
        if target is not None:
            mapping[char] = target
    return mapping


def _projection_weights(model: Any, out_features: int) -> np.ndarray:
    """Recover the final vocabulary projection matrix from the ONNX graph

    docTR reads `projection.weight` off the `nn.Linear`. The ONNX graph has no modules, but the
    projection survives as an initializer whose shape carries the vocabulary dimension, so it can
    be located by shape and returned oriented as (out_features, hidden).

    Args:
        model: the recognition model
        out_features: width of the logits, i.e. vocabulary size plus the special tokens

    Returns:
        the projection matrix of shape (out_features, hidden)
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        raise ImportError(
            "mapping='weights' reads the projection matrix out of the ONNX graph, which requires "
            "the `onnx` package. Install it with `pip install onnx`, or use the default "
            "mapping='anyascii'."
        )

    graph = onnx.load(str(model.model_path)).graph
    candidates: list[tuple[str, np.ndarray]] = []
    for initializer in graph.initializer:
        array = numpy_helper.to_array(initializer)
        if array.ndim == 2 and out_features in array.shape:
            candidates.append((initializer.name, array))

    if not candidates:
        raise RuntimeError(f"Could not locate the vocabulary projection in the ONNX graph of {type(model).__name__}.")

    # prefer an initializer whose name looks like the projection, else the last one in graph
    # order (the projection sits at the very end of the network)
    name, weights = candidates[-1]
    for hint in _PROJECTION_HINTS:
        match = [(n, a) for n, a in candidates if hint in n.lower()]
        if match:
            name, weights = match[0]
            break

    logger.debug(f"Using ONNX initializer '{name}' {weights.shape} as the vocabulary projection")
    # orient as (out_features, hidden); a MatMul initializer is stored transposed
    return weights if weights.shape[0] == out_features else weights.T


def _weights_nearest_map(vocab: str, allowed: set[str], projection: np.ndarray) -> dict[str, str]:
    """Map each forbidden character to the allowed one whose projection weights are most similar.

    This uses the model's own learned representation: the nearest allowed character is the one
    the model most confuses the forbidden character with (cosine similarity of the projection
    weight rows).
    """
    vocab_size = len(vocab)
    rows = projection[:vocab_size].astype(np.float32)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    rows = rows / np.clip(norms, 1e-12, None)
    allowed_idx = [i for i, char in enumerate(vocab) if char in allowed]
    forbidden_idx = [i for i, char in enumerate(vocab) if char not in allowed]
    if not allowed_idx or not forbidden_idx:
        return {}
    similarity = rows[forbidden_idx] @ rows[allowed_idx].T
    nearest = similarity.argmax(axis=1)
    return {vocab[forbidden_idx[k]]: vocab[allowed_idx[int(nearest[k])]] for k in range(len(forbidden_idx))}


def _keep_and_reassign(
    vocab: str, allowed: set[str], out_features: int, char_map: dict[str, str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the keep mask and the (forbidden -> allowed) index arrays for one projection."""
    vocab_size = len(vocab)
    keep = np.zeros(out_features, dtype=bool)
    for idx, char in enumerate(vocab):
        keep[idx] = char in allowed
    # NOTE: every column past the vocabulary is a special token (CTC blank / attention <eos>,
    # and for MASTER also <sos> and <pad>). All of them are kept so decoding still terminates.
    keep[vocab_size:] = True

    position = {char: idx for idx, char in enumerate(vocab)}
    src, dst = [], []
    for forbidden_char, allowed_char in char_map.items():
        src_idx, dst_idx = position.get(forbidden_char), position.get(allowed_char)
        # only reassign genuinely-forbidden characters onto genuinely-allowed ones
        if src_idx is not None and dst_idx is not None and not keep[src_idx] and allowed_char in allowed:
            src.append(src_idx)
            dst.append(dst_idx)
    return keep, np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)


def add_whitelist(
    model: Any,
    vocabs: str | Iterable[str],
    *,
    strategy: str = "mask",
    mapping: str | dict[str, str] | None = None,
    verbose: bool = False,
) -> WhitelistHandle:
    """Restrict a recognition model so it can only predict a subset of its vocabulary.

    The whitelist is enforced on the raw logits, right before they are decoded. docTR uses a
    forward hook on the final projection layer; an ONNX graph has no hooks, so OnnxTR wraps the
    recognition post-processor instead - the one point every architecture routes its logits
    through. The sequence terminator (CTC `blank` / attention `<eos>`) and any other special
    token are always kept so decoding still terminates. It works with every recognition
    architecture and with any predictor wrapping one (`ocr_predictor`, `recognition_predictor`).

    NOTE: for the autoregressive architectures (SAR, MASTER, PARSeq) the decoding loop lives
    inside the ONNX graph, so the constraint applies to the emitted logits but cannot influence
    what the graph feeds back to itself mid-word. For the CTC architectures (CRNN, VIPTR) and the
    parallel decoders (ViTSTR) this is exactly equivalent to docTR's behaviour.

    Two strategies are available:

    - `"mask"` (default) simply forbids the characters outside the whitelist.
    - `"nearest"` additionally folds each forbidden character onto the closest allowed one before
      masking (so e.g. `ä` folds onto `a`).

    A whitelist can only restrict a model to characters it already knows: characters that are
    not part of the model's own vocabulary are silently ignored.

    >>> from onnxtr.models import ocr_predictor
    >>> from onnxtr.models.utils import add_whitelist
    >>> from onnxtr.utils.vocabs import VOCABS
    >>> predictor = ocr_predictor()
    >>> handle = add_whitelist(predictor, [VOCABS["polish"], VOCABS["german"]])
    >>> # ... run the predictor; only Polish/German characters can be predicted ...
    >>> handle.remove()  # restore the original, unconstrained decoding

    Args:
        model: an `ocr_predictor`, `recognition_predictor`, or a recognition model.
        vocabs: a vocabulary string (e.g. `VOCABS["german"]`) or an iterable of vocabulary
            strings (e.g. `[VOCABS["polish"], VOCABS["german"]]`) whose characters are allowed.
        strategy: `"mask"` (default) to drop forbidden characters, or `"nearest"` to fold
            them onto the closest allowed character.
        mapping: only used when `"strategy="nearest""`. `None` or `"anyascii"` builds the
            forbidden-to-allowed map by transliteration (the default); `"weights"` derives it
            from the projection weights read out of the ONNX graph (the model's own confusions);
            a `dict` of `{forbidden_char: allowed_char}` overrides specific characters on top of
            the transliteration map.
        verbose: if True, log how many characters were kept, forbidden and reassigned per model.

    Returns:
        a :class:`WhitelistHandle`; call its :meth:`~WhitelistHandle.remove` method to restore
        the original, unconstrained decoding.
    """
    if strategy not in {"mask", "nearest"}:
        raise ValueError(f"Unknown strategy {strategy!r}; expected 'mask' or 'nearest'.")
    if strategy == "mask" and mapping is not None:
        raise ValueError("The 'mapping' argument is only used with strategy='nearest'.")
    if isinstance(mapping, str) and mapping not in {"anyascii", "weights"}:
        raise ValueError(f"Unknown mapping {mapping!r}; expected 'anyascii', 'weights', a dict or None.")
    if mapping is not None and not isinstance(mapping, (str, dict)):
        raise ValueError("The 'mapping' argument must be None, 'anyascii', 'weights' or a dict.")

    allowed = set(vocabs) if isinstance(vocabs, str) else {char for vocab in vocabs for char in vocab}

    restore: list[tuple[Any, Any]] = []
    for reco_model in _recognition_models(model):
        vocab: str = reco_model.cfg["vocab"]
        vocab_size = len(vocab)
        if not any(char in allowed for char in vocab):
            raise ValueError(
                "The whitelist shares no character with the model's vocabulary; the model would "
                "be unable to predict anything."
            )

        out_features = _logits_width(reco_model, vocab_size)

        char_map: dict[str, str] = {}
        if strategy == "nearest":
            if mapping == "weights":
                char_map = _weights_nearest_map(vocab, allowed, _projection_weights(reco_model, out_features))
            else:
                char_map = _anyascii_nearest_map(vocab, allowed)
                if isinstance(mapping, dict):
                    char_map = {**char_map, **mapping}

        keep, src, dst = _keep_and_reassign(vocab, allowed, out_features, char_map)

        restore.append((reco_model, reco_model.postprocessor))
        reco_model.postprocessor = _ConstrainedPostProcessor(reco_model.postprocessor, keep, src, dst)

        if verbose:  # pragma: no cover
            kept = sum(char in allowed for char in vocab)
            logger.info(
                f"add_whitelist: {type(reco_model).__name__} - kept {kept}/{vocab_size} vocabulary "
                f"characters, forbade {vocab_size - kept}"
                + (f", reassigned {src.size} to a nearest allowed character." if strategy == "nearest" else ".")
            )

    return WhitelistHandle(restore)


def _logits_width(model: Any, vocab_size: int) -> int:
    """Width of the model's logits, i.e. the vocabulary plus its special tokens

    Read from the graph's output metadata when it is static (all architectures declare it), and
    otherwise assumed to be the vocabulary plus a single terminator.
    """
    for output in model.runtime.get_outputs():
        last_dim = output.shape[-1] if output.shape else None
        if isinstance(last_dim, int) and last_dim >= vocab_size:
            return last_dim
    return vocab_size + 1  # pragma: no cover
