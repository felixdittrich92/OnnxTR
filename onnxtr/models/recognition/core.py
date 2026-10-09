# Copyright (C) 2021-2026, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from onnxtr.models._utils import (
    ConfidenceAggregation,
    _confidence_aggregation_repr,
    _resolve_confidence_aggregation,
)
from onnxtr.utils.repr import NestedObject

__all__ = ["RecognitionPostProcessor"]


class RecognitionPostProcessor(NestedObject):
    """Abstract class to postprocess the raw output of the model

    Args:
        vocab: string containing the ordered sequence of supported characters
        confidence_aggregation: aggregation method of the character probabilities into the word confidence:
            "mean", "min", "max", "median", "geometric_mean", "harmonic_mean" or a callable
    """

    def __init__(
        self,
        vocab: str,
        confidence_aggregation: ConfidenceAggregation = "mean",
    ) -> None:
        _resolve_confidence_aggregation(confidence_aggregation)
        self.vocab = vocab
        self.confidence_aggregation = confidence_aggregation
        self._embedding = list(self.vocab) + ["<eos>"]

    def extra_repr(self) -> str:
        return (
            f"vocab_size={len(self.vocab)}, "
            f"confidence_aggregation={_confidence_aggregation_repr(self.confidence_aggregation)}"
        )
