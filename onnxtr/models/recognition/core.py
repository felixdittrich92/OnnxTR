# Copyright (C) 2021-2025, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from onnxtr.utils.repr import NestedObject

__all__ = ["RecognitionPostProcessor"]


class RecognitionPostProcessor(NestedObject):
    """Abstract class to postprocess the raw output of the model

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    def __init__(
        self,
        vocab: str,
    ) -> None:
        self.vocab = vocab
        self._embedding = list(self.vocab) + ["<eos>"]

    def extra_repr(self) -> str:
        return f"vocab_size={len(self.vocab)}"
