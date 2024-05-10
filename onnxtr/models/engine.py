# Copyright (C) 2021-2024, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, List, Union

import numpy as np
import onnxruntime

from onnxtr.utils.data import download_from_url
from onnxtr.utils.geometry import shape_translate


class Engine:
    """Implements an abstract class for the engine of a model

    Args:
    ----
        url: the url to use to download a model if needed
        providers: list of providers to use for inference
        **kwargs: additional arguments to be passed to `download_from_url`
    """

    def __init__(
        self, url: str, providers: List[str] = ["CPUExecutionProvider", "CUDAExecutionProvider"], **kwargs: Any
    ) -> None:
        archive_path = download_from_url(url, cache_subdir="models", **kwargs) if "http" in url else url
        self.runtime = onnxruntime.InferenceSession(archive_path, providers=providers)
        self.runtime_inputs = self.runtime.get_inputs()[0]
        self.tf_exported = int(self.runtime_inputs.shape[-1]) == 3
        self.fixed_batch_size: Union[int, str] = self.runtime_inputs.shape[
            0
        ]  # mostly possible with tensorflow exported models
        self.output_name = [output.name for output in self.runtime.get_outputs()]

    def run(self, inputs: np.ndarray) -> np.ndarray:
        if self.tf_exported:
            inputs = shape_translate(inputs, format="BHWC")  # sanity check
        else:
            inputs = shape_translate(inputs, format="BCHW")
        if isinstance(self.fixed_batch_size, int) and self.fixed_batch_size != 0:  # dynamic batch size is a string
            inputs = np.broadcast_to(inputs, (self.fixed_batch_size, *inputs.shape))
            # combine the results
            logits = np.concatenate(
                [self.runtime.run(self.output_name, {"input": batch})[0] for batch in inputs], axis=0
            )
        else:
            logits = self.runtime.run(self.output_name, {"input": inputs})[0]
        return shape_translate(logits, format="BHWC")
