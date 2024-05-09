# Copyright (C) 2021-2024, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any

import numpy as np
import onnxruntime

from onnxtr.utils.data import download_from_url
from onnxtr.utils.geometry import shape_translate


class Engine:
    """Implements an abstract class for the engine of a model

    Args:
    ----
        url: the url to use to download a model if needed
        **kwargs: additional arguments to be passed to `download_from_url`
    """

    def __init__(self, url: str, **kwargs: Any) -> None:
        archive_path = download_from_url(url, cache_subdir="models", **kwargs) if "http" in url else url
        self.runtime = onnxruntime.InferenceSession(
            archive_path, providers=["CPUExecutionProvider", "CUDAExecutionProvider"]
        )

    def run(self, inputs: np.ndarray) -> np.ndarray:
        inputs = shape_translate(inputs, format="BCHW")
        logits = self.runtime.run(["logits"], {"input": inputs})[0]
        return shape_translate(logits, format="BHWC")
