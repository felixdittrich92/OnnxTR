# Copyright (C) 2021-2026, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
import os
from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np
from onnxruntime import (
    ExecutionMode,
    GraphOptimizationLevel,
    InferenceSession,
    RunOptions,
    SessionOptions,
    get_available_providers,
    get_device,
)
from onnxruntime.capi._pybind_state import set_default_logger_severity

set_default_logger_severity(int(os.getenv("ORT_LOG_SEVERITY_LEVEL", 4)))

from onnxtr.utils.data import download_from_url
from onnxtr.utils.geometry import shape_translate

__all__ = ["EngineConfig", "RunOptionsProvider"]

logger = logging.getLogger(__name__)

RunOptionsProvider: TypeAlias = Callable[[RunOptions], RunOptions]

_ORT_TO_NUMPY_DTYPE: dict[str, Any] = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(bool)": np.bool_,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
}


def _cast_to_ort_dtype(array: np.ndarray, ort_type: str) -> np.ndarray:
    """Cast an array to the dtype expected by the graph input (onnxruntime is strict about this)"""
    dtype = _ORT_TO_NUMPY_DTYPE.get(ort_type)
    if dtype is None:  # pragma: no cover
        return array
    return array if array.dtype == dtype else array.astype(dtype)


class EngineConfig:
    """Implements a configuration class for the engine of a model

    Args:
        providers: list of providers to use for inference ref.: https://onnxruntime.ai/docs/execution-providers/
        session_options: configuration for the inference session ref.: https://onnxruntime.ai/docs/api/python/api_summary.html#sessionoptions
    """

    def __init__(
        self,
        providers: list[tuple[str, dict[str, Any]]] | list[str] | None = None,
        session_options: SessionOptions | None = None,
        run_options_provider: RunOptionsProvider | None = None,
    ):
        self._providers = providers or self._init_providers()
        self._session_options = session_options or self._init_sess_opts()
        self.run_options_provider = run_options_provider

    def _init_providers(self) -> list[tuple[str, dict[str, Any]]]:
        providers: Any = [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]
        available_providers = get_available_providers()
        logger.info(f"Available providers: {available_providers}")
        if "CUDAExecutionProvider" in available_providers and get_device() == "GPU":  # pragma: no cover
            providers.insert(
                0,
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "cudnn_conv_algo_search": "DEFAULT",
                        "do_copy_in_default_stream": True,
                    },
                ),
            )
        elif "CoreMLExecutionProvider" in available_providers:  # pragma: no cover
            providers.insert(0, ("CoreMLExecutionProvider", {}))
        return providers

    def _init_sess_opts(self) -> SessionOptions:
        session_options = SessionOptions()
        session_options.enable_cpu_mem_arena = True
        session_options.execution_mode = ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = -1
        session_options.inter_op_num_threads = -1
        return session_options

    @property
    def providers(self) -> list[tuple[str, dict[str, Any]]] | list[str]:
        return self._providers

    @property
    def session_options(self) -> SessionOptions:
        return self._session_options

    def __repr__(self) -> str:
        return f"EngineConfig(providers={self.providers})"


class Engine:
    """Implements an abstract class for the engine of a model

    Args:
        url: the url to use to download a model if needed
        engine_cfg: the configuration of the engine
        **kwargs: additional arguments to be passed to `download_from_url`
    """

    def __init__(self, url: str, engine_cfg: EngineConfig | None = None, **kwargs: Any) -> None:
        engine_cfg = engine_cfg if isinstance(engine_cfg, EngineConfig) else EngineConfig()
        archive_path = download_from_url(url, cache_subdir="models", **kwargs) if "http" in url else url
        # NOTE: older onnxruntime versions require a string path for windows
        archive_path = rf"{archive_path}"
        # Store model path for each model
        self.model_path = archive_path
        self.session_options = engine_cfg.session_options
        self.providers = engine_cfg.providers
        self.run_options_provider = engine_cfg.run_options_provider
        self.runtime = InferenceSession(archive_path, providers=self.providers, sess_options=self.session_options)
        # NOTE: `runtime_inputs` is kept for single input models, `runtime_input_metas` covers
        # multi-input models (e.g. LW-DETR which additionally consumes a padding mask)
        self.runtime_input_metas = self.runtime.get_inputs()
        self.runtime_inputs = self.runtime_input_metas[0]
        self.input_names = [inp.name for inp in self.runtime_input_metas]
        self.tf_exported = int(self.runtime_inputs.shape[-1]) == 3
        self.fixed_batch_size: int | str = self.runtime_inputs.shape[
            0
        ]  # mostly possible with tensorflow exported models
        self.output_name = [output.name for output in self.runtime.get_outputs()]

    def _run_options(self) -> RunOptions:
        run_options = RunOptions()
        if self.run_options_provider is not None:
            run_options = self.run_options_provider(run_options)
        return run_options

    def run_multi(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run a model with multiple inputs and/or multiple outputs.

        Contrary to `run`, no layout translation is applied: the arrays are fed and returned
        exactly as the graph defines them. This is required for models whose outputs are not
        spatial feature maps (e.g. LW-DETR logits / boxes) or which expose more than one output
        (e.g. the six TableCenterNet head maps).

        Args:
            inputs: mapping from graph input name to the corresponding array

        Returns:
            mapping from graph output name to the corresponding array
        """
        feed_dict = {}
        for meta in self.runtime_input_metas:
            if meta.name not in inputs:
                raise ValueError(f"missing input '{meta.name}' - expected inputs: {self.input_names}")
            feed_dict[meta.name] = _cast_to_ort_dtype(inputs[meta.name], meta.type)

        outputs = self.runtime.run(self.output_name, feed_dict, run_options=self._run_options())
        return dict(zip(self.output_name, outputs))

    def run(self, inputs: np.ndarray) -> np.ndarray:
        run_options = RunOptions()
        if self.run_options_provider is not None:
            run_options = self.run_options_provider(run_options)
        if self.tf_exported:
            inputs = shape_translate(inputs, format="BHWC")  # sanity check
        else:
            inputs = shape_translate(inputs, format="BCHW")
        if isinstance(self.fixed_batch_size, int) and self.fixed_batch_size != 0:  # dynamic batch size is a string
            inputs = np.broadcast_to(inputs, (self.fixed_batch_size, *inputs.shape))
            # combine the results
            logits = np.concatenate(
                [
                    self.runtime.run(self.output_name, {self.runtime_inputs.name: batch}, run_options=run_options)[0]
                    for batch in inputs
                ],
                axis=0,
            )
        else:
            logits = self.runtime.run(self.output_name, {self.runtime_inputs.name: inputs}, run_options=run_options)[0]
        return shape_translate(logits, format="BHWC")
