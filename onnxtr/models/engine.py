# Copyright (C) 2021-2024, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from onnxruntime import (
    ExecutionMode,
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
    get_available_providers,
    get_device,
)

from onnxtr.utils.data import download_from_url
from onnxtr.utils.geometry import shape_translate

__all__ = ["EngineConfig"]


class EngineConfig:
    """Implements a configuration class for the engine of a model

    Args:
    ----
        providers: list of providers to use for inference ref.: https://onnxruntime.ai/docs/execution-providers/
        session_options: configuration for the inference session ref.: https://onnxruntime.ai/docs/api/python/api_summary.html#sessionoptions
    """

    def __init__(
        self,
        providers: Optional[Union[List[Tuple[str, Dict[str, Any]]], List[str]]] = None,
        session_options: Optional[SessionOptions] = None,
    ):
        self._providers = providers or self._init_providers()
        self._session_options = session_options or self._init_sess_opts()

    def _init_providers(self) -> List[Tuple[str, Dict[str, Any]]]:
        providers: Any = [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]
        available_providers = get_available_providers()
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
    def providers(self) -> Union[List[Tuple[str, Dict[str, Any]]], List[str]]:
        return self._providers

    @property
    def session_options(self) -> SessionOptions:
        return self._session_options

    def __repr__(self) -> str:
        return f"EngineConfig(providers={self.providers}"


class Engine:
    """Implements an abstract class for the engine of a model

    Args:
    ----
        url: the url to use to download a model if needed
        engine_cfg: the configuration of the engine
        **kwargs: additional arguments to be passed to `download_from_url`
    """

    def __init__(self, url: str, engine_cfg: Optional[EngineConfig] = None, **kwargs: Any) -> None:
        engine_cfg = engine_cfg if isinstance(engine_cfg, EngineConfig) else EngineConfig()
        archive_path = download_from_url(url, cache_subdir="models", **kwargs) if "http" in url else url
        # Store model path for each model
        self.model_path = archive_path
        self.session_options = engine_cfg.session_options
        self.providers = engine_cfg.providers
        self.runtime = InferenceSession(archive_path, providers=self.providers, sess_options=self.session_options)
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
                [self.runtime.run(self.output_name, {self.runtime_inputs.name: batch})[0] for batch in inputs], axis=0
            )
        else:
            logits = self.runtime.run(self.output_name, {self.runtime_inputs.name: inputs})[0]
        return shape_translate(logits, format="BHWC")
