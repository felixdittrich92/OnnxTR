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
    OrtAllocatorType,
    OrtArenaCfg,
    OrtMemoryInfo,
    OrtMemType,
    RunOptions,
    SessionOptions,
    create_and_register_allocator_v2,
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


def _provider_name(provider: tuple[str, dict[str, Any]] | str) -> str:
    return provider[0] if isinstance(provider, (tuple, list)) else provider


# Provider fallbacks already reported, so that a pipeline loading several models warns only once
_REPORTED_FALLBACKS: set[tuple[str, ...]] = set()


def _warn_on_provider_fallback(requested: list[tuple[str, dict[str, Any]]] | list[str], active: list[str]) -> list[str]:
    """Warn when onnxruntime could not enable some of the requested execution providers

    onnxruntime silently falls back to the next provider (ultimately the CPU) when one cannot be loaded, e.g. when
    the CUDA / cuDNN libraries matching the installed `onnxruntime-gpu` are missing. Since its own logs are muted
    by default (cf. `ORT_LOG_SEVERITY_LEVEL`), this would otherwise go unnoticed: the inference just runs on the CPU.

    Args:
        requested: the providers passed to the inference session
        active: the providers actually enabled by the inference session

    Returns:
        the names of the requested providers which are not enabled
    """
    requested_names = [_provider_name(p) for p in requested]
    missing = [name for name in requested_names if name not in active]
    key = tuple(missing)
    if missing and key not in _REPORTED_FALLBACKS:
        _REPORTED_FALLBACKS.add(key)
        hint = ""
        if any(name in ("CUDAExecutionProvider", "TensorrtExecutionProvider") for name in missing):
            hint = (
                " Make sure the CUDA and cuDNN versions required by your onnxruntime-gpu version are installed and "
                'on the library path (e.g. `pip install "onnxruntime-gpu[cuda,cudnn]"`), cf. '
                "https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements."
            )
        logger.warning(
            f"onnxruntime could not enable the requested execution provider(s) {missing} and falls back to "
            f"{active}.{hint} Set the environment variable ORT_LOG_SEVERITY_LEVEL=2 to see the onnxruntime logs."
        )
    return missing


# CUDA devices on which a memory arena shared by all the inference sessions of the process is registered (True), or
# could not be registered (False)
_SHARED_CUDA_ARENAS: dict[int, bool] = {}


def _register_shared_cuda_arena(device_id: int) -> bool:
    """Register (once per process) a CUDA memory arena shared by the sessions which opt in for it

    By default, each inference session owns its memory arena, sized after its own peak usage and never released: the
    GPU memory footprint of a pipeline is then the sum of the peaks of all its models (detection, recognition, layout,
    table, formula, ...), although they run one after the other. With a shared arena, it is closer to the largest one.

    Args:
        device_id: the CUDA device

    Returns:
        whether the shared arena is available
    """
    if device_id in _SHARED_CUDA_ARENAS:
        return _SHARED_CUDA_ARENAS[device_id]
    try:
        mem_info = OrtMemoryInfo("Cuda", OrtAllocatorType.ORT_ARENA_ALLOCATOR, device_id, OrtMemType.DEFAULT)
        # arena_extend_strategy 1: kSameAsRequested
        create_and_register_allocator_v2(
            "CUDAExecutionProvider", mem_info, {"device_id": str(device_id)}, OrtArenaCfg({"arena_extend_strategy": 1})
        )
    except Exception as e:  # pragma: no cover
        logger.info(f"Could not register a shared CUDA memory arena, each session keeps its own: {e}")
        _SHARED_CUDA_ARENAS[device_id] = False
        return False
    _SHARED_CUDA_ARENAS[device_id] = True
    return True


def _cuda_device_id(providers: list[tuple[str, dict[str, Any]]] | list[str]) -> int | None:
    """The CUDA device of the providers, if the CUDA execution provider is requested"""
    for provider in providers:
        if isinstance(provider, (tuple, list)) and provider[0] == "CUDAExecutionProvider":
            return int(provider[1].get("device_id", 0))
        if provider == "CUDAExecutionProvider":
            return 0
    return None


class EngineConfig:
    """Implements a configuration class for the engine of a model

    Args:
        providers: list of providers to use for inference ref.: https://onnxruntime.ai/docs/execution-providers/
        session_options: configuration for the inference session ref.: https://onnxruntime.ai/docs/api/python/api_summary.html#sessionoptions
        run_options_provider: function to customize the run options of each inference call
    """

    def __init__(
        self,
        providers: list[tuple[str, dict[str, Any]]] | list[str] | None = None,
        session_options: SessionOptions | None = None,
        run_options_provider: RunOptionsProvider | None = None,
    ):
        self._providers = providers or self._init_providers()
        self._session_options = session_options or self._init_sess_opts()
        # The default sessions on a GPU share a single CUDA memory arena: user-provided providers (e.g. with a
        # `gpu_mem_limit`) and session options are kept as is
        cuda_device = _cuda_device_id(self._providers)
        if (
            providers is None
            and session_options is None
            and cuda_device is not None
            and _register_shared_cuda_arena(cuda_device)
        ):
            self._session_options.add_session_config_entry("session.use_env_allocators", "1")
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
                        # Grow the memory arena by the requested amount only: rounding each extension up to the next
                        # power of two (kNextPowerOfTwo) roughly doubles the GPU memory footprint of each session,
                        # which runs out of memory once several models (layout, table, formula, ...) are loaded
                        "arena_extend_strategy": "kSameAsRequested",
                        # HEURISTIC picks much faster convolution kernels than DEFAULT (up to ~3x on the
                        # convolutional models) without the benchmarking pass of EXHAUSTIVE on each new input shape
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "do_copy_in_default_stream": True,
                    },
                ),
            )
        elif "OpenVINOExecutionProvider" in available_providers:  # pragma: no cover
            providers.insert(0, ("OpenVINOExecutionProvider", {"device_type": "CPU", "precision": "FP32"}))
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
        # The idle threads of a session otherwise keep spinning after each run, slowing down the other models of the
        # pipeline which run in between (e.g. detection & recognition): up to 2x faster OCR on the CPU
        session_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
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

    # Whether the unused GPU memory of the arena is released after each inference call. Enabled for the models called
    # once in a while with large inputs (classification, layout, table, formula): their peak memory would otherwise
    # stay reserved, on top of the one of the detection & recognition models which are called the most
    _release_gpu_memory_after_run: bool = False

    def __init__(self, url: str, engine_cfg: EngineConfig | None = None, **kwargs: Any) -> None:
        engine_cfg = engine_cfg if isinstance(engine_cfg, EngineConfig) else EngineConfig()
        archive_path = (
            download_from_url(url, cache_subdir="models", **kwargs) if url.startswith(("http://", "https://")) else url
        )
        # NOTE: older onnxruntime versions require a string path for windows
        archive_path = rf"{archive_path}"
        # Store model path for each model
        self.model_path = archive_path
        self.session_options = engine_cfg.session_options
        self.providers = engine_cfg.providers
        self.run_options_provider = engine_cfg.run_options_provider
        self.runtime = self._create_session(archive_path)
        _warn_on_provider_fallback(self.providers, self.runtime.get_providers())
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
        cuda_device = _cuda_device_id(self.providers)
        self._arena_shrinkage: str | None = (
            f"gpu:{cuda_device}"
            if self._release_gpu_memory_after_run
            and cuda_device is not None
            and "CUDAExecutionProvider" in self.runtime.get_providers()
            else None
        )

    def _create_session(self, model_path: str) -> InferenceSession:
        try:
            return InferenceSession(model_path, providers=self.providers, sess_options=self.session_options)
        except Exception as e:
            # Some graphs cannot be compiled by OpenVINO (e.g. unsupported operators): run them on the other providers
            fallback: Any = [p for p in self.providers if _provider_name(p) != "OpenVINOExecutionProvider"]
            if not fallback or len(fallback) == len(self.providers):
                raise
            try:
                session = InferenceSession(model_path, providers=fallback, sess_options=self.session_options)
            except Exception:
                # Not an OpenVINO issue (e.g. a missing or invalid model file): report the original error
                raise e from None
            lines = [line.strip() for line in str(e).splitlines() if line.strip()] or [type(e).__name__]
            reason = next((line for line in lines if line.startswith("While validating ONNX node")), lines[0])[:200]
            logger.warning(
                f"The OpenVINO execution provider cannot run the model {os.path.basename(model_path)}, falling back "
                f"to {[_provider_name(p) for p in fallback]}: {reason}"
            )
            self.providers = fallback
            return session

    def _run_options(self) -> RunOptions:
        run_options = RunOptions()
        if self._arena_shrinkage is not None:
            run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", self._arena_shrinkage)
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
        run_options = self._run_options()
        if self.tf_exported:
            inputs = shape_translate(inputs, format="BHWC")  # sanity check
        else:
            inputs = shape_translate(inputs, format="BCHW")
        if isinstance(self.fixed_batch_size, int) and self.fixed_batch_size > 0:  # dynamic batch size is a string
            # The graph only accepts batches of exactly `fixed_batch_size`: feed it chunk by chunk, padding the
            # last chunk with zeros, and drop the outputs of the padding
            fbs, num_samples = self.fixed_batch_size, inputs.shape[0]
            outputs = []
            for start in range(0, num_samples, fbs):
                chunk = inputs[start : start + fbs]
                if chunk.shape[0] < fbs:
                    pad = np.zeros((fbs - chunk.shape[0], *chunk.shape[1:]), dtype=chunk.dtype)
                    chunk = np.concatenate([chunk, pad], axis=0)
                out = self.runtime.run(self.output_name, {self.runtime_inputs.name: chunk}, run_options=run_options)[0]
                outputs.append(out[: min(fbs, num_samples - start)])
            logits = np.concatenate(outputs, axis=0)
        else:
            logits = self.runtime.run(self.output_name, {self.runtime_inputs.name: inputs}, run_options=run_options)[0]
        return shape_translate(logits, format="BHWC")
