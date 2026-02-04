# Copyright (C) 2021-2026, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

try:
    from onnxconverter_common import auto_convert_mixed_precision
except ImportError:
    raise ImportError("Failed to import onnxconverter_common. Please install `pip install onnxconverter-common`.")

# Check GPU availability
import onnxruntime

if onnxruntime.get_device() != "GPU":
    raise RuntimeError(
        "Please install OnnxTR with GPU support to run this script. "
        + "`pip install onnxtr[gpu]` or `pip install -e .[gpu]`"
    )

import argparse
import time
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import onnx

from onnxtr.models import classification, detection, recognition
from onnxtr.models.classification.zoo import ORIENTATION_ARCHS
from onnxtr.models.detection.zoo import ARCHS as DETECTION_ARCHS
from onnxtr.models.recognition.zoo import ARCHS as RECOGNITION_ARCHS


def _load_model(arch: str, model_path: str | None = None) -> Any:
    if arch in DETECTION_ARCHS:
        model = detection.__dict__[arch]() if model_path is None else detection.__dict__[arch](model_path)
    elif args.arch in RECOGNITION_ARCHS:
        model = recognition.__dict__[arch]() if model_path is None else recognition.__dict__[arch](model_path)
    elif args.arch in ORIENTATION_ARCHS:
        model = classification.__dict__[arch]() if model_path is None else classification.__dict__[arch](model_path)
    else:
        raise ValueError(f"Unknown architecture {arch}")
    return model


def _latency_check(args: Any, size: tuple[int], model: Any, img_tensor: np.ndarray) -> None:
    # Warmup
    for _ in range(10):
        _ = model(img_tensor)

    timings = []

    # Evaluation runs
    for _ in range(args.it):
        start_ts = time.perf_counter()
        _ = model(img_tensor)
        timings.append(time.perf_counter() - start_ts)

    _timings = np.array(timings)
    print(f"{args.arch} ({args.it} runs on ({size}) inputs)")
    print(f"mean {1000 * _timings.mean():.2f}ms, std {1000 * _timings.std():.2f}ms")


def _validate(fp32_in: list[np.ndarray], fp16_in: list[np.ndarray]) -> bool:
    assert fp32_in[0].shape == fp16_in[0].shape, "Input shapes are not the same"
    # print mean difference between fp32 and fp16 inputs
    if np.abs(fp32_in[0] - fp16_in[0]).mean() > 1e-3:
        print(
            f"Mean difference between fp32 and fp16 inputs: {np.abs(fp32_in[0] - fp16_in[0]).mean()} "
            + "-> YOU MAY EXPECT DIFFERING RESULTS"
        )
    return True  # NOTE: Only warning, not error


def main(args):
    model_float32 = _load_model(args.arch, model_path=args.input_model if args.input_model else None)
    size = (1, *model_float32.cfg["input_shape"])

    img_tensor = np.random.rand(*size).astype(np.float32)

    with TemporaryDirectory() as temp_dir:
        model_fp16_path = f"{temp_dir}/model_fp16.onnx"
        input_feed = {model_float32.runtime_inputs.name: img_tensor}
        model_float16 = auto_convert_mixed_precision(
            # NOTE: keep_io_types=True is required to keep the input/output type as float32
            onnx.load(str(model_float32.model_path)),
            input_feed,
            validate_fn=_validate,
            keep_io_types=True,
        )
        onnx.save(model_float16, model_fp16_path)
        model_fp16 = _load_model(args.arch, model_fp16_path)

    # Latency check
    _latency_check(args, size, model_float32, img_tensor)
    _latency_check(args, size, model_fp16, img_tensor)

    onnx.save(model_float16, args.arch + "_fp16.onnx")
    print(f"FP16 model saved at {args.arch}_fp16.onnx")
    print("Attention: FP16 converted models can only run on GPU devices.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OnnxTR FP32 to FP16 conversion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "arch",
        type=str,
        choices=DETECTION_ARCHS + RECOGNITION_ARCHS + ORIENTATION_ARCHS,
        help="Architecture to convert",
    )
    parser.add_argument("--input_model", type=str, help="Path to the input model", required=False)
    parser.add_argument("--it", type=int, default=1000, help="Number of iterations to run")
    args = parser.parse_args()

    main(args)
