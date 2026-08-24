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

from onnxtr.models import classification, detection, layout, recognition, table_structure
from onnxtr.models.classification.zoo import ORIENTATION_ARCHS
from onnxtr.models.detection.zoo import ARCHS as DETECTION_ARCHS
from onnxtr.models.layout.zoo import ARCHS as LAYOUT_ARCHS
from onnxtr.models.recognition.zoo import ARCHS as RECOGNITION_ARCHS
from onnxtr.models.table_structure.zoo import ARCHS as TABLE_ARCHS

ALL_ARCHS = DETECTION_ARCHS + RECOGNITION_ARCHS + ORIENTATION_ARCHS + LAYOUT_ARCHS + TABLE_ARCHS


def _load_model(arch: str, model_path: str | None = None) -> Any:
    if arch in DETECTION_ARCHS:
        model = detection.__dict__[arch]() if model_path is None else detection.__dict__[arch](model_path)
    elif arch in RECOGNITION_ARCHS:
        model = recognition.__dict__[arch]() if model_path is None else recognition.__dict__[arch](model_path)
    elif arch in ORIENTATION_ARCHS:
        model = classification.__dict__[arch]() if model_path is None else classification.__dict__[arch](model_path)
    elif arch in LAYOUT_ARCHS:
        model = layout.__dict__[arch]() if model_path is None else layout.__dict__[arch](model_path)
    elif arch in TABLE_ARCHS:
        model = table_structure.__dict__[arch]() if model_path is None else table_structure.__dict__[arch](model_path)
    else:
        raise ValueError(f"Unknown architecture {arch}")
    return model


def _build_input_feed(model: Any, img_tensor: np.ndarray) -> dict[str, np.ndarray]:
    """Build the feed for every graph input

    Args:
        model: the loaded OnnxTR model
        img_tensor: the image tensor of shape (N, C, H, W)

    Returns:
        the mapping from graph input name to array
    """
    feed: dict[str, np.ndarray] = {model.runtime_inputs.name: img_tensor}
    for meta in model.runtime_input_metas[1:]:
        # a full-True mask means "no padding": every pixel is valid image content
        mask = np.ones((img_tensor.shape[0], *img_tensor.shape[-2:]), dtype=bool)
        feed[meta.name] = mask if "bool" in meta.type else mask.astype(np.float32)
    return feed


def _latency_check(args: Any, size: tuple[int], model: Any, img_tensor: np.ndarray) -> None:
    # layout models takes the padding mask as a second positional argument, the others do not
    extra_args = (
        (np.ones((img_tensor.shape[0], *img_tensor.shape[-2:]), dtype=bool),)
        if len(model.runtime_input_metas) > 1
        else ()
    )

    # Warmup
    for _ in range(10):
        _ = model(img_tensor, *extra_args)

    timings = []

    # Evaluation runs
    for _ in range(args.it):
        start_ts = time.perf_counter()
        _ = model(img_tensor, *extra_args)
        timings.append(time.perf_counter() - start_ts)

    _timings = np.array(timings)
    print(f"{args.arch} ({args.it} runs on ({size}) inputs)")
    print(f"mean {1000 * _timings.mean():.2f}ms, std {1000 * _timings.std():.2f}ms")


def _validate(fp32_in: list[np.ndarray], fp16_in: list[np.ndarray]) -> bool:
    assert len(fp32_in) == len(fp16_in), "Number of outputs is not the same"
    # print mean difference between fp32 and fp16 outputs
    for idx, (fp32_out, fp16_out) in enumerate(zip(fp32_in, fp16_in)):
        assert fp32_out.shape == fp16_out.shape, f"Output {idx} shapes are not the same"
        mean_diff = np.abs(fp32_out - fp16_out).mean()
        if mean_diff > 1e-3:
            print(
                f"Mean difference between fp32 and fp16 outputs (output {idx}): {mean_diff} "
                + "-> YOU MAY EXPECT DIFFERING RESULTS"
            )
    return True  # NOTE: Only warning, not error


def main(args):
    model_float32 = _load_model(args.arch, model_path=args.input_model if args.input_model else None)
    size = (1, *model_float32.cfg["input_shape"])

    img_tensor = np.random.rand(*size).astype(np.float32)

    with TemporaryDirectory() as temp_dir:
        model_fp16_path = f"{temp_dir}/model_fp16.onnx"
        input_feed = _build_input_feed(model_float32, img_tensor)
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
        choices=ALL_ARCHS,
        help="Architecture to convert",
    )
    parser.add_argument("--input_model", type=str, help="Path to the input model", required=False)
    parser.add_argument("--it", type=int, default=1000, help="Number of iterations to run")
    args = parser.parse_args()

    main(args)
