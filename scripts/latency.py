# Copyright (C) 2021-2026, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import argparse
import time
from typing import Any

import numpy as np

from onnxtr.models import classification, detection, layout, recognition, table_structure
from onnxtr.models.classification.zoo import ORIENTATION_ARCHS
from onnxtr.models.detection.zoo import ARCHS as DETECTION_ARCHS
from onnxtr.models.layout.zoo import ARCHS as LAYOUT_ARCHS
from onnxtr.models.recognition.zoo import ARCHS as RECOGNITION_ARCHS
from onnxtr.models.table_structure.zoo import ARCHS as TABLE_ARCHS

ALL_ARCHS = DETECTION_ARCHS + RECOGNITION_ARCHS + ORIENTATION_ARCHS + LAYOUT_ARCHS + TABLE_ARCHS


def _model_args(model: Any, img_tensor: np.ndarray) -> tuple[np.ndarray, ...]:
    """Build the extra positional arguments the model expects beyond the image

    Args:
        model: the loaded OnnxTR model
        img_tensor: the image tensor of shape (N, C, H, W)

    Returns:
        the extra positional arguments, empty for single-input models
    """
    if len(model.runtime_input_metas) < 2:
        return ()
    # a full-True mask means "no padding": every pixel is valid image content
    return (np.ones((img_tensor.shape[0], *img_tensor.shape[-2:]), dtype=bool),)


def main(args):
    if args.arch in DETECTION_ARCHS:
        model = detection.__dict__[args.arch](load_in_8_bit=args.load8bit)
    elif args.arch in RECOGNITION_ARCHS:
        model = recognition.__dict__[args.arch](load_in_8_bit=args.load8bit)
    elif args.arch in ORIENTATION_ARCHS:
        model = classification.__dict__[args.arch](load_in_8_bit=args.load8bit)
    elif args.arch in LAYOUT_ARCHS:
        model = layout.__dict__[args.arch](load_in_8_bit=args.load8bit)
    elif args.arch in TABLE_ARCHS:
        model = table_structure.__dict__[args.arch](load_in_8_bit=args.load8bit)
    else:
        raise ValueError(f"Unknown architecture {args.arch}")

    size = (1, *model.cfg["input_shape"])
    img_tensor = np.random.rand(*size).astype(np.float32)
    extra_args = _model_args(model, img_tensor)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OnnxTR latency benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "arch",
        type=str,
        choices=ALL_ARCHS,
        help="Architecture to benchmark",
    )
    parser.add_argument("--load8bit", action="store_true", help="Load the 8-bit quantized model")
    parser.add_argument("--it", type=int, default=1000, help="Number of iterations to run")
    args = parser.parse_args()

    main(args)
