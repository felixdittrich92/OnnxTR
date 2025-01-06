# Copyright (C) 2021-2025, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import argparse
import time

import numpy as np

from onnxtr.models import classification, detection, recognition
from onnxtr.models.classification.zoo import ORIENTATION_ARCHS
from onnxtr.models.detection.zoo import ARCHS as DETECTION_ARCHS
from onnxtr.models.recognition.zoo import ARCHS as RECOGNITION_ARCHS


def main(args):
    if args.arch in DETECTION_ARCHS:
        model = detection.__dict__[args.arch](load_in_8_bit=args.load8bit)
    elif args.arch in RECOGNITION_ARCHS:
        model = recognition.__dict__[args.arch](load_in_8_bit=args.load8bit)
    elif args.arch in ORIENTATION_ARCHS:
        model = classification.__dict__[args.arch](load_in_8_bit=args.load8bit)
    else:
        raise ValueError(f"Unknown architecture {args.arch}")

    size = (1, *model.cfg["input_shape"])
    img_tensor = np.random.rand(*size).astype(np.float32)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OnnxTR latency benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "arch",
        type=str,
        choices=DETECTION_ARCHS + RECOGNITION_ARCHS + ORIENTATION_ARCHS,
        help="Architecture to benchmark",
    )
    parser.add_argument("--load8bit", action="store_true", help="Load the 8-bit quantized model")
    parser.add_argument("--it", type=int, default=1000, help="Number of iterations to run")
    args = parser.parse_args()

    main(args)
