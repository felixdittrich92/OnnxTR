import argparse
import os
import time
from enum import Enum
from typing import Tuple

import numpy as np
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic, quantize_static

from onnxtr.io.image import read_img_as_numpy
from onnxtr.models.preprocessor import PreProcessor
from onnxtr.utils.geometry import shape_translate


class TaskShapes(Enum):
    """Enum class to define the shapes of the input tensors for different tasks"""

    crop_orientation = (256, 256)
    page_orientation = (512, 512)
    detection = (1024, 1024)
    recognition = (32, 128)


class CalibrationDataLoader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str, task_shape: Tuple[int]):
        self.enum_data = None
        self.preprocessor = PreProcessor(output_size=task_shape, batch_size=1)
        self.dataset = [
            self.preprocessor(
                np.expand_dims(read_img_as_numpy(os.path.join(calibration_image_folder, img_file)), axis=0)
            )
            for img_file in os.listdir(calibration_image_folder)[:500]  # limit to 500 images
        ]

        session = onnxruntime.InferenceSession(model_path, None)
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.dataset)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter([
                {self.input_name: shape_translate(input_data[0], format="BCHW")} for input_data in self.dataset
            ])
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


def benchmark(calibration_image_folder: str, model_path: str, task_shape: Tuple[int]):
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = [output.name for output in session.get_outputs()]
    dataset = CalibrationDataLoader(calibration_image_folder, model_path, task_shape)
    sample = shape_translate(dataset.dataset[0][0], format="BCHW")  # take 1 sample for benchmarking

    total = 0.0
    runs = 10
    # Warming up
    _ = session.run(output_name, {input_name: sample})
    for _ in range(runs):
        start = time.perf_counter()
        _ = session.run(output_name, {input_name: sample})
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")


def benchmark_mean_diff(
    calibration_image_folder: str, model_path: str, quantized_model_path: str, task_shape: Tuple[int]
):
    """Check the mean difference between the original and quantized model"""
    session = onnxruntime.InferenceSession(model_path)
    quantized_session = onnxruntime.InferenceSession(quantized_model_path)
    input_name = session.get_inputs()[0].name
    output_name = [output.name for output in session.get_outputs()]
    quantized_output_name = [output.name for output in quantized_session.get_outputs()]
    dataset = CalibrationDataLoader(calibration_image_folder, model_path, task_shape)
    sample = shape_translate(dataset.dataset[0][0], format="BCHW")  # take 1 sample for benchmarking

    output = session.run(output_name, {input_name: sample})[0]
    quantized_output = quantized_session.run(quantized_output_name, {input_name: sample})[0]

    mean_diff = np.mean(np.abs(output - quantized_output))
    print(f"Mean difference between original and quantized model: {mean_diff:.2f}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", required=True, help="input model")
    parser.add_argument(
        "--task",
        required=True,
        type=str,
        choices=["crop_orientation", "page_orientation", "detection", "recognition"],
        help="task shape",
    )
    parser.add_argument(
        "--calibrate_dataset",
        type=str,
        required=True,
        help="calibration data set (word crop images for recognition, crop_orientation else page images for detection, page_orientation)",  # noqa
    )
    parser.add_argument(
        "--quant_format",
        default=QuantFormat.QDQ,
        type=QuantFormat.from_string,
        choices=list(QuantFormat),
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_model_path = args.input_model
    calibration_dataset_path = args.calibrate_dataset
    if args.task == "crop_orientation":
        task_shape = TaskShapes.crop_orientation.value
    elif args.task == "page_orientation":
        task_shape = TaskShapes.page_orientation.value
    elif args.task == "detection":
        task_shape = TaskShapes.detection.value
    else:
        task_shape = TaskShapes.recognition.value
    print(f"Task: {args.task} | Task shape: {task_shape}")

    dr = CalibrationDataLoader(calibration_dataset_path, input_model_path, task_shape)
    base_model_name = input_model_path.split("/")[-1].split("-")[0]
    static_out_name = base_model_name + "_static_8_bit.onnx"
    dynamic_out_name = base_model_name + "_dynamic_8_bit.onnx"

    print("benchmarking fp32 model...")
    benchmark(calibration_dataset_path, input_model_path, task_shape)

    # Calibrate and quantize model
    # Turn off model optimization during quantization
    if "parseq" not in input_model_path:  # Skip static quantization for Parseq
        print("Calibrating and quantizing model static...")
        try:
            quantize_static(
                input_model_path,
                static_out_name,
                dr,
                quant_format=args.quant_format,
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QUInt8,
                reduce_range=True,
            )
        except Exception:
            print("Error during static quantization --> Change weight_type also to QUInt8")
            quantize_static(
                input_model_path,
                static_out_name,
                dr,
                quant_format=args.quant_format,
                weight_type=QuantType.QUInt8,
                activation_type=QuantType.QUInt8,
                reduce_range=True,
            )

        print("benchmarking static int8 model...")
        benchmark(calibration_dataset_path, static_out_name, task_shape)

        print("benchmarking mean difference between fp32 and static int8 model...")
        benchmark_mean_diff(calibration_dataset_path, input_model_path, static_out_name, task_shape)

        print("Calibrated and quantized static model saved.")

    if "sar" not in input_model_path:  # Skip dynamic quantization for SAR_ResNet31
        print("Dynamic int 8 quantization...")
        quantize_dynamic(
            input_model_path,
            dynamic_out_name,
            weight_type=QuantType.QUInt8,
        )
        print("Dynamic model saved.")

        print("benchmarking dynamic int8 model...")
        benchmark(calibration_dataset_path, dynamic_out_name, task_shape)

        print("benchmarking mean difference between fp32 and dynamic int8 model...")
        benchmark_mean_diff(calibration_dataset_path, input_model_path, dynamic_out_name, task_shape)


if __name__ == "__main__":
    main()
