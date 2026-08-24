import argparse
import os
import time
from dataclasses import dataclass, field

import numpy as np
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic, quantize_static

from onnxtr.io.image import read_img_as_numpy
from onnxtr.models.preprocessor import PreProcessor
from onnxtr.utils.geometry import shape_translate


@dataclass
class TaskConfig:
    """Preprocessing configuration of a task, mirroring what the matching predictor does at inference time."""

    shape: tuple[int, int]
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    preserve_aspect_ratio: bool = False
    symmetric_pad: bool = False


TASKS: dict[str, TaskConfig] = {
    "crop_orientation": TaskConfig(
        shape=(256, 256),
        mean=(0.798, 0.785, 0.772),
        std=(0.264, 0.2749, 0.287),
        preserve_aspect_ratio=True,
        symmetric_pad=True,
    ),
    "page_orientation": TaskConfig(
        shape=(512, 512),
        mean=(0.798, 0.785, 0.772),
        std=(0.264, 0.2749, 0.287),
        preserve_aspect_ratio=True,
        symmetric_pad=True,
    ),
    "detection": TaskConfig(
        shape=(1024, 1024),
        mean=(0.798, 0.785, 0.772),
        std=(0.264, 0.2749, 0.287),
        preserve_aspect_ratio=True,
        symmetric_pad=True,
    ),
    "recognition": TaskConfig(
        shape=(32, 128),
        mean=(0.694, 0.695, 0.693),
        std=(0.299, 0.296, 0.301),
        preserve_aspect_ratio=True,
    ),
    "layout": TaskConfig(
        shape=(1024, 1024),
        mean=(0.798, 0.785, 0.772),
        std=(0.264, 0.2749, 0.287),
        preserve_aspect_ratio=True,
        symmetric_pad=True,
    ),
    "table_structure": TaskConfig(
        shape=(1024, 1024),
        mean=(0.798, 0.785, 0.772),
        std=(0.264, 0.2749, 0.287),
        preserve_aspect_ratio=True,
        symmetric_pad=True,
    ),
}


@dataclass
class ModelIO:
    """Input/output names of the graph being quantized"""

    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)

    @property
    def needs_padding_mask(self) -> bool:
        return len(self.input_names) > 1


def _model_io(model_path: str) -> ModelIO:
    session = onnxruntime.InferenceSession(model_path, None)
    return ModelIO(
        input_names=[inp.name for inp in session.get_inputs()],
        output_names=[out.name for out in session.get_outputs()],
    )


class CalibrationDataLoader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str, task_config: TaskConfig):
        self.enum_data = None
        self.io = _model_io(model_path)

        self.preprocessor = PreProcessor(
            output_size=task_config.shape,
            batch_size=1,
            mean=task_config.mean,
            std=task_config.std,
            preserve_aspect_ratio=task_config.preserve_aspect_ratio,
            symmetric_pad=task_config.symmetric_pad,
        )
        # Multi-input models additionally need the padding mask produced by the resize
        self.preprocessor.resize.return_padding_mask = self.io.needs_padding_mask

        self.dataset: list[dict[str, np.ndarray]] = []
        for img_file in sorted(os.listdir(calibration_image_folder))[:500]:  # limit to 500 images
            img = read_img_as_numpy(os.path.join(calibration_image_folder, img_file))
            batch = self.preprocessor([img])[0]
            if self.io.needs_padding_mask:
                images, masks = batch
                self.dataset.append({
                    self.io.input_names[0]: shape_translate(images, format="BCHW"),
                    self.io.input_names[1]: masks,
                })
            else:
                self.dataset.append({self.io.input_names[0]: shape_translate(batch, format="BCHW")})

        self.datasize = len(self.dataset)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.dataset)
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


def benchmark(calibration_image_folder: str, model_path: str, task_config: TaskConfig):
    session = onnxruntime.InferenceSession(model_path)
    output_names = [output.name for output in session.get_outputs()]
    dataset = CalibrationDataLoader(calibration_image_folder, model_path, task_config)
    sample = dataset.dataset[0]  # take 1 sample for benchmarking

    total = 0.0
    runs = 10
    # Warming up
    _ = session.run(output_names, sample)
    for _ in range(runs):
        start = time.perf_counter()
        _ = session.run(output_names, sample)
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")


def benchmark_mean_diff(
    calibration_image_folder: str, model_path: str, quantized_model_path: str, task_config: TaskConfig
):
    """Check the mean difference between the original and quantized model"""
    session = onnxruntime.InferenceSession(model_path)
    quantized_session = onnxruntime.InferenceSession(quantized_model_path)
    output_names = [output.name for output in session.get_outputs()]
    quantized_output_names = [output.name for output in quantized_session.get_outputs()]
    dataset = CalibrationDataLoader(calibration_image_folder, model_path, task_config)
    sample = dataset.dataset[0]  # take 1 sample for benchmarking

    outputs = session.run(output_names, sample)
    quantized_outputs = quantized_session.run(quantized_output_names, sample)

    worst = 0.0
    for name, output, quantized_output in zip(output_names, outputs, quantized_outputs):
        mean_diff = float(np.mean(np.abs(output - quantized_output)))
        worst = max(worst, mean_diff)
        if len(output_names) > 1:
            print(f"  {name}: mean difference {mean_diff:.4f}")
    print(f"Mean difference between original and quantized model: {worst:.2f}")


def main(args):
    input_model_path = args.input_model
    calibration_dataset_path = args.calibrate_dataset
    task_config = TASKS[args.task]
    print(f"Task: {args.task} | Task shape: {task_config.shape}")

    io = _model_io(input_model_path)
    if io.needs_padding_mask:
        print(f"Model expects {len(io.input_names)} inputs {io.input_names}: a padding mask will be calibrated too")

    dr = CalibrationDataLoader(calibration_dataset_path, input_model_path, task_config)
    base_model_name = input_model_path.split("/")[-1].split("-")[0]
    static_out_name = base_model_name + "_static_8_bit.onnx"
    dynamic_out_name = base_model_name + "_dynamic_8_bit.onnx"

    print("benchmarking fp32 model...")
    benchmark(calibration_dataset_path, input_model_path, task_config)

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
            # the reader was consumed by the failed attempt
            dr.rewind()
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
        benchmark(calibration_dataset_path, static_out_name, task_config)

        print("benchmarking mean difference between fp32 and static int8 model...")
        benchmark_mean_diff(calibration_dataset_path, input_model_path, static_out_name, task_config)

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
        benchmark(calibration_dataset_path, dynamic_out_name, task_config)

        print("benchmarking mean difference between fp32 and dynamic int8 model...")
        benchmark_mean_diff(calibration_dataset_path, input_model_path, dynamic_out_name, task_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OnnxTR script to quantize models and benchmark the quantized models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_model", required=True, help="input model")
    parser.add_argument(
        "--task",
        required=True,
        type=str,
        choices=list(TASKS),
        help="task shape",
    )
    parser.add_argument(
        "--calibrate_dataset",
        type=str,
        required=True,
        help="calibration data set (word crop images for recognition, crop_orientation else page images for detection, page_orientation, layout, table_structure)",  # noqa
    )
    parser.add_argument(
        "--quant_format",
        default=QuantFormat.QDQ,
        type=QuantFormat.from_string,
        choices=list(QuantFormat),
    )
    args = parser.parse_args()

    main(args)
