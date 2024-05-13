import cv2
import numpy as np
import pytest

from onnxtr.models import classification
from onnxtr.models.classification.predictor import OrientationPredictor
from onnxtr.models.engine import Engine


@pytest.mark.parametrize(
    "arch_name, input_shape",
    [
        ["mobilenet_v3_small_crop_orientation", (256, 256, 3)],
        ["mobilenet_v3_small_page_orientation", (512, 512, 3)],
    ],
)
def test_classification_models(arch_name, input_shape):
    batch_size = 8
    model = classification.__dict__[arch_name]()
    assert isinstance(model, Engine)
    input_tensor = np.random.rand(batch_size, *input_shape).astype(np.float32)
    out = model(input_tensor)
    assert isinstance(out, np.ndarray)
    assert out.shape == (8, 4)


@pytest.mark.parametrize(
    "arch_name",
    [
        "mobilenet_v3_small_crop_orientation",
        "mobilenet_v3_small_page_orientation",
    ],
)
def test_classification_zoo(arch_name):
    if "crop" in arch_name:
        batch_size = 16
        input_array = np.random.rand(batch_size, 3, 256, 256).astype(np.float32)
        # Model
        predictor = classification.zoo.crop_orientation_predictor(arch_name)

        with pytest.raises(ValueError):
            predictor = classification.zoo.crop_orientation_predictor(arch="wrong_model")
    else:
        batch_size = 2
        input_array = np.random.rand(batch_size, 3, 512, 512).astype(np.float32)
        # Model
        predictor = classification.zoo.page_orientation_predictor(arch_name)

        with pytest.raises(ValueError):
            predictor = classification.zoo.page_orientation_predictor(arch="wrong_model")
    # object check
    assert isinstance(predictor, OrientationPredictor)

    out = predictor(input_array)
    class_idxs, classes, confs = out[0], out[1], out[2]
    assert isinstance(class_idxs, list) and len(class_idxs) == batch_size
    assert isinstance(classes, list) and len(classes) == batch_size
    assert isinstance(confs, list) and len(confs) == batch_size
    assert all(isinstance(pred, int) for pred in class_idxs)
    assert all(isinstance(pred, int) for pred in classes) and all(pred in [0, 90, 180, -90] for pred in classes)
    assert all(isinstance(pred, float) for pred in confs)


@pytest.mark.parametrize("quantized", [False, True])
def test_crop_orientation_model(mock_text_box, quantized):
    text_box_0 = cv2.imread(mock_text_box)
    # rotates counter-clockwise
    text_box_270 = np.rot90(text_box_0, 1)
    text_box_180 = np.rot90(text_box_0, 2)
    text_box_90 = np.rot90(text_box_0, 3)
    classifier = classification.crop_orientation_predictor(
        "mobilenet_v3_small_crop_orientation", load_in_8_bit=quantized
    )
    assert classifier([text_box_0, text_box_270, text_box_180, text_box_90])[0] == [0, 1, 2, 3]
    # 270 degrees is equivalent to -90 degrees
    assert classifier([text_box_0, text_box_270, text_box_180, text_box_90])[1] == [0, -90, 180, 90]
    assert all(isinstance(pred, float) for pred in classifier([text_box_0, text_box_270, text_box_180, text_box_90])[2])


@pytest.mark.parametrize("quantized", [False, True])
def test_page_orientation_model(mock_payslip, quantized):
    text_box_0 = cv2.imread(mock_payslip)
    # rotates counter-clockwise
    text_box_270 = np.rot90(text_box_0, 1)
    text_box_180 = np.rot90(text_box_0, 2)
    text_box_90 = np.rot90(text_box_0, 3)
    classifier = classification.crop_orientation_predictor(
        "mobilenet_v3_small_page_orientation", load_in_8_bit=quantized
    )
    assert classifier([text_box_0, text_box_270, text_box_180, text_box_90])[0] == [0, 1, 2, 3]
    # 270 degrees is equivalent to -90 degrees
    assert classifier([text_box_0, text_box_270, text_box_180, text_box_90])[1] == [0, -90, 180, 90]
    assert all(isinstance(pred, float) for pred in classifier([text_box_0, text_box_270, text_box_180, text_box_90])[2])
