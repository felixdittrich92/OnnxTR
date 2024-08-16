import json
import os
import tempfile

import pytest

from onnxtr import models
from onnxtr.models.factory import _save_model_and_config_for_hf_hub, from_hub, push_to_hf_hub

AVAILABLE_ARCHS = {
    "classification": models.classification.zoo.ORIENTATION_ARCHS,
    "detection": models.detection.zoo.ARCHS,
    "recognition": models.recognition.zoo.ARCHS,
}


def test_push_to_hf_hub():
    model = models.classification.mobilenet_v3_small_crop_orientation()
    with pytest.raises(ValueError):
        # run_config and/or arch must be specified
        push_to_hf_hub(model, model_name="test", task="classification")
    with pytest.raises(ValueError):
        # task must be one of classification, detection, recognition, obj_detection
        push_to_hf_hub(model, model_name="test", task="invalid_task", arch="mobilenet_v3_small")
    with pytest.raises(ValueError):
        # arch not in available architectures for task
        push_to_hf_hub(model, model_name="test", task="detection", arch="crnn_mobilenet_v3_large")


def test_models_huggingface_hub(tmpdir):
    with tempfile.TemporaryDirectory() as tmp_dir:
        for task_name, archs in AVAILABLE_ARCHS.items():
            for arch_name in archs:
                model = models.__dict__[task_name].__dict__[arch_name]()

                _save_model_and_config_for_hf_hub(model, arch=arch_name, task=task_name, save_dir=tmp_dir)

                assert hasattr(model, "cfg")
                assert len(os.listdir(tmp_dir)) == 2
                assert os.path.exists(tmp_dir + "/model.onnx")
                assert os.path.exists(tmp_dir + "/config.json")
                tmp_config = json.load(open(tmp_dir + "/config.json"))
                assert arch_name == tmp_config["arch"]
                assert task_name == tmp_config["task"]
                assert all(key in model.cfg.keys() for key in tmp_config.keys())

                # test from hub
                hub_model = from_hub(repo_id="Felix92/onnxtr-{}".format(arch_name).replace("_", "-"))
                assert isinstance(hub_model, type(model))
