# Copyright (C) 2021-2025, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Inspired by: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/hub.py

import json
import logging
import os
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any

from huggingface_hub import (
    HfApi,
    Repository,
    get_token,
    get_token_permission,
    hf_hub_download,
    login,
)

from onnxtr import models
from onnxtr.models.engine import EngineConfig

__all__ = ["login_to_hub", "push_to_hf_hub", "from_hub", "_save_model_and_config_for_hf_hub"]


AVAILABLE_ARCHS = {
    "classification": models.classification.zoo.ORIENTATION_ARCHS,
    "detection": models.detection.zoo.ARCHS,
    "recognition": models.recognition.zoo.ARCHS,
}


def login_to_hub() -> None:  # pragma: no cover
    """Login to huggingface hub"""
    access_token = get_token()
    if access_token is not None and get_token_permission(access_token):
        logging.info("Huggingface Hub token found and valid")
        login(token=access_token, write_permission=True)
    else:
        login()
    # check if git lfs is installed
    try:
        subprocess.call(["git", "lfs", "version"])
    except FileNotFoundError:
        raise OSError(
            "Looks like you do not have git-lfs installed, please install. \
                      You can install from https://git-lfs.github.com/. \
                      Then run `git lfs install` (you only have to do this once)."
        )


def _save_model_and_config_for_hf_hub(model: Any, save_dir: str, arch: str, task: str) -> None:
    """Save model and config to disk for pushing to huggingface hub

    Args:
        model: Onnx model to be saved
        save_dir: directory to save model and config
        arch: architecture name
        task: task name
    """
    save_directory = Path(save_dir)
    shutil.copy2(model.model_path, save_directory / "model.onnx")

    config_path = save_directory / "config.json"

    # add model configuration
    model_config = model.cfg
    model_config["arch"] = arch
    model_config["task"] = task

    with config_path.open("w") as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)


def push_to_hf_hub(
    model: Any, model_name: str, task: str, override: bool = False, **kwargs
) -> None:  # pragma: no cover
    """Save model and its configuration on HF hub

    >>> from onnxtr.models import login_to_hub, push_to_hf_hub
    >>> from onnxtr.models.recognition import crnn_mobilenet_v3_small
    >>> login_to_hub()
    >>> model = crnn_mobilenet_v3_small()
    >>> push_to_hf_hub(model, 'my-model', 'recognition', arch='crnn_mobilenet_v3_small')

    Args:
        model: Onnx model to be saved
        model_name: name of the model which is also the repository name
        task: task name
        override: whether to override the existing model / repo on HF hub
        **kwargs: keyword arguments for push_to_hf_hub
    """
    run_config = kwargs.get("run_config", None)
    arch = kwargs.get("arch", None)

    if run_config is None and arch is None:
        raise ValueError("run_config or arch must be specified")
    if task not in ["classification", "detection", "recognition"]:
        raise ValueError("task must be one of classification, detection, recognition")

    # default readme
    readme = textwrap.dedent(
        f"""
    ---
    language:
    - en
    - fr
    license: apache-2.0
    ---

    <p align="center">
    <img src="https://github.com/felixdittrich92/OnnxTR/raw/main/docs/images/logo.jpg" width="40%">
    </p>

    **Optical Character Recognition made seamless & accessible to anyone, powered by Onnxruntime**

    ## Task: {task}

    https://github.com/felixdittrich92/OnnxTR

    ### Example usage:

    ```python
    >>> from onnxtr.io import DocumentFile
    >>> from onnxtr.models import ocr_predictor, from_hub

    >>> img = DocumentFile.from_images(['<image_path>'])
    >>> # Load your model from the hub
    >>> model = from_hub('onnxtr/my-model')

    >>> # Pass it to the predictor
    >>> # If your model is a recognition model:
    >>> predictor = ocr_predictor(det_arch='db_mobilenet_v3_large',
    >>>                           reco_arch=model)

    >>> # If your model is a detection model:
    >>> predictor = ocr_predictor(det_arch=model,
    >>>                           reco_arch='crnn_mobilenet_v3_small')

    >>> # Get your predictions
    >>> res = predictor(img)
    ```
    """
    )

    # add run configuration to readme if available
    if run_config is not None:
        arch = run_config.arch
        readme += textwrap.dedent(
            f"""### Run Configuration
                                  \n{json.dumps(vars(run_config), indent=2, ensure_ascii=False)}"""
        )

    if arch not in AVAILABLE_ARCHS[task]:
        raise ValueError(
            f"Architecture: {arch} for task: {task} not found.\
                         \nAvailable architectures: {AVAILABLE_ARCHS}"
        )

    commit_message = f"Add {model_name} model"

    local_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", model_name)
    repo_url = HfApi().create_repo(model_name, token=get_token(), exist_ok=override)
    repo = Repository(local_dir=local_cache_dir, clone_from=repo_url)

    with repo.commit(commit_message):
        _save_model_and_config_for_hf_hub(model, repo.local_dir, arch=arch, task=task)
        readme_path = Path(repo.local_dir) / "README.md"
        readme_path.write_text(readme)

    repo.git_push()


def from_hub(repo_id: str, engine_cfg: EngineConfig | None = None, **kwargs: Any):
    """Instantiate & load a pretrained model from HF hub.

    >>> from onnxtr.models import from_hub
    >>> model = from_hub("onnxtr/my-model")

    Args:
        repo_id: HuggingFace model hub repo
        engine_cfg: configuration for the inference engine (optional)
        **kwargs: kwargs of `hf_hub_download`

    Returns:
        Model loaded with the checkpoint
    """
    # Get the config
    with open(hf_hub_download(repo_id, filename="config.json", **kwargs), "rb") as f:
        cfg = json.load(f)
        model_path = hf_hub_download(repo_id, filename="model.onnx", **kwargs)

    arch = cfg["arch"]
    task = cfg["task"]
    cfg.pop("arch")
    cfg.pop("task")

    if task == "classification":
        model = models.classification.__dict__[arch](model_path, classes=cfg["classes"], engine_cfg=engine_cfg)
    elif task == "detection":
        model = models.detection.__dict__[arch](model_path, engine_cfg=engine_cfg)
    elif task == "recognition":
        model = models.recognition.__dict__[arch](
            model_path, input_shape=cfg["input_shape"], vocab=cfg["vocab"], engine_cfg=engine_cfg
        )

    # convert all values which are lists to tuples
    for key, value in cfg.items():
        if isinstance(value, list):
            cfg[key] = tuple(value)
    # update model cfg
    model.cfg = cfg

    return model
