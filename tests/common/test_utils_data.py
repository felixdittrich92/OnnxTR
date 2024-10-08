import os
import tempfile
from pathlib import PosixPath
from unittest.mock import patch

import pytest

from onnxtr.utils.data import _urlretrieve, download_from_url


def test__urlretrieve():
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "crnn_mobilenet_v3_small-bded4d49.onnx")
        _urlretrieve(
            "https://github.com/felixdittrich92/OnnxTR/releases/download/v0.0.1/crnn_mobilenet_v3_small-bded4d49.onnx",
            file_path,
        )
        assert os.path.exists(file_path), f"File {file_path} does not exist."


@patch("onnxtr.utils.data._urlretrieve")
@patch("pathlib.Path.mkdir")
@patch.dict(os.environ, {"HOME": "/"}, clear=True)
def test_download_from_url(mkdir_mock, urlretrieve_mock):
    download_from_url("test_url")
    urlretrieve_mock.assert_called_with("test_url", PosixPath("/.cache/onnxtr/test_url"))


@patch.dict(os.environ, {"ONNXTR_CACHE_DIR": "/test"}, clear=True)
@patch("onnxtr.utils.data._urlretrieve")
@patch("pathlib.Path.mkdir")
def test_download_from_url_customizing_cache_dir(mkdir_mock, urlretrieve_mock):
    download_from_url("test_url")
    urlretrieve_mock.assert_called_with("test_url", PosixPath("/test/test_url"))


@patch.dict(os.environ, {"HOME": "/"}, clear=True)
@patch("pathlib.Path.mkdir", side_effect=OSError)
@patch("logging.error")
def test_download_from_url_error_creating_directory(logging_mock, mkdir_mock):
    with pytest.raises(OSError):
        download_from_url("test_url")
    logging_mock.assert_called_with(
        "Failed creating cache direcotry at /.cache/onnxtr."
        " You can change default cache directory using 'ONNXTR_CACHE_DIR' environment variable if needed."
    )


@patch.dict(os.environ, {"HOME": "/", "ONNXTR_CACHE_DIR": "/test"}, clear=True)
@patch("pathlib.Path.mkdir", side_effect=OSError)
@patch("logging.error")
def test_download_from_url_error_creating_directory_with_env_var(logging_mock, mkdir_mock):
    with pytest.raises(OSError):
        download_from_url("test_url")
    logging_mock.assert_called_with(
        "Failed creating cache direcotry at /test using path from 'ONNXTR_CACHE_DIR' environment variable."
    )
