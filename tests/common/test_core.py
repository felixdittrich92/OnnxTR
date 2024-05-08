import pytest

import onnxtr
from onnxtr.file_utils import requires_package


def test_version():
    assert len(onnxtr.__version__.split(".")) == 3


def test_requires_package():
    requires_package("numpy")  # availbable
    with pytest.raises(ImportError):  # not available
        requires_package("non_existent_package")
