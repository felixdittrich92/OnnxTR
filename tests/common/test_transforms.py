import numpy as np
import pytest

from onnxtr.transforms import Normalize, Resize


def test_resize():
    output_size = (32, 32)
    transfo = Resize(output_size)
    input_t = np.ones((64, 64, 3), dtype=np.float32)
    out = transfo(input_t)

    assert np.all(out == 255)
    assert out.shape[:2] == output_size
    assert repr(transfo) == f"Resize(output_size={output_size}, interpolation='2')"

    transfo = Resize(output_size, preserve_aspect_ratio=True)
    input_t = np.ones((32, 64, 3), dtype=np.float32)
    out = transfo(input_t)

    assert out.shape[:2] == output_size
    assert not np.all(out == 255)
    # Asymetric padding
    assert np.all(out[-1] == 0) and np.all(out[0] == 255)

    # Symetric padding
    transfo = Resize(output_size, preserve_aspect_ratio=True, symmetric_pad=True)
    assert repr(transfo) == (
        f"Resize(output_size={output_size}, interpolation='2', " f"preserve_aspect_ratio=True, symmetric_pad=True)"
    )
    out = transfo(input_t)
    assert out.shape[:2] == output_size
    # symetric padding
    assert np.all(out[-1] == 0) and np.all(out[0] == 0)

    # Inverse aspect ratio
    input_t = np.ones((64, 32, 3), dtype=np.float32)
    out = transfo(input_t)

    assert not np.all(out == 1)
    assert out.shape[:2] == output_size

    # Same aspect ratio
    output_size = (32, 128)
    transfo = Resize(output_size, preserve_aspect_ratio=True)
    out = transfo(np.ones((16, 64, 3), dtype=np.float32))
    assert out.shape[:2] == output_size


@pytest.mark.parametrize(
    "input_shape",
    [
        [8, 32, 32, 3],
        [32, 32, 3],
        [32, 3],
    ],
)
def test_normalize(input_shape):
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    transfo = Normalize(mean, std)
    input_t = np.ones(input_shape, dtype=np.float32)

    out = transfo(input_t)

    assert np.all(out == 1)
    assert repr(transfo) == f"Normalize(mean={mean}, std={std})"
