import numpy as np
import pytest

from onnxtr.transforms import Normalize, Resize
from onnxtr.utils import Sample


def test_resize():
    output_size = (32, 32)
    transfo = Resize(output_size)
    input_t = np.ones((64, 64, 3), dtype=np.float32)
    out = transfo(Sample(image=input_t)).image

    assert np.all(out == 255)
    assert out.shape[:2] == output_size
    assert repr(transfo) == f"Resize(output_size={output_size}, interpolation='2')"

    transfo = Resize(output_size, preserve_aspect_ratio=True)
    input_t = np.ones((32, 64, 3), dtype=np.float32)
    out = transfo(Sample(image=input_t)).image

    assert out.shape[:2] == output_size
    assert not np.all(out == 255)
    # Asymetric padding
    assert np.all(out[-1] == 0) and np.all(out[0] == 255)

    # Symetric padding
    transfo = Resize(output_size, preserve_aspect_ratio=True, symmetric_pad=True)
    assert repr(transfo) == (
        f"Resize(output_size={output_size}, interpolation='2', preserve_aspect_ratio=True, symmetric_pad=True)"
    )
    out = transfo(Sample(image=input_t)).image
    assert out.shape[:2] == output_size
    # symetric padding
    assert np.all(out[-1] == 0) and np.all(out[0] == 0)

    # Inverse aspect ratio
    input_t = np.ones((64, 32, 3), dtype=np.float32)
    out = transfo(Sample(image=input_t)).image

    assert not np.all(out == 1)
    assert out.shape[:2] == output_size

    # Same aspect ratio
    output_size = (32, 128)
    transfo = Resize(output_size, preserve_aspect_ratio=True)
    out = transfo(Sample(image=np.ones((16, 64, 3), dtype=np.float32))).image
    assert out.shape[:2] == output_size


def test_sample():
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    sample = Sample(image=img)
    assert sample.mask is None and sample.target is None

    # `replace` returns a new Sample and leaves the original untouched
    mask = np.ones((8, 8), dtype=bool)
    other = sample.replace(mask=mask)
    assert other is not sample
    assert sample.mask is None
    assert other.mask is mask and other.image is img
    # unspecified fields are carried over
    target = np.zeros((1, 4), dtype=np.float32)
    third = other.replace(target=target)
    assert third.image is img and third.mask is mask and third.target is target


def test_resize_padding_mask():
    output_size = (32, 32)

    # Plain (stretching) resize adds no padding -> every pixel is valid
    transfo = Resize(output_size, return_padding_mask=True)
    out = transfo(Sample(image=np.ones((64, 64, 3), dtype=np.float32)))
    assert out.image.shape[:2] == output_size
    assert out.mask.shape == output_size and out.mask.dtype == bool
    assert out.mask.all()
    assert "return_padding_mask=True" in repr(transfo)

    # Asymmetric padding -> the mask is False on the padded bottom rows
    transfo = Resize(output_size, preserve_aspect_ratio=True, return_padding_mask=True)
    out = transfo(Sample(image=np.ones((32, 64, 3), dtype=np.float32)))
    assert out.image.shape[:2] == output_size and out.mask.shape == output_size
    assert out.mask[0].all() and not out.mask[-1].any()
    # The mask marks exactly the non-padded content
    assert np.array_equal(out.mask, out.image.any(axis=-1) > 0)

    # Symmetric padding -> both ends are masked out
    transfo = Resize(output_size, preserve_aspect_ratio=True, symmetric_pad=True, return_padding_mask=True)
    out = transfo(Sample(image=np.ones((32, 64, 3), dtype=np.float32)))
    assert not out.mask[0].any() and not out.mask[-1].any()
    assert out.mask[output_size[0] // 2].all()

    # Inverse aspect ratio -> padding on the columns instead
    out = transfo(Sample(image=np.ones((64, 32, 3), dtype=np.float32)))
    assert not out.mask[:, 0].any() and not out.mask[:, -1].any()
    assert out.mask[:, output_size[1] // 2].all()

    # Without the flag no mask is synthesised
    out = Resize(output_size, preserve_aspect_ratio=True)(Sample(image=np.ones((32, 64, 3), dtype=np.float32)))
    assert out.mask is None


def test_resize_incoming_mask():
    """An incoming mask is resized alongside the image instead of being replaced."""
    output_size = (32, 32)
    img = np.ones((32, 64, 3), dtype=np.uint8)
    # a label mask: left half 1, right half 2
    mask = np.full((32, 64), 2, dtype=np.uint8)
    mask[:, :32] = 1

    transfo = Resize(output_size, preserve_aspect_ratio=True, symmetric_pad=True, return_padding_mask=True)
    out = transfo(Sample(image=img, mask=mask))

    assert out.mask.shape == output_size
    # nearest interpolation -> no interpolated label values are invented
    assert set(np.unique(out.mask)) <= {0, 1, 2}
    # the two label regions survive, and the padded rows are zero
    content = out.mask[out.mask.any(axis=1)]
    assert 1 in content and 2 in content
    assert (out.mask[0] == 0).all() and (out.mask[-1] == 0).all()

    # Same for the plain resize branch
    out = Resize(output_size, return_padding_mask=True)(Sample(image=img, mask=mask))
    assert out.mask.shape == output_size
    assert set(np.unique(out.mask)) <= {1, 2}


@pytest.mark.parametrize("use_polygons", [False, True])
def test_resize_target(use_polygons):
    """Relative boxes are rescaled to match the padded image."""
    output_size = (32, 32)
    img = np.ones((32, 64, 3), dtype=np.uint8)  # landscape -> padded top/bottom
    boxes = (
        np.array([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], dtype=np.float32)
        if use_polygons
        else np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    )

    # No aspect-ratio preservation -> nothing to rescale
    out = Resize(output_size)(Sample(image=img, target=boxes.copy()))
    assert np.allclose(out.target, boxes)

    # Symmetric padding -> y shrinks by half and is offset
    out = Resize(output_size, preserve_aspect_ratio=True, symmetric_pad=True)(Sample(image=img, target=boxes.copy()))
    ys = out.target[..., 1] if use_polygons else out.target[:, [1, 3]]
    xs = out.target[..., 0] if use_polygons else out.target[:, [0, 2]]
    assert np.allclose(sorted(np.unique(ys)), [0.25, 0.75])
    assert np.allclose(sorted(np.unique(xs)), [0.0, 1.0])
    assert out.target.min() >= 0 and out.target.max() <= 1

    # A dict of targets is rescaled per class
    out = Resize(output_size, preserve_aspect_ratio=True, symmetric_pad=True)(
        Sample(image=img, target={"words": boxes.copy()})
    )
    assert isinstance(out.target, dict) and "words" in out.target

    # Bad box shape
    with pytest.raises(AssertionError):
        Resize(output_size, preserve_aspect_ratio=True)(Sample(image=img, target=np.zeros((1, 5), dtype=np.float32)))


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

    with pytest.raises(AssertionError):
        Normalize(mean="32")

    with pytest.raises(AssertionError):
        Normalize(std="32")
