import numpy as np
from test_io_elements import _mock_pages

from onnxtr.utils import reconstitution


def test_synthesize_page():
    pages = _mock_pages()
    # Test without probability rendering
    render_no_proba = reconstitution.synthesize_page(pages[0].export(), draw_proba=False)
    assert isinstance(render_no_proba, np.ndarray)
    assert render_no_proba.shape == (*pages[0].dimensions, 3)

    # Test with probability rendering
    render_with_proba = reconstitution.synthesize_page(pages[0].export(), draw_proba=True)
    assert isinstance(render_with_proba, np.ndarray)
    assert render_with_proba.shape == (*pages[0].dimensions, 3)

    # Test with only one line
    pages_one_line = pages[0].export()
    pages_one_line["blocks"][0]["lines"] = [pages_one_line["blocks"][0]["lines"][0]]
    render_one_line = reconstitution.synthesize_page(pages_one_line, draw_proba=True)
    assert isinstance(render_one_line, np.ndarray)
    assert render_one_line.shape == (*pages[0].dimensions, 3)
