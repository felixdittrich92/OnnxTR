from xml.etree.ElementTree import ElementTree

import numpy as np
import pytest

from onnxtr.io import elements


def _mock_words(size=(1.0, 1.0), offset=(0, 0), confidence=0.9, objectness_score=0.9, polygons=False):
    box_word_elements = [
        elements.Word(
            "hello",
            confidence,
            ((offset[0], offset[1]), (size[0] / 2 + offset[0], size[1] / 2 + offset[1])),
            objectness_score,
            {"value": 0, "confidence": None},
        ),
        elements.Word(
            "world",
            confidence,
            ((size[0] / 2 + offset[0], size[1] / 2 + offset[1]), (size[0] + offset[0], size[1] + offset[1])),
            objectness_score,
            {"value": 0, "confidence": None},
        ),
    ]
    polygons_word_elements = [
        elements.Word(
            "hello",
            confidence,
            # (x1, y1), (x2, y2), (x3, y3), (x4, y4) with shape (4, 2)
            np.array([
                [offset[0], offset[1]],
                [size[0] / 2 + offset[0], offset[1]],
                [size[0] / 2 + offset[0], size[1] / 2 + offset[1]],
                [offset[0], size[1] / 2 + offset[1]],
            ]),
            objectness_score,
            {"value": 0, "confidence": None},
        ),
        elements.Word(
            "world",
            confidence,
            # (x1, y1), (x2, y2), (x3, y3), (x4, y4) with shape (4, 2)
            np.array([
                [size[0] / 2 + offset[0], size[1] / 2 + offset[1]],
                [size[0] + offset[0], size[1] / 2 + offset[1]],
                [size[0] + offset[0], size[1] + offset[1]],
                [size[0] / 2 + offset[0], size[1] + offset[1]],
            ]),
            objectness_score,
            {"value": 0, "confidence": None},
        ),
    ]
    return polygons_word_elements if polygons else box_word_elements


def _mock_artefacts(size=(1, 1), offset=(0, 0), confidence=0.8):
    sub_size = (size[0] / 2, size[1] / 2)
    return [
        elements.Artefact(
            "qr_code", confidence, ((offset[0], offset[1]), (sub_size[0] + offset[0], sub_size[1] + offset[1]))
        ),
        elements.Artefact(
            "qr_code",
            confidence,
            ((sub_size[0] + offset[0], sub_size[1] + offset[1]), (size[0] + offset[0], size[1] + offset[1])),
        ),
    ]


def _mock_lines(size=(1, 1), offset=(0, 0), polygons=False):
    sub_size = (size[0] / 2, size[1] / 2)
    return [
        elements.Line(_mock_words(size=sub_size, offset=offset, polygons=polygons)),
        elements.Line(
            _mock_words(size=sub_size, offset=(offset[0] + sub_size[0], offset[1] + sub_size[1]), polygons=polygons)
        ),
    ]


def _mock_blocks(size=(1, 1), offset=(0, 0), polygons=False):
    sub_size = (size[0] / 4, size[1] / 4)
    return [
        elements.Block(
            _mock_lines(size=sub_size, offset=offset, polygons=polygons),
            _mock_artefacts(size=sub_size, offset=(offset[0] + sub_size[0], offset[1] + sub_size[1])),
        ),
        elements.Block(
            _mock_lines(
                size=sub_size, offset=(offset[0] + 2 * sub_size[0], offset[1] + 2 * sub_size[1]), polygons=polygons
            ),
            _mock_artefacts(size=sub_size, offset=(offset[0] + 3 * sub_size[0], offset[1] + 3 * sub_size[1])),
        ),
    ]


def _mock_pages(block_size=(1, 1), block_offset=(0, 0), polygons=False):
    return [
        elements.Page(
            np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8),
            _mock_blocks(block_size, block_offset, polygons),
            0,
            (300, 200),
            {"value": 0.0, "confidence": 1.0},
            {"value": "EN", "confidence": 0.8},
        ),
        elements.Page(
            np.random.randint(0, 255, (500, 1000, 3), dtype=np.uint8),
            _mock_blocks(block_size, block_offset),
            1,
            (500, 1000),
            {"value": 0.15, "confidence": 0.8},
            {"value": "FR", "confidence": 0.7},
        ),
    ]


def test_element():
    with pytest.raises(KeyError):
        elements.Element(sub_elements=[1])


def test_word():
    word_str = "hello"
    conf = 0.8
    geom = ((0, 0), (1, 1))
    objectness_score = 0.9
    crop_orientation = {"value": 0, "confidence": None}
    word = elements.Word(word_str, conf, geom, objectness_score, crop_orientation)

    # Attribute checks
    assert word.value == word_str
    assert word.confidence == conf
    assert word.geometry == geom
    assert word.objectness_score == objectness_score
    assert word.crop_orientation == crop_orientation

    # Render
    assert word.render() == word_str

    # Export
    assert word.export() == {
        "value": word_str,
        "confidence": conf,
        "geometry": geom,
        "objectness_score": objectness_score,
        "crop_orientation": crop_orientation,
    }

    # Repr
    assert word.__repr__() == f"Word(value='hello', confidence={conf:.2})"

    # Class method
    state_dict = {
        "value": "there",
        "confidence": 0.1,
        "geometry": ((0, 0), (0.5, 0.5)),
        "objectness_score": objectness_score,
        "crop_orientation": crop_orientation,
    }
    word = elements.Word.from_dict(state_dict)
    assert word.export() == state_dict


def test_line():
    geom = ((0, 0), (0.5, 0.5))
    objectness_score = 0.9
    words = _mock_words(size=geom[1], offset=geom[0])
    line = elements.Line(words)

    # Attribute checks
    assert len(line.words) == len(words)
    assert all(isinstance(w, elements.Word) for w in line.words)
    assert line.geometry == geom
    assert line.objectness_score == objectness_score

    # Render
    assert line.render() == "hello world"

    # Export
    assert line.export() == {
        "words": [w.export() for w in words],
        "geometry": geom,
        "objectness_score": objectness_score,
    }

    # Repr
    words_str = " " * 4 + ",\n    ".join(repr(word) for word in words) + ","
    assert line.__repr__() == f"Line(\n  (words): [\n{words_str}\n  ]\n)"

    # Ensure that words repr does't span on several lines when there are none
    assert repr(elements.Line([], ((0, 0), (1, 1)))) == "Line(\n  (words): []\n)"

    # from dict
    state_dict = {
        "words": [
            {
                "value": "there",
                "confidence": 0.1,
                "geometry": ((0, 0), (1.0, 1.0)),
                "objectness_score": objectness_score,
                "crop_orientation": {"value": 0, "confidence": None},
            }
        ],
        "geometry": ((0, 0), (1.0, 1.0)),
        "objectness_score": objectness_score,
    }
    line = elements.Line.from_dict(state_dict)
    assert line.export() == state_dict


def test_artefact():
    artefact_type = "qr_code"
    conf = 0.8
    geom = ((0, 0), (1, 1))
    artefact = elements.Artefact(artefact_type, conf, geom)

    # Attribute checks
    assert artefact.type == artefact_type
    assert artefact.confidence == conf
    assert artefact.geometry == geom

    # Render
    assert artefact.render() == "[QR_CODE]"

    # Export
    assert artefact.export() == {"type": artefact_type, "confidence": conf, "geometry": geom}

    # Repr
    assert artefact.__repr__() == f"Artefact(type='{artefact_type}', confidence={conf:.2})"


def test_block():
    geom = ((0, 0), (1, 1))
    sub_size = (geom[1][0] / 2, geom[1][0] / 2)
    objectness_score = 0.9
    lines = _mock_lines(size=sub_size, offset=geom[0])
    artefacts = _mock_artefacts(size=sub_size, offset=sub_size)
    block = elements.Block(lines, artefacts)

    # Attribute checks
    assert len(block.lines) == len(lines)
    assert len(block.artefacts) == len(artefacts)
    assert all(isinstance(w, elements.Line) for w in block.lines)
    assert all(isinstance(a, elements.Artefact) for a in block.artefacts)
    assert block.geometry == geom

    # Render
    assert block.render() == "hello world\nhello world"

    # Export
    assert block.export() == {
        "lines": [line.export() for line in lines],
        "artefacts": [artefact.export() for artefact in artefacts],
        "geometry": geom,
        "objectness_score": objectness_score,
    }


def test_page():
    page = np.zeros((300, 200, 3), dtype=np.uint8)
    page_idx = 0
    page_size = (300, 200)
    orientation = {"value": 0.0, "confidence": 0.0}
    language = {"value": "EN", "confidence": 0.8}
    blocks = _mock_blocks()
    page = elements.Page(page, blocks, page_idx, page_size, orientation, language)

    # Attribute checks
    assert len(page.blocks) == len(blocks)
    assert all(isinstance(b, elements.Block) for b in page.blocks)
    assert isinstance(page.page, np.ndarray)
    assert page.page_idx == page_idx
    assert page.dimensions == page_size
    assert page.orientation == orientation
    assert page.language == language

    # Render
    assert page.render() == "hello world\nhello world\n\nhello world\nhello world"

    # Export
    assert page.export() == {
        "blocks": [b.export() for b in blocks],
        "page_idx": page_idx,
        "dimensions": page_size,
        "orientation": orientation,
        "language": language,
    }

    # Export XML
    assert (
        isinstance(page.export_as_xml(), tuple)
        and isinstance(page.export_as_xml()[0], (bytes, bytearray))
        and isinstance(page.export_as_xml()[1], ElementTree)
    )

    # Repr
    assert "\n".join(repr(page).split("\n")[:2]) == f"Page(\n  dimensions={page_size!r}"

    # Show
    page.show(block=False)

    # Synthesize
    img = page.synthesize()
    assert isinstance(img, np.ndarray)
    assert img.shape == (*page_size, 3)


def test_document():
    pages = _mock_pages()
    doc = elements.Document(pages)

    # Attribute checks
    assert len(doc.pages) == len(pages)
    assert all(isinstance(p, elements.Page) for p in doc.pages)

    # Render
    page_export = "hello world\nhello world\n\nhello world\nhello world"
    assert doc.render() == f"{page_export}\n\n\n\n{page_export}"

    # Export
    assert doc.export() == {"pages": [p.export() for p in pages]}

    # Export XML
    xml_output = doc.export_as_xml()
    assert isinstance(xml_output, list) and len(xml_output) == len(pages)
    # Check that the XML is well-formed in hOCR format
    for xml_bytes, xml_tree in xml_output:
        assert isinstance(xml_bytes, bytes)
        assert isinstance(xml_tree, ElementTree)
        root = xml_tree.getroot()
        assert root.tag == "html"
        assert root[0].tag == "head"
        assert root[1].tag == "body"
        assert root[1][0].tag == "div" and root[1][0].attrib["class"] == "ocr_page"
        for block in root[1][0]:
            assert block.tag == "div" and block.attrib["class"] == "ocr_carea"
            assert block[0].tag == "p" and block[0].attrib["class"] == "ocr_par"
            for line in block[0]:
                assert line.tag == "span" and line.attrib["class"] == "ocr_line"
                for word in line:
                    assert word.tag == "span" and word.attrib["class"] == "ocrx_word"

    # Show
    doc.show(block=False)

    # Synthesize
    img_list = doc.synthesize()
    assert isinstance(img_list, list) and len(img_list) == len(pages)
