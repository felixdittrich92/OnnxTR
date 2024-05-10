# Copyright (C) 2021-2024, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, Dict, List, Optional, Tuple, Union

from defusedxml import defuse_stdlib

defuse_stdlib()
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element as ETElement
from xml.etree.ElementTree import SubElement

import numpy as np

import onnxtr
from onnxtr.file_utils import requires_package
from onnxtr.utils.common_types import BoundingBox
from onnxtr.utils.geometry import resolve_enclosing_bbox, resolve_enclosing_rbbox
from onnxtr.utils.reconstitution import synthesize_page
from onnxtr.utils.repr import NestedObject

try:  # optional dependency for visualization
    from onnxtr.utils.visualization import visualize_page
except ModuleNotFoundError:  # pragma: no cover
    pass

__all__ = ["Element", "Word", "Artefact", "Line", "Block", "Page", "Document"]


class Element(NestedObject):
    """Implements an abstract document element with exporting and text rendering capabilities"""

    _children_names: List[str] = []
    _exported_keys: List[str] = []

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in self._children_names:
                setattr(self, k, v)
            else:
                raise KeyError(f"{self.__class__.__name__} object does not have any attribute named '{k}'")

    def export(self) -> Dict[str, Any]:
        """Exports the object into a nested dict format"""
        export_dict = {k: getattr(self, k) for k in self._exported_keys}
        for children_name in self._children_names:
            export_dict[children_name] = [c.export() for c in getattr(self, children_name)]

        return export_dict

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
        raise NotImplementedError

    def render(self) -> str:
        raise NotImplementedError


class Word(Element):
    """Implements a word element

    Args:
    ----
        value: the text string of the word
        confidence: the confidence associated with the text prediction
        geometry: bounding box of the word in format ((xmin, ymin), (xmax, ymax)) where coordinates are relative to
        the page's size
        crop_orientation: the general orientation of the crop in degrees and its confidence
    """

    _exported_keys: List[str] = ["value", "confidence", "geometry", "crop_orientation"]
    _children_names: List[str] = []

    def __init__(
        self,
        value: str,
        confidence: float,
        geometry: Union[BoundingBox, np.ndarray],
        crop_orientation: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.value = value
        self.confidence = confidence
        self.geometry = geometry
        self.crop_orientation = crop_orientation

    def render(self) -> str:
        """Renders the full text of the element"""
        return self.value

    def extra_repr(self) -> str:
        return f"value='{self.value}', confidence={self.confidence:.2}"

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        return cls(**kwargs)


class Artefact(Element):
    """Implements a non-textual element

    Args:
    ----
        artefact_type: the type of artefact
        confidence: the confidence of the type prediction
        geometry: bounding box of the word in format ((xmin, ymin), (xmax, ymax)) where coordinates are relative to
            the page's size.
    """

    _exported_keys: List[str] = ["geometry", "type", "confidence"]
    _children_names: List[str] = []

    def __init__(self, artefact_type: str, confidence: float, geometry: BoundingBox) -> None:
        super().__init__()
        self.geometry = geometry
        self.type = artefact_type
        self.confidence = confidence

    def render(self) -> str:
        """Renders the full text of the element"""
        return f"[{self.type.upper()}]"

    def extra_repr(self) -> str:
        return f"type='{self.type}', confidence={self.confidence:.2}"

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        return cls(**kwargs)


class Line(Element):
    """Implements a line element as a collection of words

    Args:
    ----
        words: list of word elements
        geometry: bounding box of the word in format ((xmin, ymin), (xmax, ymax)) where coordinates are relative to
            the page's size. If not specified, it will be resolved by default to the smallest bounding box enclosing
            all words in it.
    """

    _exported_keys: List[str] = ["geometry"]
    _children_names: List[str] = ["words"]
    words: List[Word] = []

    def __init__(
        self,
        words: List[Word],
        geometry: Optional[Union[BoundingBox, np.ndarray]] = None,
    ) -> None:
        # Resolve the geometry using the smallest enclosing bounding box
        if geometry is None:
            # Check whether this is a rotated or straight box
            box_resolution_fn = resolve_enclosing_rbbox if len(words[0].geometry) == 4 else resolve_enclosing_bbox
            geometry = box_resolution_fn([w.geometry for w in words])  # type: ignore[operator]

        super().__init__(words=words)
        self.geometry = geometry

    def render(self) -> str:
        """Renders the full text of the element"""
        return " ".join(w.render() for w in self.words)

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        kwargs.update({
            "words": [Word.from_dict(_dict) for _dict in save_dict["words"]],
        })
        return cls(**kwargs)


class Block(Element):
    """Implements a block element as a collection of lines and artefacts

    Args:
    ----
        lines: list of line elements
        artefacts: list of artefacts
        geometry: bounding box of the word in format ((xmin, ymin), (xmax, ymax)) where coordinates are relative to
            the page's size. If not specified, it will be resolved by default to the smallest bounding box enclosing
            all lines and artefacts in it.
    """

    _exported_keys: List[str] = ["geometry"]
    _children_names: List[str] = ["lines", "artefacts"]
    lines: List[Line] = []
    artefacts: List[Artefact] = []

    def __init__(
        self,
        lines: List[Line] = [],
        artefacts: List[Artefact] = [],
        geometry: Optional[Union[BoundingBox, np.ndarray]] = None,
    ) -> None:
        # Resolve the geometry using the smallest enclosing bounding box
        if geometry is None:
            line_boxes = [word.geometry for line in lines for word in line.words]
            artefact_boxes = [artefact.geometry for artefact in artefacts]
            box_resolution_fn = (
                resolve_enclosing_rbbox if isinstance(lines[0].geometry, np.ndarray) else resolve_enclosing_bbox
            )
            geometry = box_resolution_fn(line_boxes + artefact_boxes)  # type: ignore[operator]

        super().__init__(lines=lines, artefacts=artefacts)
        self.geometry = geometry

    def render(self, line_break: str = "\n") -> str:
        """Renders the full text of the element"""
        return line_break.join(line.render() for line in self.lines)

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        kwargs.update({
            "lines": [Line.from_dict(_dict) for _dict in save_dict["lines"]],
            "artefacts": [Artefact.from_dict(_dict) for _dict in save_dict["artefacts"]],
        })
        return cls(**kwargs)


class Page(Element):
    """Implements a page element as a collection of blocks

    Args:
    ----
        page: image encoded as a numpy array in uint8
        blocks: list of block elements
        page_idx: the index of the page in the input raw document
        dimensions: the page size in pixels in format (height, width)
        orientation: a dictionary with the value of the rotation angle in degress and confidence of the prediction
        language: a dictionary with the language value and confidence of the prediction
    """

    _exported_keys: List[str] = ["page_idx", "dimensions", "orientation", "language"]
    _children_names: List[str] = ["blocks"]
    blocks: List[Block] = []

    def __init__(
        self,
        page: np.ndarray,
        blocks: List[Block],
        page_idx: int,
        dimensions: Tuple[int, int],
        orientation: Optional[Dict[str, Any]] = None,
        language: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(blocks=blocks)
        self.page = page
        self.page_idx = page_idx
        self.dimensions = dimensions
        self.orientation = orientation if isinstance(orientation, dict) else dict(value=None, confidence=None)
        self.language = language if isinstance(language, dict) else dict(value=None, confidence=None)

    def render(self, block_break: str = "\n\n") -> str:
        """Renders the full text of the element"""
        return block_break.join(b.render() for b in self.blocks)

    def extra_repr(self) -> str:
        return f"dimensions={self.dimensions}"

    def show(self, interactive: bool = True, preserve_aspect_ratio: bool = False, **kwargs) -> None:
        """Overlay the result on a given image

        Args:
            interactive: whether the display should be interactive
            preserve_aspect_ratio: pass True if you passed True to the predictor
            **kwargs: additional keyword arguments passed to the matplotlib.pyplot.show method
        """
        requires_package("matplotlib", "`.show()` requires matplotlib & mplcursors installed")
        requires_package("mplcursors", "`.show()` requires matplotlib & mplcursors installed")
        import matplotlib.pyplot as plt

        visualize_page(self.export(), self.page, interactive=interactive, preserve_aspect_ratio=preserve_aspect_ratio)
        plt.show(**kwargs)

    def synthesize(self, **kwargs) -> np.ndarray:
        """Synthesize the page from the predictions

        Returns
        -------
            synthesized page
        """
        return synthesize_page(self.export(), **kwargs)

    def export_as_xml(self, file_title: str = "OnnxTR - XML export (hOCR)") -> Tuple[bytes, ET.ElementTree]:
        """Export the page as XML (hOCR-format)
        convention: https://github.com/kba/hocr-spec/blob/master/1.2/spec.md

        Args:
        ----
            file_title: the title of the XML file

        Returns:
        -------
            a tuple of the XML byte string, and its ElementTree
        """
        p_idx = self.page_idx
        block_count: int = 1
        line_count: int = 1
        word_count: int = 1
        height, width = self.dimensions
        language = self.language if "language" in self.language.keys() else "en"
        # Create the XML root element
        page_hocr = ETElement("html", attrib={"xmlns": "http://www.w3.org/1999/xhtml", "xml:lang": str(language)})
        # Create the header / SubElements of the root element
        head = SubElement(page_hocr, "head")
        SubElement(head, "title").text = file_title
        SubElement(head, "meta", attrib={"http-equiv": "Content-Type", "content": "text/html; charset=utf-8"})
        SubElement(
            head,
            "meta",
            attrib={"name": "ocr-system", "content": f" {onnxtr.__version__}"},  # type: ignore[attr-defined]
        )
        SubElement(
            head,
            "meta",
            attrib={"name": "ocr-capabilities", "content": "ocr_page ocr_carea ocr_par ocr_line ocrx_word"},
        )
        # Create the body
        body = SubElement(page_hocr, "body")
        SubElement(
            body,
            "div",
            attrib={
                "class": "ocr_page",
                "id": f"page_{p_idx + 1}",
                "title": f"image; bbox 0 0 {width} {height}; ppageno 0",
            },
        )
        # iterate over the blocks / lines / words and create the XML elements in body line by line with the attributes
        for block in self.blocks:
            if len(block.geometry) != 2:
                raise TypeError("XML export is only available for straight bounding boxes for now.")
            (xmin, ymin), (xmax, ymax) = block.geometry
            block_div = SubElement(
                body,
                "div",
                attrib={
                    "class": "ocr_carea",
                    "id": f"block_{block_count}",
                    "title": f"bbox {int(round(xmin * width))} {int(round(ymin * height))} \
                    {int(round(xmax * width))} {int(round(ymax * height))}",
                },
            )
            paragraph = SubElement(
                block_div,
                "p",
                attrib={
                    "class": "ocr_par",
                    "id": f"par_{block_count}",
                    "title": f"bbox {int(round(xmin * width))} {int(round(ymin * height))} \
                    {int(round(xmax * width))} {int(round(ymax * height))}",
                },
            )
            block_count += 1
            for line in block.lines:
                (xmin, ymin), (xmax, ymax) = line.geometry
                # NOTE: baseline, x_size, x_descenders, x_ascenders is currently initalized to 0
                line_span = SubElement(
                    paragraph,
                    "span",
                    attrib={
                        "class": "ocr_line",
                        "id": f"line_{line_count}",
                        "title": f"bbox {int(round(xmin * width))} {int(round(ymin * height))} \
                        {int(round(xmax * width))} {int(round(ymax * height))}; \
                        baseline 0 0; x_size 0; x_descenders 0; x_ascenders 0",
                    },
                )
                line_count += 1
                for word in line.words:
                    (xmin, ymin), (xmax, ymax) = word.geometry
                    conf = word.confidence
                    word_div = SubElement(
                        line_span,
                        "span",
                        attrib={
                            "class": "ocrx_word",
                            "id": f"word_{word_count}",
                            "title": f"bbox {int(round(xmin * width))} {int(round(ymin * height))} \
                            {int(round(xmax * width))} {int(round(ymax * height))}; \
                            x_wconf {int(round(conf * 100))}",
                        },
                    )
                    # set the text
                    word_div.text = word.value
                    word_count += 1

        return (ET.tostring(page_hocr, encoding="utf-8", method="xml"), ET.ElementTree(page_hocr))

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        kwargs.update({"blocks": [Block.from_dict(block_dict) for block_dict in save_dict["blocks"]]})
        return cls(**kwargs)


class Document(Element):
    """Implements a document element as a collection of pages

    Args:
    ----
        pages: list of page elements
    """

    _children_names: List[str] = ["pages"]
    pages: List[Page] = []

    def __init__(
        self,
        pages: List[Page],
    ) -> None:
        super().__init__(pages=pages)

    def render(self, page_break: str = "\n\n\n\n") -> str:
        """Renders the full text of the element"""
        return page_break.join(p.render() for p in self.pages)

    def show(self, **kwargs) -> None:
        """Overlay the result on a given image"""
        for result in self.pages:
            result.show(**kwargs)

    def synthesize(self, **kwargs) -> List[np.ndarray]:
        """Synthesize all pages from their predictions

        Returns
        -------
            list of synthesized pages
        """
        return [page.synthesize() for page in self.pages]

    def export_as_xml(self, **kwargs) -> List[Tuple[bytes, ET.ElementTree]]:
        """Export the document as XML (hOCR-format)

        Args:
        ----
            **kwargs: additional keyword arguments passed to the Page.export_as_xml method

        Returns:
        -------
            list of tuple of (bytes, ElementTree)
        """
        return [page.export_as_xml(**kwargs) for page in self.pages]

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        kwargs.update({"pages": [Page.from_dict(page_dict) for page_dict in save_dict["pages"]]})
        return cls(**kwargs)
