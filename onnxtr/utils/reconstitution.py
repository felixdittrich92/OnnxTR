# Copyright (C) 2021-2024, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.
from typing import Any, Dict, Optional

import numpy as np
from anyascii import anyascii
from PIL import Image, ImageDraw

from .fonts import get_font

__all__ = ["synthesize_page"]


def synthesize_page(
    page: Dict[str, Any],
    draw_proba: bool = False,
    font_family: Optional[str] = None,
) -> np.ndarray:
    """Draw the content of the element page (OCR response) on a blank page.

    Args:
    ----
        page: exported Page object to represent
        draw_proba: if True, draw words in colors to represent confidence. Blue: p=1, red: p=0
        font_size: size of the font, default font = 13
        font_family: family of the font

    Returns:
    -------
        the synthesized page
    """
    # Draw template
    h, w = page["dimensions"]
    response = 255 * np.ones((h * 2, w *2, 3), dtype=np.int32)

    # Draw each line
    for block in page["blocks"]:
        for line in block["lines"]:
            # Get absolute line geometry
            (xmin, ymin), (xmax, ymax) = line["geometry"]
            xmin, xmax = int(round(w * xmin)), int(round(w * xmax))
            ymin, ymax = int(round(h * ymin)), int(round(h * ymax))

            # Concatenate words to form the line text
            line_text = ' '.join([word["value"] for word in line["words"]])

            # White drawing context adapted to font size, 0.75 factor to convert pts --> pix
            font = get_font(font_family, int(0.75 * (ymax - ymin)))
            d = ImageDraw.Draw(Image.new("RGB", (1, 1)))  # Temporary context for sizing
            text_size = font.getbbox(line_text)[-2:]

            # Resize xmax/ymax to fit the text
            img_w, img_h = text_size
            if img_w > (xmax - xmin):
                xmax = xmin + img_w
            if img_h > (ymax - ymin):
                ymax = ymin + img_h

            # Ensure the dimensions don't exceed the response array's size
            xmax = min(xmax, w)
            ymax = min(ymax, h)
            img_w = xmax - xmin
            img_h = ymax - ymin

            # Draw the final image with the adjusted size
            img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
            d = ImageDraw.Draw(img)

            try:
                d.text((0, 0), line_text, font=font, fill=(0, 0, 0))
            except UnicodeEncodeError:
                d.text((0, 0), anyascii(line_text), font=font, fill=(0, 0, 0))

            # Colorize if draw_proba
            if draw_proba:
                avg_confidence = np.mean([word["confidence"] for word in line["words"]])
                p = int(255 * avg_confidence)
                mask = np.where(np.array(img) == 0, 1, 0)
                proba = np.array([255 - p, 0, p])
                color = mask * proba[np.newaxis, np.newaxis, :]
                white_mask = 255 * (1 - mask)
                img = color + white_mask

            # Ensure that the dimensions match and fit within the response
            img_slice = np.array(img)[:(ymax - ymin), :(xmax - xmin), :]

            response[ymin:ymax, xmin:xmax, :] = img_slice

        # TODO: Test and optimize more also rotated text possible ??

    return response
