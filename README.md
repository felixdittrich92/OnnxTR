[![codecov](https://codecov.io/gh/felixdittrich92/OnnxTR/graph/badge.svg?token=WVFRCQBOLI)](https://codecov.io/gh/felixdittrich92/OnnxTR)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/4fff4d764bb14fb8b4f4afeb9587231b)](https://app.codacy.com/gh/felixdittrich92/OnnxTR/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) ![Build Status](https://github.com/felixdittrich92/onnxtr/workflows/builds/badge.svg)
# OnnxTR

## Work in progress

### docTR meets Onnx (doctr wrapper - onnx pipeline)

```python
from onnxtr.io import DocumentFile
from onnxtr.models import ocr_predictor

# PDF
pdf_doc = DocumentFile.from_pdf("path/to/your/doc.pdf")
# Image
single_img_doc = DocumentFile.from_images("path/to/your/img.jpg")
# Webpage (requires `weasyprint` to be installed)
webpage_doc = DocumentFile.from_url("https://www.yoursite.com")
# Multiple page images
multi_img_doc = DocumentFile.from_images(["path/to/page1.jpg", "path/to/page2.jpg"])

model = ocr_predictor()
# PDF
doc = DocumentFile.from_pdf("path/to/your/doc.pdf")
# Analyze
result = model(doc)
```
