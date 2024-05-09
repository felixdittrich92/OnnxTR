[![Codacy Badge](https://api.codacy.com/project/badge/Grade/3111ce48ae7f4d0e8f09108aa638d44d)](https://app.codacy.com/gh/felixdittrich92/OnnxTR?utm_source=github.com&utm_medium=referral&utm_content=felixdittrich92/OnnxTR&utm_campaign=Badge_Grade)
[![codecov](https://codecov.io/gh/felixdittrich92/OnnxTR/graph/badge.svg?token=WVFRCQBOLI)](https://codecov.io/gh/felixdittrich92/OnnxTR)
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
