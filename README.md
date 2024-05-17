<p align="center">
  <img src="https://github.com/felixdittrich92/OnnxTR/raw/main/docs/images/logo.jpg" width="40%">
</p>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Build Status](https://github.com/felixdittrich92/onnxtr/workflows/builds/badge.svg)
[![codecov](https://codecov.io/gh/felixdittrich92/OnnxTR/graph/badge.svg?token=WVFRCQBOLI)](https://codecov.io/gh/felixdittrich92/OnnxTR)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/4fff4d764bb14fb8b4f4afeb9587231b)](https://app.codacy.com/gh/felixdittrich92/OnnxTR/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![CodeFactor](https://www.codefactor.io/repository/github/felixdittrich92/onnxtr/badge)](https://www.codefactor.io/repository/github/felixdittrich92/onnxtr)
[![Pypi](https://img.shields.io/badge/pypi-v0.2.0-blue.svg)](https://pypi.org/project/OnnxTR/)

> :warning: Please note that this is a wrapper around the [doctr](https://github.com/mindee/doctr) library to provide a Onnx pipeline for docTR. For feature requests, which are not directly related to the Onnx pipeline, please refer to the base project.

**Optical Character Recognition made seamless & accessible to anyone, powered by Onnx**

What you can expect from this repository:

- efficient ways to parse textual information (localize and identify each word) from your documents
- a Onnx pipeline for docTR, a wrapper around the [doctr](https://github.com/mindee/doctr) library - no PyTorch or TensorFlow dependencies
- more lightweight package with faster inference latency and less required resources
- 8-Bit quantized models for faster inference on CPU

![OCR_example](https://github.com/felixdittrich92/OnnxTR/raw/main/docs/images/ocr.png)

## Installation

### Prerequisites

Python 3.9 (or higher) and [pip](https://pip.pypa.io/en/stable/) are required to install OnnxTR.

### Latest release

You can then install the latest release of the package using [pypi](https://pypi.org/project/OnnxTR/) as follows:

**NOTE:**

For GPU support please take a look at: [ONNX Runtime](https://onnxruntime.ai/getting-started). Currently supported execution providers by default are: CPU, CUDA

- **Prerequisites:** CUDA & cuDNN needs to be installed before [Version table](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

```shell
pip install "onnxtr[cpu]"
# with gpu support
pip install "onnxtr[gpu]"
# with HTML support
pip install "onnxtr[html]"
# with support for visualization
pip install "onnxtr[viz]"
# with support for all dependencies
pip install "onnxtr[html, gpu, viz]"
```

### Reading files

Documents can be interpreted from PDF / Images / Webpages / Multiple page images using the following code snippet:

```python
from onnxtr.io import DocumentFile
# PDF
pdf_doc = DocumentFile.from_pdf("path/to/your/doc.pdf")
# Image
single_img_doc = DocumentFile.from_images("path/to/your/img.jpg")
# Webpage (requires `weasyprint` to be installed)
webpage_doc = DocumentFile.from_url("https://www.yoursite.com")
# Multiple page images
multi_img_doc = DocumentFile.from_images(["path/to/page1.jpg", "path/to/page2.jpg"])
```

### Putting it together

Let's use the default pretrained model for an example:

```python
from onnxtr.io import DocumentFile
from onnxtr.models import ocr_predictor

model = ocr_predictor(
    det_arch='fast_base',  # detection architecture
    reco_arch='vitstr_base',  # recognition architecture
    det_bs=4, # detection batch size
    reco_bs=1024, # recognition batch size
    assume_straight_pages=True,  # set to `False` if the pages are not straight (rotation, perspective, etc.) (default: True)
    straighten_pages=False,  # set to `True` if the pages should be straightened before final processing (default: False)
    # Preprocessing related parameters
    preserve_aspect_ratio=True,  # set to `False` if the aspect ratio should not be preserved (default: True)
    symmetric_pad=True,  # set to `False` to disable symmetric padding (default: True)
    # Additional parameters - meta information
    detect_orientation=False,  # set to `True` if the orientation of the pages should be detected (default: False)
    detect_language=False, # set to `True` if the language of the pages should be detected (default: False)
    # DocumentBuilder specific parameters
    resolve_lines=True,  # whether words should be automatically grouped into lines (default: True)
    resolve_blocks=True,  # whether lines should be automatically grouped into blocks (default: True)
    paragraph_break=0.035,  # relative length of the minimum space separating paragraphs (default: 0.035)
    # OnnxTR specific parameters
    # NOTE: 8-Bit quantized models are not available for FAST detection models and can in general lead to poorer accuracy
    load_in_8_bit=False,  # set to `True` to load 8-bit quantized models instead of the full precision onces (default: False)
)
# PDF
doc = DocumentFile.from_pdf("path/to/your/doc.pdf")
# Analyze
result = model(doc)
# Display the result (requires matplotlib & mplcursors to be installed)
result.show()
```

![Visualization sample](https://github.com/felixdittrich92/OnnxTR/raw/main/docs/images/doctr_example_script.gif)

Or even rebuild the original document from its predictions:

```python
import matplotlib.pyplot as plt

synthetic_pages = result.synthesize()
plt.imshow(synthetic_pages[0]); plt.axis('off'); plt.show()
```

![Synthesis sample](https://github.com/felixdittrich92/OnnxTR/raw/main/docs/images/synthesized_sample.png)

The `ocr_predictor` returns a `Document` object with a nested structure (with `Page`, `Block`, `Line`, `Word`, `Artefact`).
To get a better understanding of the document model, check out [documentation](https://mindee.github.io/doctr/modules/io.html#document-structure):

You can also export them as a nested dict, more appropriate for JSON format / render it or export as XML (hocr format):

```python
json_output = result.export()  # nested dict
text_output = result.render()  # human-readable text
xml_output = result.export_as_xml()  # hocr format
for output in xml_output:
    xml_bytes_string = output[0]
    xml_element = output[1]

```

## Loading custom exported models

You can also load docTR custom exported models:
For exporting please take a look at the [doctr documentation](https://mindee.github.io/doctr/using_doctr/using_model_export.html#export-to-onnx).

```python
from onnxtr.models import ocr_predictor, linknet_resnet18, parseq

reco_model = parseq("path_to_custom_model.onnx", vocab="ABC")
det_model = linknet_resnet18("path_to_custom_model.onnx")
model = ocr_predictor(det_arch=det_model, reco_arch=reco_model)
```

## Models architectures

Credits where it's due: this repository is implementing, among others, architectures from published research papers.

### Text Detection

- DBNet: [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/pdf/1911.08947.pdf).
- LinkNet: [LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation](https://arxiv.org/pdf/1707.03718.pdf)
- FAST: [FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation](https://arxiv.org/pdf/2111.02394.pdf)

### Text Recognition

- CRNN: [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/pdf/1507.05717.pdf).
- SAR: [Show, Attend and Read:A Simple and Strong Baseline for Irregular Text Recognition](https://arxiv.org/pdf/1811.00751.pdf).
- MASTER: [MASTER: Multi-Aspect Non-local Network for Scene Text Recognition](https://arxiv.org/pdf/1910.02562.pdf).
- ViTSTR: [Vision Transformer for Fast and Efficient Scene Text Recognition](https://arxiv.org/pdf/2105.08582.pdf).
- PARSeq: [Scene Text Recognition with Permuted Autoregressive Sequence Models](https://arxiv.org/pdf/2207.06966).

```python
predictor = ocr_predictor()
predictor.list_archs()
{
    'detection archs':
        [
            'db_resnet34',
            'db_resnet50',
            'db_mobilenet_v3_large',
            'linknet_resnet18',
            'linknet_resnet34',
            'linknet_resnet50',
            'fast_tiny',  # No 8-bit support
            'fast_small',  # No 8-bit support
            'fast_base'  # No 8-bit support
        ],
    'recognition archs':
        [
            'crnn_vgg16_bn',
            'crnn_mobilenet_v3_small',
            'crnn_mobilenet_v3_large',
            'sar_resnet31',
            'master',
            'vitstr_small',
            'vitstr_base',
            'parseq'
        ]
}
```

### Documentation

This repository is in sync with the [doctr](https://github.com/mindee/doctr) library, which provides a high-level API to perform OCR on documents.
This repository stays up-to-date with the latest features and improvements from the base project.
So we can refer to the [doctr documentation](https://mindee.github.io/doctr/) for more detailed information.

NOTE:

- `pretrained` is the default in OnnxTR, and not available as a parameter.
- docTR specific environment variables (e.g.: DOCTR_CACHE_DIR -> ONNXTR_CACHE_DIR) needs to be replaced with `ONNXTR_` prefix.

### Benchmarks

The CPU benchmarks was measured on a `i7-14700K Intel CPU`.

The GPU benchmarks was measured on a `RTX 4080 Nvidia GPU`.

Benchmarking performed on the FUNSD dataset and CORD dataset.

docTR / OnnxTR models used for the benchmarks are `fast_base` (full precision) | `db_resnet50` (8-bit variant) for detection and `crnn_vgg16_bn` for recognition.

The smallest combination in OnnxTR (docTR) of `db_mobilenet_v3_large` and `crnn_mobilenet_v3_small` takes as comparison `~0.17s / Page` on the FUNSD dataset and `~0.12s / Page` on the CORD dataset in **full precision**.

- CPU benchmarks:

|Library                          |FUNSD (199 pages)              |CORD  (900 pages)              |
|---------------------------------|-------------------------------|-------------------------------|
|docTR (CPU) - v0.8.1             | ~1.29s / Page                 | ~0.60s / Page                 |
|**OnnxTR (CPU)** - v0.1.2        | ~0.57s / Page                 | **~0.25s / Page**             |
|**OnnxTR (CPU) 8-bit** - v0.1.2  | **~0.38s / Page**             | **~0.14s / Page**             |
|EasyOCR (CPU) - v1.7.1           | ~1.96s / Page                 | ~1.75s / Page                 |
|**PyTesseract (CPU)** - v0.3.10  | **~0.50s / Page**             | ~0.52s / Page                 |
|Surya (line) (CPU) - v0.4.4      | ~48.76s / Page                | ~35.49s / Page                |
|PaddleOCR (CPU) - no cls - v2.7.3| ~1.27s / Page                 | ~0.38s / Page                 |

- GPU benchmarks:

|Library                              |FUNSD (199 pages)              |CORD  (900 pages)              |
|-------------------------------------|-------------------------------|-------------------------------|
|docTR (GPU) - v0.8.1                 | ~0.07s / Page                 | ~0.05s / Page                 |
|**docTR (GPU) float16** - v0.8.1     | **~0.06s / Page**             | **~0.03s / Page**             |
|OnnxTR (GPU) - v0.1.2                | **~0.06s / Page**             | ~0.04s / Page                 |
|EasyOCR (GPU) - v1.7.1               | ~0.31s / Page                 | ~0.19s / Page                 |
|Surya (GPU) float16 - v0.4.4         | ~3.70s / Page                 | ~2.81s / Page                 |
|**PaddleOCR (GPU) - no cls - v2.7.3**| ~0.08s / Page                 | **~0.03s / Page**             |

## Citation

If you wish to cite please refer to the base project citation, feel free to use this [BibTeX](http://www.bibtex.org/) reference:

```bibtex
@misc{doctr2021,
    title={docTR: Document Text Recognition},
    author={Mindee},
    year={2021},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/mindee/doctr}}
}
```

## License

Distributed under the Apache 2.0 License. See [`LICENSE`](https://github.com/felixdittrich92/OnnxTR?tab=Apache-2.0-1-ov-file#readme) for more information.
