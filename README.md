<p align="center">
  <img src="https://github.com/felixdittrich92/OnnxTR/raw/main/docs/images/logo.jpg" width="40%">
</p>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Build Status](https://github.com/felixdittrich92/onnxtr/workflows/builds/badge.svg)
[![codecov](https://codecov.io/gh/felixdittrich92/OnnxTR/graph/badge.svg?token=WVFRCQBOLI)](https://codecov.io/gh/felixdittrich92/OnnxTR)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/4fff4d764bb14fb8b4f4afeb9587231b)](https://app.codacy.com/gh/felixdittrich92/OnnxTR/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![CodeFactor](https://www.codefactor.io/repository/github/felixdittrich92/onnxtr/badge)](https://www.codefactor.io/repository/github/felixdittrich92/onnxtr)
[![Socket Badge](https://socket.dev/api/badge/pypi/package/onnxtr/0.8.0?artifact_id=tar-gz)](https://socket.dev/pypi/package/onnxtr/overview/0.8.0/tar-gz)
[![Pypi](https://img.shields.io/badge/pypi-v0.8.0-blue.svg)](https://pypi.org/project/OnnxTR/)
[![Docker Images](https://img.shields.io/badge/Docker-4287f5?style=flat&logo=docker&logoColor=white)](https://github.com/felixdittrich92/OnnxTR/pkgs/container/onnxtr)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Felix92/OnnxTR-OCR)
![PyPI - Downloads](https://img.shields.io/pypi/dm/onnxtr)

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

Python 3.10 (or higher) and [pip](https://pip.pypa.io/en/stable/) are required to install OnnxTR.

### Latest release

You can then install the latest release of the package using [pypi](https://pypi.org/project/OnnxTR/) as follows:

**NOTE:**

Currently supported execution providers by default are: CPU, CUDA (NVIDIA GPU), OpenVINO (Intel CPU | GPU).

For GPU support please take a look at: [ONNX Runtime](https://onnxruntime.ai/getting-started).

- **Prerequisites:** CUDA & cuDNN needs to be installed before [Version table](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

```shell
# standard cpu support
pip install "onnxtr[cpu]"
pip install "onnxtr[cpu-headless]"  # same as cpu but with opencv-headless
# with gpu support
pip install "onnxtr[gpu]"
pip install "onnxtr[gpu-headless]"  # same as gpu but with opencv-headless
# OpenVINO cpu | gpu support for Intel CPUs | GPUs
pip install "onnxtr[openvino]"
pip install "onnxtr[openvino-headless]"  # same as openvino but with opencv-headless
# with HTML support
pip install "onnxtr[html]"
# with support for visualization
pip install "onnxtr[viz]"
# with support for all dependencies
pip install "onnxtr[html, gpu, viz]"
```

**Recommendation:**

If you have:

- a NVIDIA GPU, use one of the `gpu` variants
- an Intel CPU or GPU, use one of the `openvino` variants
- otherwise, use one of the `cpu` variants

**OpenVINO:**

By default OnnxTR running with the OpenVINO execution provider backend uses the `CPU` device with `FP32` precision, to change the device or for further configuaration please refer to the [ONNX Runtime OpenVINO documentation](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#summary-of-options).

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

Let's use the default `ocr_predictor` model for an example:

```python
from onnxtr.io import DocumentFile
from onnxtr.models import ocr_predictor, EngineConfig

model = ocr_predictor(
    det_arch='fast_base',  # detection architecture
    reco_arch='vitstr_base',  # recognition architecture
    det_bs=2, # detection batch size
    reco_bs=512, # recognition batch size
    # Document related parameters
    assume_straight_pages=True,  # set to `False` if the pages are not straight (rotation, perspective, etc.) (default: True)
    straighten_pages=False,  # set to `True` if the pages should be straightened before final processing (default: False)
    export_as_straight_boxes=False,  # set to `True` if the boxes should be exported as if the pages were straight (default: False)
    # Preprocessing related parameters
    preserve_aspect_ratio=True,  # set to `False` if the aspect ratio should not be preserved (default: True)
    symmetric_pad=True,  # set to `False` to disable symmetric padding (default: True)
    # Additional parameters - meta information
    detect_orientation=False,  # set to `True` if the orientation of the pages should be detected (default: False)
    detect_language=False, # set to `True` if the language of the pages should be detected (default: False)
    # Orientation specific parameters in combination with `assume_straight_pages=False` and/or `straighten_pages=True`
    disable_crop_orientation=False,  # set to `True` if the crop orientation classification should be disabled (default: False)
    disable_page_orientation=False,  # set to `True` if the general page orientation classification should be disabled (default: False)
    # DocumentBuilder specific parameters
    resolve_lines=True,  # whether words should be automatically grouped into lines (default: True)
    resolve_blocks=False,  # whether lines should be automatically grouped into blocks (default: False)
    paragraph_break=0.035,  # relative length of the minimum space separating paragraphs (default: 0.035)
    # OnnxTR specific parameters
    # NOTE: 8-Bit quantized models are not available for FAST detection models and can in general lead to poorer accuracy
    load_in_8_bit=False,  # set to `True` to load 8-bit quantized models instead of the full precision onces (default: False)
    # Advanced engine configuration options
    det_engine_cfg=EngineConfig(),  # detection model engine configuration (default: internal predefined configuration)
    reco_engine_cfg=EngineConfig(),  # recognition model engine configuration (default: internal predefined configuration)
    clf_engine_cfg=EngineConfig(),  # classification (orientation) model engine configuration (default: internal predefined configuration)
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

<details>
  <summary>Advanced engine configuration options</summary>

You can also define advanced engine configurations for the models / predictors:

```python
from onnxruntime import SessionOptions

from onnxtr.models import ocr_predictor, EngineConfig

general_options = SessionOptions()  # For configuartion options see: https://onnxruntime.ai/docs/api/python/api_summary.html#sessionoptions
general_options.enable_cpu_mem_arena = False

# NOTE: The following would force to run only on the GPU if no GPU is available it will raise an error
# List of strings e.g. ["CUDAExecutionProvider", "CPUExecutionProvider"] or a list of tuples with the provider and its options e.g.
# [("CUDAExecutionProvider", {"device_id": 0}), ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]
providers = [("CUDAExecutionProvider", {"device_id": 0, "cudnn_conv_algo_search": "DEFAULT"})]  # For available providers see: https://onnxruntime.ai/docs/execution-providers/

engine_config = EngineConfig(
    session_options=general_options,
    providers=providers
)
# We use the default predictor with the custom engine configuration
# NOTE: You can define differnt engine configurations for detection, recognition and classification depending on your needs
predictor = ocr_predictor(
    det_engine_cfg=engine_config,
    reco_engine_cfg=engine_config,
    clf_engine_cfg=engine_config
)
```

You can also dynamically configure whether the memory arena should shrink:

```python
from random import random
from onnxruntime import RunOptions, SessionOptions

from onnxtr.models import ocr_predictor, EngineConfig

def arena_shrinkage_handler(run_options: RunOptions) -> RunOptions:
  """
  Shrink the memory arena on 10% of inference runs.
  """
  if random() < 0.1:
    run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0")
  return run_options

engine_config = EngineConfig(run_options_provider=arena_shrinkage_handler)
engine_config.session_options.enable_mem_pattern = False

predictor = ocr_predictor(
    det_engine_cfg=engine_config,
    reco_engine_cfg=engine_config,
    clf_engine_cfg=engine_config
)
```

</details>

## Loading custom exported models

You can also load docTR custom exported models:
For exporting please take a look at the [doctr documentation](https://mindee.github.io/doctr/using_doctr/using_model_export.html#export-to-onnx).

```python
from onnxtr.models import ocr_predictor, linknet_resnet18, parseq

reco_model = parseq("path_to_custom_model.onnx", vocab="ABC")
det_model = linknet_resnet18("path_to_custom_model.onnx")
model = ocr_predictor(det_arch=det_model, reco_arch=reco_model)
```

## Loading models from HuggingFace Hub

You can also load models from the HuggingFace Hub:

```python
from onnxtr.io import DocumentFile
from onnxtr.models import ocr_predictor, from_hub

img = DocumentFile.from_images(['<image_path>'])
# Load your model from the hub
model = from_hub('onnxtr/my-model')

# Pass it to the predictor
# If your model is a recognition model:
predictor = ocr_predictor(
    det_arch='db_mobilenet_v3_large',
    reco_arch=model
)

# If your model is a detection model:
predictor = ocr_predictor(
    det_arch=model,
    reco_arch='crnn_mobilenet_v3_small'
)

# Get your predictions
res = predictor(img)
```

HF Hub search: [here](https://huggingface.co/models?search=onnxtr).

Collection: [here](https://huggingface.co/collections/Felix92/onnxtr-66bf213a9f88f7346c90e842)

Or push your own models to the hub:

```python
from onnxtr.models import parseq, push_to_hf_hub, login_to_hub
from onnxtr.utils.vocabs import VOCABS

# Login to the hub
login_to_hub()

# Recogniton model
model = parseq("~/onnxtr-parseq-multilingual-v1.onnx", vocab=VOCABS["multilingual"])
push_to_hf_hub(
    model,
    model_name="onnxtr-parseq-multilingual-v1",
    task="recognition",  # The task for which the model is intended [detection, recognition, classification]
    arch="parseq",  # The name of the model architecture
    override=False  # Set to `True` if you want to override an existing model / repository
)

# Detection model
model = linknet_resnet18("~/onnxtr-linknet-resnet18.onnx")
push_to_hf_hub(
    model,
    model_name="onnxtr-linknet-resnet18",
    task="detection",
    arch="linknet_resnet18",
    override=True
)
```

## Models architectures

Credits where it's due: this repository provides ONNX models for the following architectures, converted from the docTR models:

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
- VIPTR: [A Vision Permutable Extractor for Fast and Efficient Scene Text Recognition](https://arxiv.org/abs/2401.10110).

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
            'viptr_tiny',  # No 8-bit support
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

The smallest combination in OnnxTR (docTR) of `db_mobilenet_v3_large` and `crnn_mobilenet_v3_small` takes as comparison `~0.17s / Page` on the FUNSD dataset and `~0.12s / Page` on the CORD dataset in **full precision** on CPU.

- CPU benchmarks:

|Library                             |FUNSD (199 pages)              |CORD  (900 pages)              |
|------------------------------------|-------------------------------|-------------------------------|
|docTR (CPU) - v0.8.1                | ~1.29s / Page                 | ~0.60s / Page                 |
|**OnnxTR (CPU)** - v0.6.0           | ~0.57s / Page                 | **~0.25s / Page**             |
|**OnnxTR (CPU) 8-bit** - v0.6.0     | **~0.38s / Page**             | **~0.14s / Page**             |
|**OnnxTR (CPU-OpenVINO)** - v0.6.0  | **~0.15s / Page**             | **~0.14s / Page**             |
|EasyOCR (CPU) - v1.7.1              | ~1.96s / Page                 | ~1.75s / Page                 |
|**PyTesseract (CPU)** - v0.3.10     | **~0.50s / Page**             | ~0.52s / Page                 |
|Surya (line) (CPU) - v0.4.4         | ~48.76s / Page                | ~35.49s / Page                |
|PaddleOCR (CPU) - no cls - v2.7.3   | ~1.27s / Page                 | ~0.38s / Page                 |

- GPU benchmarks:

|Library                              |FUNSD (199 pages)              |CORD  (900 pages)              |
|-------------------------------------|-------------------------------|-------------------------------|
|docTR (GPU) - v0.8.1                 | ~0.07s / Page                 | ~0.05s / Page                 |
|**docTR (GPU) float16** - v0.8.1     | **~0.06s / Page**             | **~0.03s / Page**             |
|OnnxTR (GPU) - v0.6.0                | **~0.06s / Page**             | ~0.04s / Page                 |
|**OnnxTR (GPU) float16 - v0.6.0**    | **~0.05s / Page**             | **~0.03s / Page**             |
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

```bibtex
@misc{onnxtr2024,
    title={OnnxTR: Optical Character Recognition made seamless & accessible to anyone, powered by Onnx},
    author={Felix Dittrich},
    year={2024},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/felixdittrich92/OnnxTR}}
}
```

## License

Distributed under the Apache 2.0 License. See [`LICENSE`](https://github.com/felixdittrich92/OnnxTR?tab=Apache-2.0-1-ov-file#readme) for more information.
