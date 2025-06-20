import io
import os
from typing import Any

# NOTE: This is a fix to run the demo on the HuggingFace Zero GPU or CPU spaces
if os.environ.get("SPACES_ZERO_GPU") is not None:
    import spaces
else:

    class spaces:  # noqa: N801
        @staticmethod
        def GPU(func):  # noqa: N802
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper


import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from onnxtr.io import DocumentFile
from onnxtr.models import EngineConfig, from_hub, ocr_predictor
from onnxtr.models.predictor import OCRPredictor
from onnxtr.utils.visualization import visualize_page

DET_ARCHS: list[str] = [
    "fast_base",
    "fast_small",
    "fast_tiny",
    "db_resnet50",
    "db_resnet34",
    "db_mobilenet_v3_large",
    "linknet_resnet18",
    "linknet_resnet34",
    "linknet_resnet50",
]
RECO_ARCHS: list[str] = [
    "crnn_vgg16_bn",
    "crnn_mobilenet_v3_small",
    "crnn_mobilenet_v3_large",
    "master",
    "sar_resnet31",
    "vitstr_small",
    "vitstr_base",
    "parseq",
    "viptr_tiny",
]

CUSTOM_RECO_ARCHS: list[str] = [
    "Felix92/onnxtr-parseq-multilingual-v1",
]


def load_predictor(
    det_arch: str,
    reco_arch: str,
    use_gpu: bool,
    assume_straight_pages: bool,
    straighten_pages: bool,
    export_as_straight_boxes: bool,
    detect_language: bool,
    load_in_8_bit: bool,
    bin_thresh: float,
    box_thresh: float,
    disable_crop_orientation: bool = False,
    disable_page_orientation: bool = False,
) -> OCRPredictor:
    """Load a predictor from doctr.models

    Args:
    ----
        det_arch: detection architecture
        reco_arch: recognition architecture
        use_gpu: whether to use the GPU or not
        assume_straight_pages: whether to assume straight pages or not
        disable_crop_orientation: whether to disable crop orientation or not
        disable_page_orientation: whether to disable page orientation or not
        straighten_pages: whether to straighten rotated pages or not
        export_as_straight_boxes: whether to export straight boxes
        detect_language: whether to detect the language of the text
        load_in_8_bit: whether to load the image in 8 bit mode
        bin_thresh: binarization threshold for the segmentation map
        box_thresh: minimal objectness score to consider a box

    Returns:
    -------
        instance of OCRPredictor
    """
    engine_cfg = (
        EngineConfig()
        if use_gpu
        else EngineConfig(providers=[("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})])
    )
    predictor = ocr_predictor(
        det_arch=det_arch,
        reco_arch=reco_arch if reco_arch not in CUSTOM_RECO_ARCHS else from_hub(reco_arch),
        assume_straight_pages=assume_straight_pages,
        straighten_pages=straighten_pages,
        detect_language=detect_language,
        load_in_8_bit=load_in_8_bit,
        export_as_straight_boxes=export_as_straight_boxes,
        detect_orientation=not assume_straight_pages,
        disable_crop_orientation=disable_crop_orientation,
        disable_page_orientation=disable_page_orientation,
        det_engine_cfg=engine_cfg,
        reco_engine_cfg=engine_cfg,
        clf_engine_cfg=engine_cfg,
    )
    predictor.det_predictor.model.postprocessor.bin_thresh = bin_thresh
    predictor.det_predictor.model.postprocessor.box_thresh = box_thresh
    return predictor


def forward_image(predictor: OCRPredictor, image: np.ndarray) -> np.ndarray:
    """Forward an image through the predictor

    Args:
    ----
        predictor: instance of OCRPredictor
        image: image to process

    Returns:
    -------
        segmentation map
    """
    processed_batches = predictor.det_predictor.pre_processor([image])
    out = predictor.det_predictor.model(processed_batches[0], return_model_output=True)
    seg_map = out["out_map"]

    return seg_map


def matplotlib_to_pil(fig: Figure | np.ndarray) -> Image.Image:
    """Convert a matplotlib figure to a PIL image

    Args:
    ----
        fig: matplotlib figure or numpy array

    Returns:
    -------
        PIL image
    """
    buf = io.BytesIO()
    if isinstance(fig, Figure):
        fig.savefig(buf)
    else:
        plt.imsave(buf, fig)
    buf.seek(0)
    return Image.open(buf)


@spaces.GPU
def analyze_page(
    uploaded_file: Any,
    page_idx: int,
    det_arch: str,
    reco_arch: str,
    use_gpu: bool,
    assume_straight_pages: bool,
    disable_crop_orientation: bool,
    disable_page_orientation: bool,
    straighten_pages: bool,
    export_as_straight_boxes: bool,
    detect_language: bool,
    load_in_8_bit: bool,
    bin_thresh: float,
    box_thresh: float,
):
    """Analyze a page

    Args:
    ----
        uploaded_file: file to analyze
        page_idx: index of the page to analyze
        det_arch: detection architecture
        reco_arch: recognition architecture
        use_gpu: whether to use the GPU or not
        assume_straight_pages: whether to assume straight pages or not
        disable_crop_orientation: whether to disable crop orientation or not
        disable_page_orientation: whether to disable page orientation or not
        straighten_pages: whether to straighten rotated pages or not
        export_as_straight_boxes: whether to export straight boxes
        detect_language: whether to detect the language of the text
        load_in_8_bit: whether to load the image in 8 bit mode
        bin_thresh: binarization threshold for the segmentation map
        box_thresh: minimal objectness score to consider a box

    Returns:
    -------
        input image, segmentation heatmap, output image, OCR output, synthesized page
    """
    if uploaded_file is None:
        return None, "Please upload a document", None, None, None

    if uploaded_file.name.endswith(".pdf"):
        doc = DocumentFile.from_pdf(uploaded_file)
    else:
        doc = DocumentFile.from_images(uploaded_file)
    try:
        page = doc[page_idx - 1]
    except IndexError:
        page = doc[-1]

    img = page

    predictor = load_predictor(
        det_arch=det_arch,
        reco_arch=reco_arch,
        use_gpu=use_gpu,
        assume_straight_pages=assume_straight_pages,
        straighten_pages=straighten_pages,
        export_as_straight_boxes=export_as_straight_boxes,
        detect_language=detect_language,
        load_in_8_bit=load_in_8_bit,
        bin_thresh=bin_thresh,
        box_thresh=box_thresh,
        disable_crop_orientation=disable_crop_orientation,
        disable_page_orientation=disable_page_orientation,
    )

    seg_map = forward_image(predictor, page)
    seg_map = np.squeeze(seg_map)
    seg_map = cv2.resize(seg_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    seg_heatmap = matplotlib_to_pil(seg_map)

    out = predictor([page])

    page_export = out.pages[0].export()
    fig = visualize_page(out.pages[0].export(), out.pages[0].page, interactive=False, add_labels=False)

    out_img = matplotlib_to_pil(fig)

    if assume_straight_pages or (not assume_straight_pages and straighten_pages):
        synthesized_page = out.pages[0].synthesize()
    else:
        synthesized_page = None

    return img, seg_heatmap, out_img, page_export, synthesized_page


with gr.Blocks(fill_height=True) as demo:
    gr.HTML(
        """
        <div style="text-align: center;">
            <p style="display: flex; justify-content: center;">
                <img src="https://github.com/felixdittrich92/OnnxTR/raw/main/docs/images/logo.jpg" width="15%">
            </p>

            <h1>OnnxTR OCR Demo</h1>

            <p style="display: flex; justify-content: center; gap: 10px;">
                <a href="https://github.com/felixdittrich92/OnnxTR" target="_blank">
                    <img src="https://img.shields.io/badge/GitHub-blue?logo=github" alt="GitHub OnnxTR">
                </a>
                <a href="https://pypi.org/project/onnxtr/" target="_blank">
                    <img src="https://img.shields.io/pypi/v/onnxtr?color=blue" alt="PyPI">
                </a>
            </p>
        </div>
        <h2>To use this interactive demo for OnnxTR:</h2>
        <h3> 1. Upload a document (PDF, JPG, or PNG)</h3>
        <h3> 2. Select the model architectures for text detection and recognition you want to use</h3>
        <h3> 3. Press the "Analyze page" button to process the uploaded document</h3>
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            upload = gr.File(label="Upload File [JPG | PNG | PDF]", file_types=[".pdf", ".jpg", ".png"])
            page_selection = gr.Slider(minimum=1, maximum=10, step=1, value=1, label="Page selection")
            det_model = gr.Dropdown(choices=DET_ARCHS, value=DET_ARCHS[0], label="Text detection model")
            reco_model = gr.Dropdown(
                choices=RECO_ARCHS + CUSTOM_RECO_ARCHS, value=RECO_ARCHS[0], label="Text recognition model"
            )
            use_gpu = gr.Checkbox(value=True, label="Use GPU")
            assume_straight = gr.Checkbox(value=True, label="Assume straight pages")
            disable_crop_orientation = gr.Checkbox(value=False, label="Disable crop orientation")
            disable_page_orientation = gr.Checkbox(value=False, label="Disable page orientation")
            straighten = gr.Checkbox(value=False, label="Straighten pages")
            export_as_straight_boxes = gr.Checkbox(value=False, label="Export as straight boxes")
            det_language = gr.Checkbox(value=False, label="Detect language")
            load_in_8_bit = gr.Checkbox(value=False, label="Load 8-bit quantized models")
            binarization_threshold = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.3, step=0.1, label="Binarization threshold"
            )
            box_threshold = gr.Slider(minimum=0.1, maximum=0.9, value=0.1, step=0.1, label="Box threshold")
            analyze_button = gr.Button("Analyze page")
        with gr.Column(scale=3):
            with gr.Row():
                input_image = gr.Image(label="Input page", width=700, height=500)
                segmentation_heatmap = gr.Image(label="Segmentation heatmap", width=700, height=500)
                output_image = gr.Image(label="Output page", width=700, height=500)
            with gr.Row():
                with gr.Column(scale=3):
                    ocr_output = gr.JSON(label="OCR output", render=True, scale=1, height=500)
                with gr.Column(scale=3):
                    synthesized_page = gr.Image(label="Synthesized page", width=700, height=500)

    analyze_button.click(
        analyze_page,
        inputs=[
            upload,
            page_selection,
            det_model,
            reco_model,
            use_gpu,
            assume_straight,
            disable_crop_orientation,
            disable_page_orientation,
            straighten,
            export_as_straight_boxes,
            det_language,
            load_in_8_bit,
            binarization_threshold,
            box_threshold,
        ],
        outputs=[input_image, segmentation_heatmap, output_image, ocr_output, synthesized_page],
    )

demo.launch(inbrowser=True, allowed_paths=["./data/logo.jpg"])
