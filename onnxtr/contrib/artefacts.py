# Copyright (C) 2021-2024, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from onnxtr.file_utils import requires_package

from .base import _BasePredictor

__all__ = ["ArtefactDetector"]

default_cfgs: Dict[str, Dict[str, Any]] = {
    "yolov8_artefact": {
        "input_shape": (3, 1024, 1024),
        "labels": ["bar_code", "qr_code", "logo", "photo"],
        "url": "https://github.com/felixdittrich92/OnnxTR/releases/download/v0.0.1/yolo_artefact-f9d66f14.onnx",
    },
}


class ArtefactDetector(_BasePredictor):
    """
    A class to detect artefacts in images

    >>> from onnxtr.io import DocumentFile
    >>> from onnxtr.contrib.artefacts import ArtefactDetector
    >>> doc = DocumentFile.from_images(["path/to/image.jpg"])
    >>> detector = ArtefactDetector()
    >>> results = detector(doc)

    Args:
    ----
        arch: the architecture to use
        batch_size: the batch size to use
        model_path: the path to the model to use
        labels: the labels to use
        input_shape: the input shape to use
        mask_labels: the mask labels to use
        conf_threshold: the confidence threshold to use
        iou_threshold: the intersection over union threshold to use
        **kwargs: additional arguments to be passed to `download_from_url`
    """

    def __init__(
        self,
        arch: str = "yolov8_artefact",
        batch_size: int = 2,
        model_path: Optional[str] = None,
        labels: Optional[List[str]] = None,
        input_shape: Optional[Tuple[int, int, int]] = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(batch_size=batch_size, url=default_cfgs[arch]["url"], model_path=model_path, **kwargs)
        self.labels = labels or default_cfgs[arch]["labels"]
        self.input_shape = input_shape or default_cfgs[arch]["input_shape"]
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        return np.transpose(cv2.resize(img, (self.input_shape[2], self.input_shape[1])), (2, 0, 1)) / np.array(255.0)

    def postprocess(self, output: List[np.ndarray], input_images: List[List[np.ndarray]]) -> List[List[Dict[str, Any]]]:
        results = []

        for batch in zip(output, input_images):
            for out, img in zip(batch[0], batch[1]):
                org_height, org_width = img.shape[:2]
                width_scale, height_scale = org_width / self.input_shape[2], org_height / self.input_shape[1]
                for res in out:
                    sample_results = []
                    for row in np.transpose(np.squeeze(res)):
                        classes_scores = row[4:]
                        max_score = np.amax(classes_scores)
                        if max_score >= self.conf_threshold:
                            class_id = np.argmax(classes_scores)
                            x, y, w, h = row[0], row[1], row[2], row[3]
                            # to rescaled xmin, ymin, xmax, ymax
                            xmin = int((x - w / 2) * width_scale)
                            ymin = int((y - h / 2) * height_scale)
                            xmax = int((x + w / 2) * width_scale)
                            ymax = int((y + h / 2) * height_scale)

                            sample_results.append({
                                "label": self.labels[class_id],
                                "confidence": float(max_score),
                                "box": [xmin, ymin, xmax, ymax],
                            })

                    # Filter out overlapping boxes
                    boxes = [res["box"] for res in sample_results]
                    scores = [res["confidence"] for res in sample_results]
                    keep_indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)  # type: ignore[arg-type]
                    sample_results = [sample_results[i] for i in keep_indices]

                    results.append(sample_results)

        self._results = results
        return results

    def show(self, **kwargs: Any) -> None:
        """
        Display the results

        Args:
        ----
            **kwargs: additional keyword arguments to be passed to `plt.show`
        """
        requires_package("matplotlib", "`.show()` requires matplotlib installed")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        # visualize the results with matplotlib
        if self._results and self._inputs:
            for img, res in zip(self._inputs, self._results):
                plt.figure(figsize=(10, 10))
                plt.imshow(img)
                for obj in res:
                    xmin, ymin, xmax, ymax = obj["box"]
                    label = obj["label"]
                    plt.text(xmin, ymin, f"{label} {obj['confidence']:.2f}", color="red")
                    plt.gca().add_patch(
                        Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor="red", linewidth=2)
                    )
                plt.show(**kwargs)
