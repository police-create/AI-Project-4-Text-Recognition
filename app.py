"""Gradio Web Application for AI Project 4 - Building the Machine's Optic Nerve.

Provides two distinct operating modes:
1. 👁️ Text Recognition (OCR): OpenCV Preprocessing + Tesseract OCR Engine
2. 🎯 Object Detection: OpenCV DNN + MobileNet-SSD Deep Learning Architecture
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
from PIL import Image

from src.config import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_PSM_MODE,
    EXPECTED_OUTPUT_PATH,
    SAMPLE_INPUT_PATH,
    SUPPORTED_PSM_MODES,
)
from src.evaluator import evaluate_object_detections, evaluate_ocr
from src.object_detector import ObjectDetector, ModelNotFoundError
from src.ocr_engine import OCREngine, TesseractNotFoundError
from src.pipeline import OCRPipeline

logger = logging.getLogger("WebUI")

SAMPLE_OBJECTS_PATH = Path("data/sample_objects.jpg")
GROUND_TRUTH_OBJECTS_PATH = Path("data/ground_truth.json")

# Initialize OCR Pipeline
try:
    _OCR_ENGINE = OCREngine()
    OCR_PIPELINE = OCRPipeline(ocr_engine=_OCR_ENGINE)
    OCR_STATUS_MSG = f"Active (v{_OCR_ENGINE.version})"
except Exception as e:
    OCR_PIPELINE = None  # type: ignore
    OCR_STATUS_MSG = f"Unavailable ({e})"

# Initialize Object Detector
try:
    OBJECT_DETECTOR = ObjectDetector()
    DETECTOR_STATUS_MSG = "Active (MobileNet-SSD PASCAL VOC)"
except Exception as e:
    OBJECT_DETECTOR = None  # type: ignore
    DETECTOR_STATUS_MSG = f"Unavailable ({e})"


# ============================================================================
# OCR TAB HANDLERS
# ============================================================================


def analyze_ocr_fn(
    image: Optional[np.ndarray | Image.Image],
    psm_mode: str,
    threshold: float,
    ground_truth_text: Optional[str] = None,
):
    """Callback for OCR execution."""
    if OCR_PIPELINE is None:
        return (
            None, None, None, None,
            "Tesseract OCR engine is not available on this machine.",
            "N/A", "N/A", "FAIL",
        )

    if image is None:
        return (
            None, None, None, None,
            "Please upload an image or click 'Run Sample Validation'.",
            "Waiting for image", "N/A", "READY",
        )

    try:
        res = OCR_PIPELINE.process(
            image_input=image,
            expected_text_path=None,
            save_artifacts=True,
            psm_mode=psm_mode,
            confidence_threshold=threshold,
        )

        prep = res.preprocessing
        ocr = res.ocr

        extracted_text = (
            f"=== RAW EXTRACTED TEXT ===\n{ocr.raw_text.strip() or '[No text detected]'}\n\n"
            f"=== FILTERED TEXT (>= {threshold:.0f}%) ===\n{ocr.filtered_text or '[No tokens passed confidence threshold]'}"
        )

        confidence_info = (
            f"Mean OCR Confidence : {ocr.mean_confidence:.2f}%\n"
            f"Acceptance Threshold: {threshold:.0f}%\n"
            f"Accepted Tokens     : {ocr.accepted_token_count} / {len(ocr.tokens)}\n"
            f"Rejected Tokens     : {ocr.rejected_token_count}\n"
            f"Acceptance Rate     : {ocr.acceptance_rate:.2f}%\n"
            f"Deskew Angle        : {prep.deskew_angle:.2f}°"
        )

        accuracy_info = "Provide ground-truth text to calculate accuracy."
        benchmark_passed = ocr.mean_confidence >= threshold

        if ground_truth_text and ground_truth_text.strip():
            eval_rep = evaluate_ocr(
                recognized_text=ocr.filtered_text or ocr.raw_text,
                ground_truth=ground_truth_text,
                ocr_confidence=ocr.mean_confidence,
                threshold=threshold,
            )
            accuracy_info = (
                f"Character Accuracy : {eval_rep.character_accuracy:.2f}%\n"
                f"Word Accuracy      : {eval_rep.word_accuracy:.2f}%\n"
                f"Char Error Rate    : {eval_rep.cer:.4f}\n"
                f"Word Error Rate    : {eval_rep.wer:.4f}\n"
                f"Overall Accuracy   : {eval_rep.overall_accuracy:.2f}%\n"
                f"Benchmark Status   : {eval_rep.status}"
            )
            benchmark_passed = eval_rep.passed

        final_status = "PASS" if benchmark_passed else "FAIL"

        return (
            prep.grayscale,
            prep.blurred,
            prep.deskewed,
            prep.thresholded,
            extracted_text,
            confidence_info,
            accuracy_info,
            final_status,
        )
    except Exception as exc:
        logger.error("OCR execution error: %s", exc, exc_info=True)
        return (
            None, None, None, None,
            f"Error during OCR execution: {exc}",
            "Error", "Error", "FAIL",
        )


def run_sample_ocr_fn(psm_mode: str, threshold: float):
    """Load controlled sample text image and execute full OCR validation benchmark."""
    if not SAMPLE_INPUT_PATH.is_file():
        return (
            None, None, None, None, None,
            f"Sample image missing at {SAMPLE_INPUT_PATH}",
            "Missing", "Missing", "FAIL",
        )

    gt_text = ""
    if EXPECTED_OUTPUT_PATH.is_file():
        gt_text = EXPECTED_OUTPUT_PATH.read_text(encoding="utf-8")

    sample_img = Image.open(SAMPLE_INPUT_PATH)
    gray, blur, deskew, thresh, text, conf, acc, status = analyze_ocr_fn(
        image=sample_img,
        psm_mode=psm_mode,
        threshold=threshold,
        ground_truth_text=gt_text,
    )
    return (
        sample_img, gt_text,
        gray, blur, deskew, thresh,
        text, conf, acc, status,
    )


# ============================================================================
# OBJECT DETECTION TAB HANDLERS
# ============================================================================


def analyze_detection_fn(
    image: Optional[np.ndarray | Image.Image],
    threshold: float,
    ground_truth_json_str: Optional[str] = None,
):
    """Callback for MobileNet-SSD Object Detection."""
    if OBJECT_DETECTOR is None:
        return (
            None,
            "MobileNet-SSD model is not loaded.",
            "N/A", "N/A", "FAIL",
        )

    if image is None:
        return (
            None,
            "Please upload an image or click 'Run Sample Validation'.",
            "Waiting for image", "N/A", "READY",
        )

    try:
        norm_thresh = threshold / 100.0 if threshold > 1.0 else threshold
        res = OBJECT_DETECTOR.detect(
            image_input=image,
            confidence_threshold=norm_thresh,
            save_annotated=True,
        )

        # Summary text
        lines = [f"=== DETECTED OBJECTS ({len(res.detections)} total, {res.accepted_count} accepted) ==="]
        for d in res.detections:
            tag = "ACCEPTED" if d.accepted else "FILTERED"
            lines.append(f"• [{tag}] {d.class_name.upper()}: {d.confidence_percent:.1f}% | bbox: {d.bbox}")
        summary_text = "\n".join(lines)

        confidence_info = (
            f"Mean Accepted Confidence : {res.mean_confidence:.2f}%\n"
            f"Acceptance Threshold     : {threshold:.0f}% ({norm_thresh:.2f})\n"
            f"Accepted Objects         : {res.accepted_count} / {len(res.detections)}\n"
            f"Rejected Low-Confidence  : {res.rejected_count}\n"
            f"Acceptance Rate          : {res.acceptance_rate:.2f}%"
        )

        accuracy_info = "Provide ground-truth JSON to evaluate detection precision/recall."
        benchmark_passed = res.accepted_count > 0 and res.mean_confidence >= threshold

        if ground_truth_json_str and ground_truth_json_str.strip():
            try:
                gt_data = json.loads(ground_truth_json_str)
                eval_rep = evaluate_object_detections(
                    predictions=res.detections,
                    ground_truth=gt_data,
                    iou_threshold=0.50,
                    confidence_threshold=norm_thresh,
                )
                accuracy_info = (
                    f"Precision   : {eval_rep.precision * 100.0:.2f}%\n"
                    f"Recall      : {eval_rep.recall * 100.0:.2f}%\n"
                    f"F1-Score    : {eval_rep.f1_score * 100.0:.2f}%\n"
                    f"Mean IoU    : {eval_rep.mean_iou * 100.0:.2f}%\n"
                    f"TP / FP / FN: {eval_rep.true_positives} / {eval_rep.false_positives} / {eval_rep.false_negatives}\n"
                    f"Status      : {eval_rep.status}"
                )
                benchmark_passed = eval_rep.passed
            except Exception as e:
                accuracy_info = f"Ground Truth JSON parse error: {e}"

        final_status = "PASS" if benchmark_passed else "FAIL"

        # Convert annotated BGR image to RGB for Gradio display
        annotated_rgb = (
            np.array(Image.fromarray(res.annotated_image[:, :, ::-1]))
            if res.annotated_image is not None
            else None
        )

        return (
            annotated_rgb,
            summary_text,
            confidence_info,
            accuracy_info,
            final_status,
        )

    except Exception as exc:
        logger.error("Detection error: %s", exc, exc_info=True)
        return (
            None,
            f"Error during Object Detection: {exc}",
            "Error", "Error", "FAIL",
        )


def run_sample_detection_fn(threshold: float):
    """Load sample objects image and execute object detection benchmark."""
    if not SAMPLE_OBJECTS_PATH.is_file():
        return (
            None, "",
            None,
            f"Sample image missing at {SAMPLE_OBJECTS_PATH}",
            "Missing", "Missing", "FAIL",
        )

    gt_json = ""
    if GROUND_TRUTH_OBJECTS_PATH.is_file():
        gt_json = GROUND_TRUTH_OBJECTS_PATH.read_text(encoding="utf-8")

    sample_img = Image.open(SAMPLE_OBJECTS_PATH)
    ann_img, summary, conf, acc, status = analyze_detection_fn(
        image=sample_img,
        threshold=threshold,
        ground_truth_json_str=gt_json,
    )
    return (
        sample_img, gt_json,
        ann_img, summary, conf, acc, status,
    )


# ============================================================================
# APP LAYOUT
# ============================================================================


def build_app() -> gr.Blocks:
    """Construct Gradio UI layout with tabs for OCR and Object Detection."""
    with gr.Blocks(title="AI Project 4 - Building the Machine's Optic Nerve") as demo:
        gr.Markdown(
            f"""
            # 👁️ AI Project 4 — Building the Machine's Optic Nerve
            ### Basic Image & Text Recognition System
            * **OCR Engine (Tesseract)**: `{OCR_STATUS_MSG}`
            * **Object Detection (MobileNet-SSD)**: `{DETECTOR_STATUS_MSG}`
            * **Confidence Acceptance Threshold**: `80.0% (0.80)`
            """
        )

        with gr.Tabs():
            # ==================== TAB 1: OCR ====================
            with gr.TabItem("📖 Mode 1: Text Recognition (OCR)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        ocr_input_img = gr.Image(label="Input Image for OCR", type="numpy")
                        ocr_gt_box = gr.Textbox(
                            label="Optional Ground-Truth Text (for Accuracy Evaluation)",
                            lines=3,
                            placeholder="Enter expected ground truth text here...",
                        )
                        with gr.Accordion("⚙️ OCR Engine Parameters", open=False):
                            ocr_psm = gr.Dropdown(
                                label="Page Segmentation Mode (PSM)",
                                choices=list(SUPPORTED_PSM_MODES.keys()),
                                value=DEFAULT_PSM_MODE,
                                info="PSM 6: Uniform block of text (Recommended)",
                            )
                            ocr_thresh = gr.Slider(
                                label="Confidence Acceptance Threshold (%)",
                                minimum=50.0,
                                maximum=100.0,
                                value=CONFIDENCE_THRESHOLD,
                                step=1.0,
                            )
                        with gr.Row():
                            ocr_analyze_btn = gr.Button("🔍 Analyze Text", variant="primary")
                            ocr_sample_btn = gr.Button("🧪 Run Sample Validation", variant="secondary")
                            ocr_clear_btn = gr.Button("🗑️ Clear")

                    with gr.Column(scale=1):
                        ocr_final_status = gr.Textbox(label="Validation Status", value="READY", interactive=False)
                        with gr.Tabs():
                            with gr.TabItem("Recognized Text"):
                                ocr_text_out = gr.Textbox(label="OCR Output", lines=8, interactive=False)
                            with gr.TabItem("Confidence Info"):
                                ocr_conf_out = gr.Textbox(label="Token Confidence", lines=6, interactive=False)
                            with gr.TabItem("Accuracy Benchmark"):
                                ocr_acc_out = gr.Textbox(label="Accuracy Evaluation", lines=6, interactive=False)

                gr.Markdown("#### 🔬 OpenCV Image Preprocessing Pipeline Stages")
                with gr.Row():
                    out_gray = gr.Image(label="1. Grayscale Conversion", interactive=False)
                    out_blur = gr.Image(label="2. Gaussian Noise Blur", interactive=False)
                    out_deskew = gr.Image(label="3. Deskew / Rotation", interactive=False)
                    out_thresh = gr.Image(label="4. Binary Threshold (Otsu)", interactive=False)

                ocr_analyze_btn.click(
                    fn=analyze_ocr_fn,
                    inputs=[ocr_input_img, ocr_psm, ocr_thresh, ocr_gt_box],
                    outputs=[
                        out_gray, out_blur, out_deskew, out_thresh,
                        ocr_text_out, ocr_conf_out, ocr_acc_out, ocr_final_status,
                    ],
                )

                ocr_sample_btn.click(
                    fn=run_sample_ocr_fn,
                    inputs=[ocr_psm, ocr_thresh],
                    outputs=[
                        ocr_input_img, ocr_gt_box,
                        out_gray, out_blur, out_deskew, out_thresh,
                        ocr_text_out, ocr_conf_out, ocr_acc_out, ocr_final_status,
                    ],
                )

                ocr_clear_btn.click(
                    fn=lambda: (None, "", None, None, None, None, "", "", "", "READY"),
                    inputs=[],
                    outputs=[
                        ocr_input_img, ocr_gt_box,
                        out_gray, out_blur, out_deskew, out_thresh,
                        ocr_text_out, ocr_conf_out, ocr_acc_out, ocr_final_status,
                    ],
                )

            # ==================== TAB 2: OBJECT DETECTION ====================
            with gr.TabItem("🎯 Mode 2: Object Detection (MobileNet-SSD)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        det_input_img = gr.Image(label="Input Image for Object Detection", type="numpy")
                        det_gt_box = gr.Textbox(
                            label="Optional Ground-Truth JSON (for Precision/Recall/IoU)",
                            lines=4,
                            placeholder='{"objects": [{"class_name": "person", "bbox": [71, 63, 460, 328]}]}',
                        )
                        det_thresh = gr.Slider(
                            label="Detection Confidence Threshold (%)",
                            minimum=50.0,
                            maximum=100.0,
                            value=CONFIDENCE_THRESHOLD,
                            step=1.0,
                            info="Detections with confidence >= threshold are rendered with bounding boxes.",
                        )
                        with gr.Row():
                            det_analyze_btn = gr.Button("🎯 Detect Objects", variant="primary")
                            det_sample_btn = gr.Button("🧪 Run Sample Validation", variant="secondary")
                            det_clear_btn = gr.Button("🗑️ Clear")

                    with gr.Column(scale=1):
                        det_final_status = gr.Textbox(label="Validation Status", value="READY", interactive=False)
                        det_annotated_out = gr.Image(label="Annotated Detections", interactive=False)
                        with gr.Tabs():
                            with gr.TabItem("Detection List"):
                                det_summary_out = gr.Textbox(label="Detected Objects", lines=6, interactive=False)
                            with gr.TabItem("Confidence Info"):
                                det_conf_out = gr.Textbox(label="Confidence Metrics", lines=5, interactive=False)
                            with gr.TabItem("Accuracy Benchmark"):
                                det_acc_out = gr.Textbox(label="Precision, Recall & IoU", lines=6, interactive=False)

                det_analyze_btn.click(
                    fn=analyze_detection_fn,
                    inputs=[det_input_img, det_thresh, det_gt_box],
                    outputs=[
                        det_annotated_out, det_summary_out, det_conf_out, det_acc_out, det_final_status,
                    ],
                )

                det_sample_btn.click(
                    fn=run_sample_detection_fn,
                    inputs=[det_thresh],
                    outputs=[
                        det_input_img, det_gt_box,
                        det_annotated_out, det_summary_out, det_conf_out, det_acc_out, det_final_status,
                    ],
                )

                det_clear_btn.click(
                    fn=lambda: (None, "", None, "", "", "", "READY"),
                    inputs=[],
                    outputs=[
                        det_input_img, det_gt_box,
                        det_annotated_out, det_summary_out, det_conf_out, det_acc_out, det_final_status,
                    ],
                )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())
