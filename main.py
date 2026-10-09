"""Command Line Interface (CLI) for AI Project 4 - Text Recognition (OCR) & Object Detection.

Supports:
1. OCR Text Recognition on input images.
2. MobileNet-SSD Object Detection on input images.
3. Automated benchmark validation against ground truth (--validate).
4. Automated pytest test execution (--test).

Usage Examples:
    python main.py --validate
    python main.py --mode ocr --image data/sample_text.png
    python main.py --mode detect --image data/sample_objects.jpg
    python main.py --test
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from src.config import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_PSM_MODE,
    EXPECTED_OUTPUT_PATH,
    OUTPUTS_DIR,
    SAMPLE_INPUT_PATH,
    SUPPORTED_PSM_MODES,
)
from src.evaluator import evaluate_object_detections
from src.object_detector import ObjectDetector, ModelNotFoundError
from src.ocr_engine import TesseractNotFoundError
from src.pipeline import OCRPipeline

logger = logging.getLogger("MainCLI")

SAMPLE_OBJECTS_PATH = Path("data/sample_objects.jpg")
GROUND_TRUTH_OBJECTS_PATH = Path("data/ground_truth.json")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Project 4 - Building the Machine's Optic Nerve: Text Recognition & Object Detection CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["ocr", "detect"],
        default="ocr",
        help="Operating mode: 'ocr' for Optical Character Recognition, 'detect' for MobileNet-SSD object detection.",
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default=None,
        help="Path to input image for processing.",
    )
    parser.add_argument(
        "--validate",
        "-v",
        action="store_true",
        help="Execute automated benchmark validation on controlled sample data.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the complete automated pytest test suite and report results.",
    )
    parser.add_argument(
        "--psm",
        type=str,
        default=DEFAULT_PSM_MODE,
        choices=list(SUPPORTED_PSM_MODES.keys()),
        help="Tesseract Page Segmentation Mode (PSM) for OCR.",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Confidence and accuracy acceptance threshold percentage [0-100] (for detection: converted to [0-1]).",
    )
    parser.add_argument(
        "--expected",
        "-e",
        type=str,
        default=str(EXPECTED_OUTPUT_PATH),
        help="Path to expected ground-truth text or JSON file for accuracy evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=str(OUTPUTS_DIR),
        help="Directory where intermediate images, annotated outputs, and reports are stored.",
    )
    return parser.parse_args()


def run_test_suite() -> int:
    """Execute pytest test suite via python subprocess."""
    print("=" * 60)
    print("EXECUTING AUTOMATED TEST SUITE")
    print("=" * 60)
    cmd = [sys.executable, "-m", "pytest", "-v"]
    res = subprocess.run(cmd)
    return res.returncode


def run_ocr_validation(
    pipeline: OCRPipeline,
    image_path: Path,
    expected_path: Path,
    psm_mode: str,
    threshold: float,
    output_dir: Path,
) -> int:
    """Execute automated OCR validation benchmark against expected text."""
    logger.info("Executing automated OCR validation benchmark on: %s", image_path)

    if not image_path.is_file():
        logger.error("Sample image not found at: %s", image_path)
        return 1

    if not expected_path.is_file():
        logger.error("Ground truth file not found at: %s", expected_path)
        return 1

    result = pipeline.process(
        image_input=image_path,
        expected_text_path=expected_path,
        save_artifacts=True,
        artifacts_dir=output_dir,
        psm_mode=psm_mode,
        confidence_threshold=threshold,
    )

    if not result.validation:
        logger.error("Validation failed to generate report.")
        return 1

    print("\n" + result.validation.format_report() + "\n")
    print("DETAILED RESULTS:")
    print(f"- Raw Extracted Text    :\n{result.ocr.raw_text.strip()}")
    print(f"- Filtered Text (>= {threshold:.0f}%): {result.ocr.filtered_text}")
    print(f"- Total Tokens Detected : {len(result.ocr.tokens)}")
    print(f"- Tokens Accepted       : {result.ocr.accepted_token_count}")
    print(f"- Tokens Rejected       : {result.ocr.rejected_token_count}")
    print(f"- Acceptance Rate       : {result.ocr.acceptance_rate:.2f}%")
    print(f"- PSM Mode Used         : --psm {result.ocr.psm_mode} ({SUPPORTED_PSM_MODES.get(result.ocr.psm_mode, '')})")
    print(f"- Output JSON Saved To  : {output_dir / 'ocr_result.json'}")

    return 0 if result.validation.passed else 1


def run_detection_validation(
    detector: ObjectDetector,
    image_path: Path,
    ground_truth_path: Path,
    threshold_percent: float,
    output_dir: Path,
) -> int:
    """Execute automated Object Detection benchmark against ground truth JSON."""
    logger.info("Executing automated Object Detection benchmark on: %s", image_path)

    if not image_path.is_file():
        logger.error("Sample image not found at: %s", image_path)
        return 1

    if not ground_truth_path.is_file():
        logger.error("Ground truth JSON not found at: %s", ground_truth_path)
        return 1

    norm_thresh = threshold_percent / 100.0 if threshold_percent > 1.0 else threshold_percent

    res = detector.detect(
        image_input=image_path,
        confidence_threshold=norm_thresh,
        save_annotated=True,
        output_dir=output_dir,
    )

    report = evaluate_object_detections(
        predictions=res.detections,
        ground_truth=ground_truth_path,
        iou_threshold=0.50,
        confidence_threshold=norm_thresh,
    )

    print("\n" + report.format_report() + "\n")
    print("DETAILED DETECTION RESULTS:")
    print(f"- Total Objects Detected: {len(res.detections)}")
    print(f"- Accepted (>= {threshold_percent:.0f}%): {res.accepted_count}")
    print(f"- Mean Confidence       : {res.mean_confidence:.2f}%")
    print(f"- Annotated Image Saved : {res.saved_annotated_path}")

    # Also save detection evaluation JSON
    det_json_path = output_dir / "detection_result.json"
    det_dict = {
        "status": report.status,
        "precision": report.precision,
        "recall": report.recall,
        "f1_score": report.f1_score,
        "mean_iou": report.mean_iou,
        "threshold": norm_thresh,
        "accepted_objects": [
            {"class": d.class_name, "confidence": d.confidence_percent, "bbox": list(d.bbox)}
            for d in res.detections if d.accepted
        ],
    }
    with open(det_json_path, "w", encoding="utf-8") as f:
        json.dump(det_dict, f, indent=2)

    return 0 if report.passed else 1


def main() -> int:
    """Main CLI entrypoint."""
    args = parse_arguments()

    if args.test:
        return run_test_suite()

    output_dir = Path(args.output_dir)

    # 1. Object Detection Mode
    if args.mode == "detect":
        try:
            detector = ObjectDetector()
        except ModelNotFoundError as err:
            logger.critical("\n[CONFIGURATION ERROR] %s\n", err)
            return 2

        img_path = Path(args.image) if args.image else SAMPLE_OBJECTS_PATH
        # If user specified a JSON ground truth, use it; otherwise default to GROUND_TRUTH_OBJECTS_PATH
        if args.expected and args.expected.endswith(".json") and Path(args.expected).is_file():
            gt_path = Path(args.expected)
        else:
            gt_path = GROUND_TRUTH_OBJECTS_PATH

        if args.validate or (not args.image):
            return run_detection_validation(
                detector=detector,
                image_path=img_path,
                ground_truth_path=gt_path,
                threshold_percent=args.threshold,
                output_dir=output_dir,
            )
        else:
            norm_thresh = args.threshold / 100.0 if args.threshold > 1.0 else args.threshold
            res = detector.detect(
                image_input=img_path,
                confidence_threshold=norm_thresh,
                save_annotated=True,
                output_dir=output_dir,
            )
            print("\n" + "=" * 40)
            print("OBJECT DETECTION OUTPUT")
            print("=" * 40)
            print(f"Accepted Detections (>= {args.threshold:.0f}%): {res.accepted_count}")
            for d in res.detections:
                if d.accepted:
                    print(f"  • {d.class_name.upper()} ({d.confidence_percent:.1f}%) at bbox {d.bbox}")
            print(f"Annotated Image: {res.saved_annotated_path}")
            print("=" * 40)
            return 0

    # 2. OCR Mode (Default)
    else:
        try:
            pipeline = OCRPipeline()
        except TesseractNotFoundError as err:
            logger.critical("\n[CONFIGURATION ERROR] %s\n", err)
            return 2

        expected_path = Path(args.expected)

        if args.validate or (not args.image):
            sample_img = Path(args.image) if args.image else SAMPLE_INPUT_PATH
            return run_ocr_validation(
                pipeline=pipeline,
                image_path=sample_img,
                expected_path=expected_path,
                psm_mode=args.psm,
                threshold=args.threshold,
                output_dir=output_dir,
            )
        else:
            target_img = Path(args.image)
            if not target_img.is_file():
                logger.error("Image file not found: %s", target_img)
                return 1

            result = pipeline.process(
                image_input=target_img,
                expected_text_path=expected_path if expected_path.is_file() else None,
                save_artifacts=True,
                artifacts_dir=output_dir,
                psm_mode=args.psm,
                confidence_threshold=args.threshold,
            )

            print("\n" + "=" * 40)
            print("RECOGNIZED OCR OUTPUT")
            print("=" * 40)
            print(result.ocr.raw_text.strip() or "[NO TEXT DETECTED]")
            print("-" * 40)
            print(f"Mean Confidence    : {result.ocr.mean_confidence:.2f}%")
            print(f"Accepted Tokens    : {result.ocr.accepted_token_count} / {len(result.ocr.tokens)}")
            print(f"Filtered (>= {args.threshold:.0f}%) : {result.ocr.filtered_text}")
            print("=" * 40)

            if result.validation:
                print("\n" + result.validation.format_report() + "\n")
                return 0 if result.validation.passed else 1

            return 0 if result.ocr.mean_confidence >= args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
