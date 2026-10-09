# Project 4 — Building the Machine's Optic Nerve: Basic Image & Text Recognition

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision%204.x-green.svg)](https://opencv.org/)
[![Tesseract OCR](https://img.shields.io/badge/Tesseract-OCR%205.4-orange.svg)](https://github.com/tesseract-ocr/tesseract)
[![MobileNet--SSD](https://img.shields.io/badge/Model-MobileNet--SSD%20(Caffe)-yellow.svg)](https://github.com/chuanqi305/MobileNet-SSD)
[![Gradio UI](https://img.shields.io/badge/UI-Gradio-red.svg)](https://gradio.app/)
[![Tests](https://img.shields.io/badge/Tests-42%20Passed-brightgreen.svg)](https://pytest.org/)

---

## 1. Project Objective & Academic Context

In biological vision, the optic nerve transduces photonic stimuli gathered by retinal photoreceptors into electrochemical signals deciphered by the visual cortex. For an artificial intelligence system, digital image processing, **Optical Character Recognition (OCR)**, and **Object Detection** constitute this sensory gateway—the machine's optic nerve.

This project delivers an end-to-end, dual-modality computer vision system:
1. **Text Recognition (OCR) Pipeline**: Classical OpenCV image preprocessing (Grayscale conversion, Gaussian noise smoothing, Deskew orientation correction, Otsu's binary thresholding) coupled with Google's Tesseract LSTM OCR engine.
2. **Object Detection Pipeline**: Deep Convolutional Neural Network inference using OpenCV's DNN module and a pre-trained MobileNet-SSD architecture.
3. **80% Confidence Filtering**: Enforces an 80% ($0.80$) acceptance threshold on genuine model confidence scores.
4. **Independent Ground-Truth Benchmarking**: Computes Character Error Rate (CER), Word Error Rate (WER), Character Accuracy, Word Accuracy, Intersection over Union (IoU), Precision, Recall, and F1-score against independent ground truth.
5. **Interactive & Automated Interfaces**: A dual-tab Gradio Web UI and a headless DevOps CLI with exit-code verification.

---

## 2. System Architecture

```
                                  ┌────────────────────────┐
                                  │      INPUT IMAGE       │
                                  └───────────┬────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
         [MODE 1: TEXT RECOGNITION]                     [MODE 2: OBJECT DETECTION]
                      │                                               │
                      ▼                                               ▼
     ┌─────────────────────────────────┐             ┌─────────────────────────────────┐
     │   OpenCV Preprocessing Stage    │             │       OpenCV DNN Preprocessing  │
     │  1. Grayscale Conversion        │             │  • cv2.dnn.blobFromImage        │
     │  2. Gaussian Noise Filtering    │             │  • Target Dimensions: 300 x 300 │
     │  3. Deskew Contour Rotation     │             │  • Scalefactor: 0.007843        │
     │  4. Otsu Binary Thresholding    │             │  • Mean Subtraction: 127.5      │
     └────────────────┬────────────────┘             └────────────────┬────────────────┘
                      │                                               │
                      ▼                                               ▼
     ┌─────────────────────────────────┐             ┌─────────────────────────────────┐
     │      Tesseract OCR Engine       │             │   MobileNet-SSD Forward Pass    │
     │  • Configurable PSM (3/6/7/11)  │             │  • 21 PASCAL VOC Classes        │
     │  • image_to_data() Tokens       │             │  • Bounding Box Tensor Output   │
     │  • Real Word Confidences [0-100]│             │  • Normalized Coords [0.0 - 1.0]│
     └────────────────┬────────────────┘             └────────────────┬────────────────┘
                      │                                               │
                      ▼                                               ▼
     ┌─────────────────────────────────┐             ┌─────────────────────────────────┐
     │   Confidence Filter (>= 80%)    │             │   Confidence Filter (>= 0.80)   │
     │  • Accept words with conf >= 80 │             │  • Keep boxes with score >= 0.80│
     │  • Calculate Token Acceptance % │             │  • Rescale to Original Pixels   │
     └────────────────┬────────────────┘             └────────────────┬────────────────┘
                      │                                               │
                      ▼                                               ▼
     ┌─────────────────────────────────┐             ┌─────────────────────────────────┐
     │   Ground-Truth Evaluation       │             │   Ground-Truth Evaluation       │
     │  • Text Normalization           │             │  • IoU (Intersection over Union)│
     │  • Character Error Rate (CER)   │             │  • Precision, Recall, F1-Score  │
     │  • Word Error Rate (WER)        │             │  • Target: F1 >= 80.0%          │
     │  • Target: Accuracy >= 80.0%    │             └────────────────┬────────────────┘
     └────────────────┬────────────────┘                              │
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              ▼
                             ┌─────────────────────────────────┐
                             │       OUTPUT INTERFACES         │
                             │  • Gradio Interactive Web App   │
                             │  • Headless CLI Runner          │
                             │  • outputs/ Visual & JSON Files │
                             └─────────────────────────────────┘
```

---

## 3. Technology Stack

- **Python**: 3.10+ (Tested on Python 3.14.5 64-bit)
- **Computer Vision**: `OpenCV 4.14.0` (`opencv-python-headless`)
- **OCR Engine**: `Tesseract OCR v5.4.0` via `pytesseract`
- **Object Detection Model**: `MobileNet-SSD (PASCAL VOC 2007/2012)` via OpenCV DNN
- **Image Manipulation**: `Pillow 12.3.0`, `NumPy 2.5.1`
- **User Interface**: `Gradio 6.28.0`
- **Testing**: `pytest 9.1.1` (42 automated tests passing)
- **Standard Utilities**: `pathlib`, `difflib`, `argparse`, `json`, `dataclasses`, `logging`

---

## 4. Methodology & Mathematical Formulations

### A. Preprocessing Pipeline (`src/image_preprocessing.py`)

1. **Grayscale Conversion**:
   Collapses 3-channel BGR color data into luminance intensity:
   $$Y = 0.299R + 0.587G + 0.114B$$
2. **Gaussian Blur**:
   Attenuates high-frequency sensor noise while preserving stroke boundaries using a 2D Gaussian kernel:
   $$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$
3. **Deskewing (Orientation Correction)**:
   Extracts minimum area bounding box of non-zero foreground text contours (`cv2.minAreaRect`). Calculates skew angle $\theta$ and rotates the image around its centroid with white background padding.
4. **Otsu's Thresholding**:
   Automatically calculates the binarization threshold $t^*$ that minimizes intra-class variance $\sigma_w^2(t)$:
   $$\sigma_w^2(t) = \omega_0(t)\sigma_0^2(t) + \omega_1(t)\sigma_1^2(t)$$

### B. OCR Confidence & Evaluation Metrics (`src/evaluator.py`)

Tesseract's internal LSTM classifier outputs word-level confidences via `image_to_data()`.

Accuracy metrics are computed against independent ground truth:
- **Levenshtein Distance ($D_L$)**: Minimum operations (insertions, deletions, substitutions) to transform ground truth into prediction.
- **Character Error Rate (CER)**:
  $$\text{CER} = \frac{D_L(S_{\text{ref\_chars}}, S_{\text{hyp\_chars}})}{\max(1, |S_{\text{ref\_chars}}|)}$$
- **Character Accuracy**:
  $$\text{Accuracy}_{\text{char}} = \max\left(0, 1 - \text{CER}\right) \times 100\%$$
- **Word Error Rate (WER)**:
  $$\text{WER} = \frac{D_L(S_{\text{ref\_words}}, S_{\text{hyp\_words}})}{\max(1, |S_{\text{ref\_words}}|)}$$
- **Word Accuracy**:
  $$\text{Accuracy}_{\text{word}} = \max\left(0, 1 - \text{WER}\right) \times 100\%$$

### C. Object Detection & Spatial Metrics (`src/object_detector.py`)

Input images are processed with `cv2.dnn.blobFromImage` (size: $300 \times 300$, scale: $0.007843$, mean: $127.5$).

- **Intersection over Union (IoU)**:
  $$\text{IoU}(B_p, B_g) = \frac{\text{Area}(B_p \cap B_g)}{\text{Area}(B_p \cup B_g)}$$
- **Precision, Recall, and F1-Score**:
  $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 5. Project Directory Structure

```
AI_Project_4_Text_Recognition/
│
├── app.py                         # Dual-Mode Gradio Web Application
├── main.py                        # Multi-Modal DevOps CLI Runner
├── requirements.txt               # Project Dependencies
├── README.md                      # Comprehensive Project Documentation
├── .gitignore                     # Git Exclusion Rules
│
├── data/
│   ├── sample_input.png           # Controlled OCR Sample Image
│   ├── sample_text.png            # OCR Sample Alias
│   ├── expected_output.txt        # Controlled OCR Ground Truth
│   ├── expected_text.txt          # OCR Ground Truth Alias
│   ├── sample_objects.jpg         # Controlled Object Detection Image
│   └── ground_truth.json          # Object Detection Annotations
│
├── models/
│   ├── MobileNetSSD_deploy.prototxt    # Caffe Network Architecture
│   └── MobileNetSSD_deploy.caffemodel  # Pre-Trained Weights (23MB)
│
├── src/
│   ├── __init__.py                # Package Initializer
│   ├── config.py                  # Configurations, Paths & Binary Discovery
│   ├── preprocessing.py           # Preprocessing Module Interface
│   ├── image_preprocessing.py     # OpenCV Pipeline (Grayscale, Blur, Deskew, Otsu)
│   ├── ocr_engine.py              # Tesseract Engine & Token Confidence Extractor
│   ├── object_detector.py         # MobileNet-SSD Deep Learning Detector
│   ├── evaluator.py               # Levenshtein CER, WER, IoU & F1 Metrics
│   └── pipeline.py                # Unified OCR Pipeline Orchestrator
│
├── tests/
│   ├── __init__.py                # Test Suite Initializer
│   ├── test_preprocessing.py      # 11 OpenCV Unit Tests
│   ├── test_ocr.py                # 9 Tesseract Unit & Integration Tests
│   ├── test_object_detection.py   # 7 MobileNet-SSD Unit & Integration Tests
│   ├── test_evaluator.py          # 12 Error Rate & Metric Tests
│   └── test_pipeline.py           # 3 End-to-End Pipeline Tests
│
└── outputs/
    ├── grayscale.png              # Intermediate Grayscale Image
    ├── blurred.png                # Intermediate Gaussian Blurred Image
    ├── deskewed.png               # Intermediate Deskewed Image
    ├── thresholded.png            # Intermediate Binarized Image
    ├── ocr_result.json            # Machine-Readable OCR Benchmark
    ├── annotated_detection.png    # MobileNet-SSD Annotated Bounding Boxes
    ├── detection_result.json      # Machine-Readable Detection Benchmark
    └── requirement_checklist.md   # Requirement Traceability Matrix
```

---

## 6. Installation & Environment Setup

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Tesseract OCR Engine (Windows)
The project dynamically searches for Tesseract across standard system locations:
- `C:\Users\<User>\AppData\Local\Programs\Tesseract-OCR\tesseract.exe`
- `C:\Program Files\Tesseract-OCR\tesseract.exe`
- `C:\Program Files (x86)\Tesseract-OCR\tesseract.exe`
- System `PATH`
- Custom environment variable: `TESSERACT_CMD`

If not installed, install via winget:
```cmd
winget install --id UB-Mannheim.TesseractOCR
```

### 3. Model Files
The MobileNet-SSD Caffe model files are pre-downloaded and stored in `models/`:
- `models/MobileNetSSD_deploy.prototxt`
- `models/MobileNetSSD_deploy.caffemodel`

---

## 7. Execution Guide

### A. Automated DevOps Benchmark Validation
Run the automated benchmark on the controlled samples. Exits with code `0` on success, non-zero on failure.

**1. OCR Text Recognition Benchmark:**
```bash
python main.py --validate
```
*Expected Output:*
```text
========================================
OCR VALIDATION REPORT
========================================
Character Accuracy : 100.00%
Word Accuracy      : 100.00%
Char Error Rate    : 0.0000
Word Error Rate    : 0.0000
OCR Confidence     : 94.89%
Threshold          : 80%
Status             : PASS
========================================
```

**2. MobileNet-SSD Object Detection Benchmark:**
```bash
python main.py --mode detect --validate
```
*Expected Output:*
```text
========================================
OBJECT DETECTION VALIDATION REPORT
========================================
Precision          : 100.00%
Recall             : 100.00%
F1 Score           : 100.00%
Mean IoU           : 95.92%
True Positives     : 1
False Positives    : 0
False Negatives    : 0
Threshold          : 80%
Status             : PASS
========================================
```

### B. Single Image CLI Execution

**Process an Image with OCR:**
```bash
python main.py --mode ocr --image data/sample_text.png --psm 6 --threshold 80
```

**Detect Objects in an Image:**
```bash
python main.py --mode detect --image data/sample_objects.jpg --threshold 80
```

### C. Launching the Gradio Web Application
Start the interactive local web server:
```bash
python app.py
```
Open your browser at `http://127.0.0.1:7860`.
Features:
- **Tab 1 (OCR)**: Live visualization of all 4 preprocessing steps, raw/filtered text output, token confidence metrics, and accuracy benchmarks.
- **Tab 2 (Object Detection)**: Upload any image to render bounding boxes, view detected classes, filter low-confidence objects, and compute Precision/Recall/IoU against ground truth.

### D. Automated Test Suite
Run the test suite across all 5 test modules:
```bash
python -m pytest -v
```
Or directly through the CLI:
```bash
python main.py --test
```

### E. Bytecode Compilation Check
```bash
python -m compileall .
```

---

## 8. Verified Experimental Results

| Metric | OCR Text Recognition | MobileNet-SSD Object Detection | Quality Target | Status |
|:---|:---:|:---:|:---:|:---:|
| **Character Accuracy** | **100.00%** | N/A | $\ge 80.0\%$ | **PASS** |
| **Word Accuracy** | **100.00%** | N/A | $\ge 80.0\%$ | **PASS** |
| **Char Error Rate (CER)** | **0.0000** | N/A | $\le 0.20$ | **PASS** |
| **Word Error Rate (WER)** | **0.0000** | N/A | $\le 0.20$ | **PASS** |
| **Mean Model Confidence** | **94.89%** | **100.00%** | $\ge 80.0\%$ | **PASS** |
| **Precision** | N/A | **100.00%** | $\ge 80.0\%$ | **PASS** |
| **Recall** | N/A | **100.00%** | $\ge 80.0\%$ | **PASS** |
| **F1-Score** | N/A | **100.00%** | $\ge 80.0\%$ | **PASS** |
| **Mean IoU** | N/A | **95.92%** | $\ge 50.0\%$ | **PASS** |
| **Automated Tests** | 42 passed / 0 failed | 42 passed / 0 failed | 100% Pass | **PASS** |

---

## 9. Troubleshooting

| Symptom | Root Cause | Resolution |
|:---|:---|:---|
| `TesseractNotFoundError` | Binary missing from PATH and standard directories | Run `winget install UB-Mannheim.TesseractOCR` or set `TESSERACT_CMD` environment variable. |
| `ModelNotFoundError` | Prototxt or caffemodel missing in `models/` | Ensure `MobileNetSSD_deploy.prototxt` and `MobileNetSSD_deploy.caffemodel` reside in `models/`. |
| Port `7860` in use | Previous Gradio process still active | Terminate the existing process or launch with an alternative port in `app.py`. |

---

## 10. Limitations & Future Extensions

1. **Unconstrained Handwriting**: Tesseract's standard LSTM models are optimized for printed typography. Future work could integrate Transformer-based models (e.g., TrOCR) for complex handwriting.
2. **Dense Multi-Scale Object Detection**: MobileNet-SSD is optimized for high-speed edge inference on standard resolutions. Advanced modern architectures (e.g., YOLOv10, RT-DETR) can be incorporated for small or overlapping objects.
