# AI Signature Detection & Cropper

An AI-based pipeline for detecting handwritten signatures in document images, cropping candidate signature regions, and verifying whether each candidate is actually a handwritten signature.

The pipeline takes a document image and:

1. Preprocesses it using grayscale conversion, deskewing, and thresholding.
2. Detects candidate signature regions using a pretrained YOLOv8s signature detector.
3. Crops each detected region with padding.
4. Uses an independent Qwen3-VL 2B vision-language model to verify whether each crop is a handwritten signature.
5. Separates accepted and rejected candidates.
6. Evaluates the pipeline against fixed ground-truth annotations.

## Technologies Used

* Python
* OpenCV - image preprocessing and cropping
* Hugging Face Datasets - dataset loading
* YOLOv8s / Ultralytics - signature detection
* PyTorch - underlying deep learning framework
* NumPy - bounding-box and IoU calculations
* Ollama + Qwen3-VL 2B - signature verification
* scikit-learn - evaluation metrics

## Approach

A classical computer-vision contour-based approach was initially tested for detecting candidate regions. However, it frequently detected large regions containing unrelated document text and structures.

A pretrained signature-specific YOLO detector was therefore selected because it directly predicts signature bounding boxes.

A second independent model was added as a verification stage. The detector is designed to find possible signatures, while Qwen3-VL independently checks each cropped candidate with a binary YES/NO decision.

## Dataset

The project uses the `tech4humans/signature-detection` dataset from Hugging Face.

A fixed evaluation set was created before final detector evaluation:

* 40 document images
* 10 pages with no signatures
* 15 pages with one signature
* 15 pages with multiple signatures
* 48 ground-truth signatures

Ground-truth annotations are stored in `data/ground_truth.json`.

## Evaluation

Predictions are considered correct when their Intersection over Union (IoU) with a ground-truth signature is at least 0.5.
YOLO detector alone:
* Precision: 73.77%
* Recall: 93.75%
* F1: 82.57%
* True Positives: 45
* False Positives: 16
* False Negatives: 3

YOLO + verifier:
* Precision: 85.11%
* Recall: 83.33%
* F1: 84.21%
* True Positives: 40
* False Positives: 7
* False Negatives: 8
The verifier reduced false positives from 16 to 7 and increased precision, but also rejected some genuine signatures, reducing recall.

## How to Run

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python src/preprocess.py
python src/detect.py
python src/crop_candidates.py
```

Install and start the local verification model:

```bash
ollama pull qwen3-vl:2b
```

Run verification and evaluation:

```bash
python src/verify.py
python src/evaluate.py
```

Evaluation results are saved to `outputs/evaluation_results.json`.

