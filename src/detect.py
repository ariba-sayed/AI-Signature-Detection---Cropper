from pathlib import Path
import cv2
import json

from ultralytics import YOLO
from huggingface_hub import hf_hub_download
# -----------------------------
# Paths
# -----------------------------

INPUT_DIR = Path("data/evaluation")
OUTPUT_DIR = Path("outputs/detections")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load pretrained model
# -----------------------------
from huggingface_hub import hf_hub_download

print("Downloading signature detector...")

MODEL_PATH = hf_hub_download(
    repo_id="tech4humans/yolov8s-signature-detector",
    filename="yolov8s.pt"
)

print("Loading signature detection model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")
# -----------------------------
# Detect signatures
# -----------------------------

def detect_candidates(image_path):
    """
    Detect signature regions using the pretrained YOLO model.
    """

    results = model.predict(
        source=str(image_path),
        conf=0.25,
        verbose=False
    )

    result = results[0]

    candidates = []

    if result.boxes is None:
        return candidates

    for box in result.boxes:

        # Bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Confidence score
        confidence = float(box.conf[0])

        candidates.append({
            "bbox": [
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ],
            "confidence": confidence
        })

    return candidates


# -----------------------------
# Process one image
# -----------------------------

def process_image(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read {image_path}")
        return

    candidates = detect_candidates(image_path)

    print(f"\n{image_path.name}")
    print(f"Signatures detected: {len(candidates)}")

    # -------------------------
    # Save detection JSON
    # -------------------------

    json_data = {
        "image": image_path.name,
        "candidates": candidates
    }

    json_path = OUTPUT_DIR / f"{image_path.stem}.json"

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)

    # -------------------------
    # Draw detections
    # -------------------------

    visualization = image.copy()

    for i, candidate in enumerate(candidates):

        x1, y1, x2, y2 = candidate["bbox"]
        confidence = candidate["confidence"]

        cv2.rectangle(
            visualization,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        label = f"Signature {i + 1}: {confidence:.2f}"

        cv2.putText(
            visualization,
            label,
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

    output_image_path = OUTPUT_DIR / image_path.name

    cv2.imwrite(
        str(output_image_path),
        visualization
    )


# -----------------------------
# Process all evaluation images
# -----------------------------

image_files = sorted(INPUT_DIR.glob("*.png"))

print(f"Found {len(image_files)} evaluation images.")

for image_path in image_files:
    process_image(image_path)

print("\nDetection complete.")
print(f"Results saved to: {OUTPUT_DIR}")