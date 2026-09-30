from pathlib import Path
import json
import numpy as np

GROUND_TRUTH_FILE = Path("data/ground_truth.json")
DETECTION_DIR = Path("outputs/detections")
VERIFICATION_DIR = Path("outputs/verification")

# A prediction is considered correct when
# IoU is at least 0.5.
IOU_THRESHOLD = 0.5
def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU).

    Box format:
    [x1, y1, x2, y2]
    """

    # Convert the boxes into NumPy arrays.
    box1 = np.array(box1, dtype=float)
    box2 = np.array(box2, dtype=float)

    # Find the coordinates of the overlapping area.
    x1 = np.maximum(box1[0], box2[0])
    y1 = np.maximum(box1[1], box2[1])
    x2 = np.minimum(box1[2], box2[2])
    y2 = np.minimum(box1[3], box2[3])

    # Calculate width and height of the overlap.
    overlap_width = np.maximum(0, x2 - x1)
    overlap_height = np.maximum(0, y2 - y1)

    # Area where the two boxes overlap.
    intersection = overlap_width * overlap_height

    # Area of each box.
    area1 = (
        (box1[2] - box1[0]) *
        (box1[3] - box1[1])
    )

    area2 = (
        (box2[2] - box2[0]) *
        (box2[3] - box2[1])
    )

    # Total area covered by either box.
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    # IoU = intersection / union
    return intersection / union

def load_ground_truth():
    """
    Load the ground-truth annotations.
    """

    with open(GROUND_TRUTH_FILE, "r") as file:
        ground_truth = json.load(file)

    return ground_truth

def load_detections(image_name):
    """
    Load YOLO detections for one image.
    """

    json_file = (
        DETECTION_DIR /
        f"{Path(image_name).stem}.json"
    )

    if not json_file.exists():
        return []

    with open(json_file, "r") as file:
        data = json.load(file)

    return data.get("candidates", [])


def load_verification_decision(
    image_name,
    candidate_number
):
    """
    Find out whether the second model
    accepted or rejected a candidate crop.
    """

    verification_file = (
        VERIFICATION_DIR /
        f"{Path(image_name).stem}"
        f"_candidate_{candidate_number:03d}.json"
    )

    if not verification_file.exists():
        return "NO"

    with open(verification_file, "r") as file:
        data = json.load(file)

    return data.get("decision", "NO")

def match_predictions(
    predictions,
    ground_truth_boxes
):
    """
    Match predicted boxes with ground-truth boxes.

    Each ground-truth signature can be matched
    to only one prediction.
    """

    matches = []

    # Keep track of ground-truth boxes
    # that have already been matched.
    matched_ground_truth = set()

    # Check high-confidence predictions first.
    predictions = sorted(predictions,key=lambda prediction:prediction.get("confidence", 0),reverse=True
    )

    for prediction in predictions:
        best_iou = 0
        best_ground_truth = None
        # Compare this prediction with
        # every unused ground-truth box.
        for index, ground_truth_box in enumerate(ground_truth_boxes):
            if index in matched_ground_truth:
                continue
            iou = calculate_iou(prediction["bbox"],ground_truth_box)
            if iou > best_iou:
                best_iou = iou
                best_ground_truth = index
        # If the best match has IoU >= 0.5,
        # count it as a correct detection.
        if (best_ground_truth is not None and best_iou >= IOU_THRESHOLD):
            matches.append({
                "prediction": prediction,
                "ground_truth_index": best_ground_truth,
                "iou": best_iou
            })
            matched_ground_truth.add(
                best_ground_truth
            )
    return matches

#metrics

from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(tp, fp, fn):
    y_true = (
        [1] * tp +
        [1] * fn +
        [0] * fp
    )

    y_pred = (
        [1] * tp +
        [0] * fn +
        [1] * fp
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    return precision, recall, f1

# -----------------------------
# 7. Evaluate one pipeline stage
# -----------------------------

def evaluate(ground_truth,use_verifier=False):
    """
    Evaluate either:

    1. YOLO alone
    2. YOLO + verifier
    """

    true_positive = 0
    false_positive = 0
    false_negative = 0

    no_signature_false_positive_pages = []

    # Process all 40 evaluation images.
    for image_name, image_data in ground_truth.items():

        # Get the real signature boxes.
        ground_truth_boxes = [signature["bbox"] for signature in image_data["signatures"]]

        # Get YOLO predictions.
        detections = load_detections(image_name)

        predictions = []

        # Go through every YOLO candidate.
        for candidate_number, detection in enumerate(detections,start=1):
            # If we are evaluating the complete
            # pipeline, only keep candidates
            # accepted by the verifier.
            if use_verifier:
                decision = load_verification_decision(image_name,candidate_number)
                if decision != "YES":
                    continue

            predictions.append(detection)

        # Match predictions to real signatures.
        matches = match_predictions(
            predictions,
            ground_truth_boxes
        )

        # Number of correct predictions.
        image_tp = len(matches)

        # Predictions that did not match
        # any ground-truth signature.
        image_fp = (
            len(predictions) - image_tp
        )

        # Real signatures that were not detected.
        image_fn = (
            len(ground_truth_boxes) - image_tp
        )

        true_positive += image_tp
        false_positive += image_fp
        false_negative += image_fn

        # Special requirement:
        # Count no-signature pages that
        # produced at least one false positive.
        if (
            len(ground_truth_boxes) == 0
            and image_fp > 0
        ):
            no_signature_false_positive_pages.append(
                image_name
            )

    # Calculate final metrics.
    precision, recall, f1 = calculate_metrics(
        true_positive,
        false_positive,
        false_negative
    )

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "no_signature_false_positive_pages":
            no_signature_false_positive_pages
    }

def main():

    print("Loading ground truth...")

    ground_truth = load_ground_truth()

    print(
        f"Evaluation images: "
        f"{len(ground_truth)}"
    )

    print(
        f"IoU threshold: "
        f"{IOU_THRESHOLD}"
    )

    detector_results = evaluate(
        ground_truth,
        use_verifier=False
    )

    final_results = evaluate(
        ground_truth,
        use_verifier=True
    )

    print("YOLO DETECTOR")

    print(
        f"True Positives: "
        f"{detector_results['true_positive']}"
    )

    print(
        f"False Positives: "
        f"{detector_results['false_positive']}"
    )

    print(
        f"False Negatives: "
        f"{detector_results['false_negative']}"
    )

    print(
        f"Precision: "
        f"{detector_results['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{detector_results['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{detector_results['f1']:.4f}"
    )

    print(
        f"True Positives: "
        f"{final_results['true_positive']}"
    )

    print(
        f"False Positives: "
        f"{final_results['false_positive']}"
    )

    print(
        f"False Negatives: "
        f"{final_results['false_negative']}"
    )

    print(
        f"Precision: "
        f"{final_results['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{final_results['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{final_results['f1']:.4f}"
    )

    # -------------------------
    # No-signature pages
    # -------------------------

    no_signature_pages = (
        final_results[
            "no_signature_false_positive_pages"
        ]
    )

    print(
        "\nNo-signature pages with "
        "false positives:"
    )

    print(len(no_signature_pages))

    print(no_signature_pages)

    results = {
        "iou_threshold": IOU_THRESHOLD,
        "detector": detector_results,
        "final_pipeline": final_results
    }

    output_file = Path(
        "outputs/evaluation_results.json"
    )

    with open(output_file, "w") as file:
        json.dump(
            results,
            file,
            indent=4
        )

    print(
        f"\nResults saved to: "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()