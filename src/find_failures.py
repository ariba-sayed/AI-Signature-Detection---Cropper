from pathlib import Path
import json

GROUND_TRUTH_FILE = Path("data/ground_truth.json")
DETECTION_DIR = Path("outputs/detections")
VERIFICATION_DIR = Path("outputs/verification")


def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


ground_truth = load_json(GROUND_TRUTH_FILE)

print("\n=== NO-SIGNATURE PAGES WITH DETECTIONS ===")

for image_name, data in ground_truth.items():

    if data["signature_count"] != 0:
        continue

    detection_file = (
        DETECTION_DIR /
        f"{Path(image_name).stem}.json"
    )

    if not detection_file.exists():
        continue

    detections = load_json(detection_file)["candidates"]

    if len(detections) > 0:
        print(
            f"{image_name}: "
            f"{len(detections)} YOLO detection(s)"
        )


print("\n=== VERIFIER REJECTED CANDIDATES ===")

for verification_file in sorted(
    VERIFICATION_DIR.glob("*.json")
):

    result = load_json(verification_file)

    if result.get("decision") == "NO":
        print(
            f"{result['image']} | "
            f"Verifier: NO"
        )