from pathlib import Path
import cv2
import json


# Where the original evaluation images are stored
IMAGE_DIR = Path("data/evaluation")

# Where detect.py saved the candidate JSON files
DETECTION_DIR = Path("outputs/detections")

# Where cropped candidates will be saved
CROP_DIR = Path("outputs/crops")

CROP_DIR.mkdir(parents=True, exist_ok=True)


def crop_with_padding(image, bbox, padding=10):
    """
    Crop a bounding box from an image and add padding around it.

    Parameters
    ----------
    image : numpy.ndarray
        Original document image.

    bbox : list
        Bounding box in the format:
        [x1, y1, x2, y2]

    padding : int
        Number of pixels added around the candidate.

    Returns
    -------
    crop : numpy.ndarray
        Cropped image.
    """

    x1, y1, x2, y2 = bbox

    height, width = image.shape[:2]

    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)

    crop = image[y1:y2, x1:x2]

    return crop


def process_detection(json_path):

    # Read detection JSON
    with open(json_path, "r") as f:
        detection_data = json.load(f)

    image_name = detection_data["image"]

    image_path = IMAGE_DIR / image_name

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read {image_path}")
        return

    candidates = detection_data["candidates"]

    print(f"\nProcessing: {image_name}")
    print(f"Candidates: {len(candidates)}")

    for i, candidate in enumerate(candidates):

        bbox = candidate["bbox"]

        crop = crop_with_padding(
            image,
            bbox,
            padding=10
        )

        crop_name = (
            f"{Path(image_name).stem}"
            f"_candidate_{i + 1:03d}.png"
        )

        crop_path = CROP_DIR / crop_name

        cv2.imwrite(
            str(crop_path),
            crop
        )

        print(f"  Saved: {crop_name}")


# Find all detection JSON files
json_files = sorted(DETECTION_DIR.glob("*.json"))

print(f"Found {len(json_files)} detection files.")

for json_path in json_files:
    process_detection(json_path)

print("\nCropping complete.")
print(f"Crops saved to: {CROP_DIR}")