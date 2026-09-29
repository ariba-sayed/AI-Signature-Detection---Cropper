from pathlib import Path

import cv2
import numpy as np

INPUT_DIR = Path("data/evaluation")
OUTPUT_DIR = Path("data/processed")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def convert_to_grayscale(image):
    """Convert an image to greyscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def deskew(image):
    """
    Correct small rotations/skew in a document.
    """

    # Find dark pixels
    coords = np.column_stack(np.where(image < 255))

    # Not enough pixels to estimate skew
    if len(coords) < 10:
        return image

    # OpenCV expects coordinates as (x, y)
    coords = coords[:, ::-1]

    # Calculate minimum-area rectangle
    angle = cv2.minAreaRect(coords)[-1]

    # Convert OpenCV angle to rotation angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Ignore extremely small rotations
    if abs(angle) < 0.5:
        return image

    height, width = image.shape

    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated

def threshold_image(image):
    """
    Convert grayscale image into a binary image
    using Otsu's thresholding.
    """

    _, thresholded = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return thresholded

def preprocess(image):
    """
    Run the complete preprocessing pipeline.
    """
    gray = convert_to_grayscale(image)
    deskewed = deskew(gray)
    thresholded = threshold_image(deskewed)
    return gray, deskewed, thresholded



image_files = sorted(INPUT_DIR.glob("*.png"))

for image_path in image_files:
    print(f"Processing: {image_path.name}")
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read {image_path}")
        continue
    # Run preprocessing
    gray, deskewed, thresholded = preprocess(image)
    # Save final processed image
    output_path = OUTPUT_DIR / image_path.name
    cv2.imwrite(str(output_path),thresholded)

print("\nPreprocessing complete.")
