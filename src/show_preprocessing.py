from pathlib import Path

import cv2
import matplotlib.pyplot as plt


INPUT_DIR = Path("data/evaluation")
PROCESSED_DIR = Path("data/processed")


# Choose one example
image_name = "eval_001.png"

original_path = INPUT_DIR / image_name
processed_path = PROCESSED_DIR / image_name


# Read images
original = cv2.imread(str(original_path))
processed = cv2.imread(str(processed_path))


# Convert original BGR → RGB for matplotlib
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

# Processed image is already grayscale
processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)


# Display
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

axes[0].imshow(original)
axes[0].set_title("Original")

axes[1].imshow(processed)
axes[1].set_title("Preprocessed")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    "preprocessing_before_after.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()