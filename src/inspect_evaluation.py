from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
from PIL import Image


# Paths
EVALUATION_DIR = Path("data/evaluation")
GROUND_TRUTH_PATH = Path("data/ground_truth.json")


# Load ground truth
with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
    ground_truth = json.load(f)


# Get evaluation images
image_files = sorted(EVALUATION_DIR.glob("*.png"))

print(f"Found {len(image_files)} evaluation images.")


# Contact sheet settings
cols = 5
rows = math.ceil(len(image_files) / cols)

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(15, 3 * rows)
)

axes = axes.flatten()


for ax, image_path in zip(axes, image_files):

    image = Image.open(image_path)

    ax.imshow(image)

    info = ground_truth[image_path.name]

    count = info["signature_count"]
    category = info["category"]

    ax.set_title(
        f"{image_path.name}\n"
        f"{category} | {count} signature(s)",
        fontsize=9
    )

    ax.axis("off")


# Hide unused axes
for ax in axes[len(image_files):]:
    ax.axis("off")


plt.tight_layout()

plt.savefig(
    "evaluation_contact_sheet.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()