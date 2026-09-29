from datasets import load_dataset
from pathlib import Path
import json
import random

DATASET_NAME = "tech4humans/signature-detection"

OUTPUT_DIR = Path("data/evaluation")
GROUND_TRUTH_PATH = Path("data/ground_truth.json")

RANDOM_SEED = 42

NUM_NO_SIGNATURE = 10
NUM_SINGLE_SIGNATURE = 15
NUM_MULTIPLE_SIGNATURES = 15


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
#loading dataset
dataset = load_dataset(DATASET_NAME)
test_set = dataset["test"]

print(f"Test images available: {len(test_set)}")

# Separate images by number of signatures
no_signature = []
single_signature = []
multiple_signatures = []

for index, item in enumerate(test_set):
    num_signatures = len(item["objects"]["bbox"])
    if num_signatures == 0:
        no_signature.append(index)
    elif num_signatures == 1:
        single_signature.append(index)
    else:
        multiple_signatures.append(index)


print("\nAvailable images:")
print(f"No signature:       {len(no_signature)}")
print(f"One signature:      {len(single_signature)}")
print(f"Multiple signatures:{len(multiple_signatures)}")

random.seed(RANDOM_SEED)
selected_no_signature = random.sample(
    no_signature,
    NUM_NO_SIGNATURE
)
selected_single_signature = random.sample(
    single_signature,
    NUM_SINGLE_SIGNATURE
)
selected_multiple_signatures = random.sample(
    multiple_signatures,
    NUM_MULTIPLE_SIGNATURES
)
selected_indices = (
    selected_no_signature
    + selected_single_signature
    + selected_multiple_signatures
)

# Shuffle the final evaluation set
random.shuffle(selected_indices)
print(f"\nSelected {len(selected_indices)} evaluation images.")

# Generate images + ground truth

ground_truth = {}
for eval_number, dataset_index in enumerate(selected_indices, start=1):

    item = test_set[dataset_index]

    image = item["image"]

    width = item["width"]
    height = item["height"]

    bboxes = item["objects"]["bbox"]

    # Create filename
    filename = f"eval_{eval_number:03d}.png"

    image_path = OUTPUT_DIR / filename

    # Save image
    image.save(image_path)

    # Convert [x, y, width, height]
    # to [x1, y1, x2, y2]

    signatures = []
    for bbox in bboxes:
        x, y, w, h = bbox
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        signatures.append({
            "bbox": [
                round(x1, 2),
                round(y1, 2),
                round(x2, 2),
                round(y2, 2)
            ]
        })

    # Determine category
    num_signatures = len(signatures)
    if num_signatures == 0:
        category = "no_signature"

    elif num_signatures == 1:
        category = "single_signature"
    else:
        category = "multiple_signatures"
    # Add entry to ground truth
    ground_truth[filename] = {
        "source_dataset": DATASET_NAME,
        "source_split": "test",
        "source_index": dataset_index,
        "width": width,
        "height": height,
        "signature_count": num_signatures,
        "category": category,
        "signatures": signatures
    }

# Save ground_truth.json
with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
    json.dump(
        ground_truth,
        f,
        indent=2
    )

print("\nEvaluation set created successfully!")
print("\nFinal distribution:")
print(f"  No signatures:        {len(selected_no_signature)}")
print(f"  Single signature:     {len(selected_single_signature)}")
print(f"  Multiple signatures:  {len(selected_multiple_signatures)}")
print(f"  Total:                {len(selected_indices)}")