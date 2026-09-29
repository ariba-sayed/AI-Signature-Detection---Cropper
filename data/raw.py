from datasets import load_dataset

dataset= load_dataset("tech4humans/signature-detection")
print(dataset)
sample = dataset["train"][0]

print("Image ID:", sample["image_id"])
print("Width:", sample["width"])
print("Height:", sample["height"])
print("Objects:", sample["objects"])
print(sample["objects"].keys())
print(sample["objects"]["bbox"])
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.imshow(sample["image"])
plt.axis("off")
plt.show()

def count_signatures(split):
    zero = 0
    one = 0
    multiple = 0

    for item in dataset[split]:
        count = len(item["objects"]["bbox"])

        if count == 0:
            zero += 1
        elif count == 1:
            one += 1
        else:
            multiple += 1

    return zero, one, multiple


for split in ["train", "validation", "test"]:
    print(split, count_signatures(split))