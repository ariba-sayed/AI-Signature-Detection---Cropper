from pathlib import Path
import json
from ollama import chat

CROP_DIR = Path("outputs/crops")
ACCEPTED_DIR = Path("outputs/accepted")
REJECTED_DIR = Path("outputs/rejected")
RESULTS_DIR = Path("outputs/verification")

ACCEPTED_DIR.mkdir(parents=True, exist_ok=True)
REJECTED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "qwen3-vl:2b"


def verify_signature(image_path):
    """
    Ask the local vision model whether the crop contains
    a handwritten signature.
    """
    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": """
Is this image a handwritten signature?

A handwritten signature means a person's handwritten name,
initials, or stylized handwritten mark used as a signature.

Answer with exactly one word:
YES
or
NO
""",
                "images": [str(image_path)],
            }
        ],
        options={"temperature": 0},
    )

    raw_response = response.message.content.strip()

    # Enforce the required binary schema.
    normalized = raw_response.upper().strip(" .,!")

    if normalized == "YES":
        decision = "YES"
    elif normalized == "NO":
        decision = "NO"
    else:
        decision = "NO"

    return raw_response, decision


def process_crop(image_path):
    print(f"Checking: {image_path.name}")

    try:
        raw_response, decision = verify_signature(image_path)

    except Exception as error:
        print(f"  Error: {error}")

        result = {
            "image": image_path.name,
            "decision": "NO",
            "raw_response": "",
            "reason": f"Verifier error: {error}",
        }

        save_result(image_path, result)
        return

    if decision == "YES":
        destination = ACCEPTED_DIR / image_path.name
        reason = "Verifier classified the crop as a handwritten signature."
    else:
        destination = REJECTED_DIR / image_path.name
        reason = "Verifier classified the crop as not a handwritten signature."

    # Copy the crop into accepted/rejected.
    import shutil
    shutil.copy2(image_path, destination)

    result = {
        "image": image_path.name,
        "decision": decision,
        "raw_response": raw_response,
        "reason": reason,
    }

    save_result(image_path, result)

    print(f"  Decision: {decision}")


def save_result(image_path, result):
    result_path = RESULTS_DIR / f"{image_path.stem}.json"

    with open(result_path, "w") as file:
        json.dump(result, file, indent=4)


def main():
    image_files = sorted(CROP_DIR.glob("*.png"))

    if not image_files:
        print("No crops found.")
        return

    for image_path in image_files:
        process_crop(image_path)

    print("\nVerification complete.")
   
if __name__ == "__main__":
    main()