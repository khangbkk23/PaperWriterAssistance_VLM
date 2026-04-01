import os
import json
import random
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

MAX_IMAGE_SIZE = 768
ARXIV_TARGET = 16000
DOCVQA_TARGET = 4000

OUTPUT_DIR = "vlm_dataset"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
JSONL_PATH = os.path.join(OUTPUT_DIR, "train_20k.jsonl")

os.makedirs(IMG_DIR, exist_ok=True)

ARXIV_PROMPTS = [
    "Phân tích chi tiết sơ đồ khoa học trong hình ảnh.",
    "Giải thích biểu đồ này theo phong cách học thuật.",
    "Mô tả nội dung kỹ thuật của hình ảnh.",
    "Trình bày ý nghĩa khoa học của sơ đồ.",
]

DOCVQA_PROMPTS = [
    "Dựa vào tài liệu, hãy trả lời câu hỏi:",
    "Phân tích tài liệu và trả lời:",
    "Dựa trên nội dung hình ảnh, hãy giải quyết:",
]

def resize_and_save_image(image, save_path):
    if image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
    if max(w, h) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(w, h)
        image = image.resize(
            (int(w * scale), int(h * scale)),
            Image.Resampling.LANCZOS
        )

    image.save(save_path, "JPEG", quality=85)
    return os.path.abspath(save_path)


def append_to_jsonl(record, filepath):
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def process_arxiv():
    print(f"Processing {ARXIV_TARGET} ArXiv samples...")

    dataset = load_dataset(
        "MMInstruction/ArxivCap",
        split="train",
        streaming=True
    )

    count = 0
    progress_bar = tqdm(total=ARXIV_TARGET, desc="ArXiv")

    for item in dataset:
        if count >= ARXIV_TARGET:
            break

        try:
            img = item.get("image", None)
            caption = item.get("text", "")

            if img is None or not caption:
                continue

            if len(caption.split()) < 20:
                continue

            img_filename = f"arxiv_{count}.jpg"
            img_path = os.path.join(IMG_DIR, img_filename)

            img_path = resize_and_save_image(img, img_path)

            prompt = random.choice(ARXIV_PROMPTS)

            answer = f"Mô tả học thuật chi tiết: {caption}"

            record = {
                "id": f"arxiv_{count}",
                "image": img_path,
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": answer}
                ]
            }

            append_to_jsonl(record, JSONL_PATH)

            count += 1
            progress_bar.update(1)

        except Exception as e:
            continue

    progress_bar.close()


def process_docvqa():
    print(f"Processing {DOCVQA_TARGET} DocVQA samples...")

    dataset = load_dataset(
        "HuggingFaceM4/DocumentVQA",
        split="train",
        streaming=True
    )

    count = 0
    progress_bar = tqdm(total=DOCVQA_TARGET, desc="DocVQA")

    for item in dataset:
        if count >= DOCVQA_TARGET:
            break

        try:
            img = item["image"]
            question = item["question"]
            answers = item["answers"]

            if isinstance(answers, list) and len(answers) > 0:
                answer = answers[0]
            else:
                answer = answers

            if not answer:
                continue

            img_filename = f"docvqa_{count}.jpg"
            img_path = os.path.join(IMG_DIR, img_filename)

            img_path = resize_and_save_image(img, img_path)

            prompt = random.choice(DOCVQA_PROMPTS)

            final_answer = f"Dựa trên tài liệu, câu trả lời là: {answer}. Giải thích ngắn gọn nếu cần."

            record = {
                "id": f"docvqa_{count}",
                "image": img_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"{prompt} {question}"
                    },
                    {
                        "from": "gpt",
                        "value": final_answer
                    }
                ]
            }

            append_to_jsonl(record, JSONL_PATH)

            count += 1
            progress_bar.update(1)

        except Exception:
            continue

    progress_bar.close()

def shuffle_jsonl(filepath):
    print("Shuffling dataset...")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    random.shuffle(lines)

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)

if __name__ == "__main__":
    print("Starting pipeline...")

    if os.path.exists(JSONL_PATH):
        os.remove(JSONL_PATH)

    process_arxiv()
    process_docvqa()
    shuffle_jsonl(JSONL_PATH)

    print(f"Done! Dataset saved at: {OUTPUT_DIR}")
    print(f"SONL file: {JSONL_PATH}")