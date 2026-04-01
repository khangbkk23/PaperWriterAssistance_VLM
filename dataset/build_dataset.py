import os
import json
import random
import gc
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

MAX_IMAGE_SIZE = 768

ARXIV_TARGET = 16000
DOCVQA_TARGET = 4000
BUFFER_SIZE = 50

OUTPUT_DIR = "vlm_dataset"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
JSONL_PATH = os.path.join(OUTPUT_DIR, "train_20000.jsonl")

os.makedirs(IMG_DIR, exist_ok=True)

ARXIV_PROMPTS = [
    "Phân tích sơ đồ khoa học.",
    "Giải thích biểu đồ.",
    "Mô tả nội dung kỹ thuật.",
    "Quan sát hình và trả lời câu hỏi.",
]

DOCVQA_PROMPTS = [
    "Dựa vào tài liệu, hãy trả lời:",
    "Phân tích và trả lời:",
    "Đọc tài liệu và cho biết:",
]

ACADEMIC_SOURCES = [
    ("scienceqa",            2500),
    ("ai2d",                 2000),
    ("chartqa",              1500),
    ("tqa",                  1500),
    ("ocrvqa",               1500),
    ("textvqa",              1000),
    ("infographic_vqa",       500),
    ("figureqa",              500),
    ("dvqa",                  500),
    ("plotqa",                500),
    ("vistext",               500),
    ("diagram_image_to_text", 500),
    ("screen2words",          500),
    ("rendered_text",         500),
    ("websight",              500),
    ("datikz",                500),
]

def resize_and_save_image(image, save_path):
    try:
        if image is None:
            return None
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
    except Exception:
        return None

def append_batch(records):
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def process_academic_vision():
    print(f"Starting academic vision — target: {ARXIV_TARGET} samples")

    total_count = 0
    buffer = []
    progress_bar = tqdm(total=ARXIV_TARGET, desc="Academic Vision")

    for config_name, config_target in ACADEMIC_SOURCES:
        if total_count >= ARXIV_TARGET:
            break

        print(f"\n[{config_name}] target={config_target}...")

        try:
            dataset = load_dataset(
                "HuggingFaceM4/the_cauldron",
                name=config_name,
                split="train",
                streaming=True,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"  Skipping {config_name}: {e}")
            continue

        config_count = 0

        item_iter = iter(dataset)
        while total_count < ARXIV_TARGET and config_count < config_target:
            try:
                item = next(item_iter)
            except StopIteration:
                break
            except Exception as e:
                print(f"  [{config_name}] batch decode error, skipping: {type(e).__name__}")
                continue

            try:
                images = item.get("images", [])
                texts = item.get("texts", [])

                if not images or not texts:
                    continue

                img = images[0]
                qa_pair = texts[0]

                user_text = qa_pair.get("user", "").strip()
                assistant_text = qa_pair.get("assistant", "").strip()

                if not user_text or not assistant_text:
                    continue

                img_filename = f"academic_{total_count}.jpg"
                img_path = os.path.join(IMG_DIR, img_filename)

                saved_path = resize_and_save_image(img, img_path)
                if not saved_path:
                    continue

                record = {
                    "id": f"academic_{total_count}",
                    "source": config_name,
                    "image": saved_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"{random.choice(ARXIV_PROMPTS)} {user_text}"
                        },
                        {
                            "from": "gpt",
                            "value": assistant_text
                        }
                    ]
                }

                buffer.append(record)
                total_count += 1
                config_count += 1
                progress_bar.update(1)

                if len(buffer) >= BUFFER_SIZE:
                    append_batch(buffer)
                    buffer.clear()

                if total_count % 500 == 0:
                    gc.collect()

            except Exception:
                continue

        print(f"  [{config_name}] collected: {config_count}")
        del dataset
        gc.collect()

    if buffer:
        append_batch(buffer)

    progress_bar.close()
    print(f"\nAcademic vision done: {total_count} samples")

def process_docvqa():
    print(f"\nStreaming DocVQA — target: {DOCVQA_TARGET} samples")

    dataset = load_dataset(
        "HuggingFaceM4/DocumentVQA",
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    count = 0
    buffer = []
    progress = tqdm(total=DOCVQA_TARGET, desc="DocVQA")

    for item in dataset:
        if count >= DOCVQA_TARGET:
            break
        try:
            img = item["image"]
            question = item["question"]
            answers = item["answers"]
            answer = answers[0] if isinstance(answers, list) and answers else answers

            if not answer:
                continue

            img_path = os.path.join(IMG_DIR, f"docvqa_{count}.jpg")
            saved = resize_and_save_image(img, img_path)
            if not saved:
                continue

            record = {
                "id": f"docvqa_{count}",
                "source": "DocumentVQA",
                "image": saved,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"{random.choice(DOCVQA_PROMPTS)} {question}"
                    },
                    {
                        "from": "gpt",
                        "value": answer
                    }
                ]
            }

            buffer.append(record)
            count += 1
            progress.update(1)

            if len(buffer) >= BUFFER_SIZE:
                append_batch(buffer)
                buffer.clear()

            if count % 500 == 0:
                gc.collect()

        except Exception:
            continue

    if buffer:
        append_batch(buffer)

    progress.close()
    del dataset
    gc.collect()
    print(f"DocVQA done: {count} samples")

def shuffle_jsonl():
    print("\nShuffling dataset...")
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    random.shuffle(lines)
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Total samples: {len(lines)}")

if __name__ == "__main__":
    if os.path.exists(JSONL_PATH):
        os.remove(JSONL_PATH)

    process_academic_vision()
    process_docvqa()
    shuffle_jsonl()

    print(f"\nPipeline complete! Output: {OUTPUT_DIR}")