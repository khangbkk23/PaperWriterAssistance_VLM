import os
import sys
import json
import random
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import load_config


KNOWN_SCHEMAS = {
    "llava":    ({"id", "image", "conversations"}, "from",  "value"),
    "sharegpt": ({"id", "image", "conversations"}, "role",  "content"),
    "simple":   ({"image", "question", "answer"},  None,    None),
}


def detect_schema(sample):
    keys = set(sample.keys())
    for name, (required, *_) in KNOWN_SCHEMAS.items():
        if required.issubset(keys):
            return name
    return "unknown"


def normalize_to_llava(sample, schema):
    if schema == "llava":
        convs = sample.get("conversations", [])
        normalized = []
        for turn in convs:
            role = turn.get("from", "").lower()
            val  = turn.get("value", "").strip()
            if role not in ("human", "gpt") or not val:
                return None
            normalized.append({"from": role, "value": val})
        if len(normalized) < 2:
            return None
        return {
            "id":            sample.get("id", ""),
            "image":         sample["image"],
            "source":        sample.get("source", "unknown"),
            "conversations": normalized,
        }

    elif schema == "sharegpt":
        convs = sample.get("conversations", [])
        role_map = {"user": "human", "assistant": "gpt", "human": "human", "gpt": "gpt"}
        normalized = []
        for turn in convs:
            role = role_map.get(turn.get("role", "").lower())
            val  = turn.get("content", "").strip()
            if not role or not val:
                return None
            normalized.append({"from": role, "value": val})
        if len(normalized) < 2:
            return None
        return {
            "id":            sample.get("id", ""),
            "image":         sample["image"],
            "source":        sample.get("source", "unknown"),
            "conversations": normalized,
        }

    elif schema == "simple":
        q = sample.get("question", "").strip()
        a = sample.get("answer",   "").strip()
        if not q or not a:
            return None
        return {
            "id":    sample.get("id", hashlib.md5(q.encode()).hexdigest()[:8]),
            "image": sample["image"],
            "source": sample.get("source", "unknown"),
            "conversations": [
                {"from": "human", "value": q},
                {"from": "gpt",   "value": a},
            ],
        }

    return None


def build_chat_messages(sample):
    messages = []
    first_human = True
    for turn in sample["conversations"]:
        if turn["from"] == "human":
            content = []
            if first_human:
                content.append({"type": "image"})
                first_human = False
            content.append({"type": "text", "text": turn["value"]})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": turn["value"]}]
            })
    return messages


def validate_image(image_path):
    if not image_path or not os.path.exists(image_path):
        return False, "missing"
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img.load()
        return True, ""
    except Exception as e:
        return False, str(e)


def validate_images_parallel(samples, num_workers):
    valid, invalid = [], []
    paths = [s["image"] for s in samples]
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(validate_image, p): i for i, p in enumerate(paths)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Validating images"):
            idx = futures[fut]
            ok, reason = fut.result()
            if ok:
                valid.append(samples[idx])
            else:
                invalid.append({**samples[idx], "_error": reason})
    return valid, invalid


def stratified_split(samples, val_ratio, seed=42):
    random.seed(seed)
    by_source = defaultdict(list)
    for s in samples:
        by_source[s.get("source", "unknown")].append(s)
    train_all, val_all = [], []
    for _, items in by_source.items():
        random.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val_all.extend(items[:n_val])
        train_all.extend(items[n_val:])
    random.shuffle(train_all)
    random.shuffle(val_all)
    return train_all, val_all


def measure_token_lengths(samples, tokenizer):
    lengths = []
    for s in tqdm(samples, desc="Measuring token lengths"):
        text = " ".join(t["value"] for t in s["conversations"])
        ids  = tokenizer(text, add_special_tokens=False)["input_ids"]
        lengths.append(len(ids))
    return lengths


def bucket_histogram(lengths):
    buckets = {"0-128": 0, "129-256": 0, "257-512": 0,
               "513-768": 0, "769-1024": 0, ">1024": 0}
    for l in lengths:
        if   l <= 128:  buckets["0-128"]    += 1
        elif l <= 256:  buckets["129-256"]  += 1
        elif l <= 512:  buckets["257-512"]  += 1
        elif l <= 768:  buckets["513-768"]  += 1
        elif l <= 1024: buckets["769-1024"] += 1
        else:           buckets[">1024"]    += 1
    return buckets


def compute_stats(samples, tokenizer, max_tokens):
    stats = {
        "total": len(samples),
        "source_distribution": dict(Counter(s.get("source", "unknown") for s in samples)),
    }
    if tokenizer:
        lengths = measure_token_lengths(samples, tokenizer)
        too_long = sum(1 for l in lengths if l > max_tokens)
        stats["token_lengths"] = {
            "min":                 min(lengths),
            "max":                 max(lengths),
            "mean":                round(sum(lengths) / len(lengths), 1),
            "median":              sorted(lengths)[len(lengths) // 2],
            "histogram":           bucket_histogram(lengths),
            "over_max_tokens":     too_long,
            "over_max_tokens_pct": round(too_long / len(lengths) * 100, 2),
        }
    else:
        stats["token_lengths"] = "transformers not installed — skipped"
    return stats


def write_jsonl(samples, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f" Saved {len(samples):,} records → {path}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()

    cfg = load_config()

    input_path  = cfg.data.raw_jsonl
    out_dir     = Path(cfg.data.processed_dir)
    val_ratio   = cfg.data.val_ratio
    num_workers = cfg.data.num_workers
    seed        = cfg.data.seed
    model_id    = cfg.model.model_id
    max_tokens  = cfg.model.max_length

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[1/5] Loading {input_path} ...")
    raw = []
    schema_counts = Counter()
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"  Loaded {len(raw):,} raw records")

    print(f"\n[2/5] Detecting schema & normalizing ...")
    normalized, skipped_norm = [], []
    for sample in tqdm(raw, desc="Normalizing"):
        schema = detect_schema(sample)
        schema_counts[schema] += 1
        norm = normalize_to_llava(sample, schema)
        if norm is None:
            skipped_norm.append(sample)
        else:
            norm["chat_messages"] = build_chat_messages(norm)
            normalized.append(norm)

    print(f"  Schema breakdown : {dict(schema_counts)}")
    print(f"  Normalized       : {len(normalized):,}  |  Skipped: {len(skipped_norm):,}")
    if skipped_norm:
        write_jsonl(skipped_norm, str(out_dir / "skipped_bad_format.jsonl"))

    print(f"\n[3/5] Validating images ({num_workers} workers) ...")
    valid, invalid = validate_images_parallel(normalized, num_workers)
    print(f"  Valid: {len(valid):,}  |  Invalid/missing: {len(invalid):,}")
    if invalid:
        write_jsonl(invalid, str(out_dir / "skipped_bad_images.jsonl"))

    print(f"\n[4/5] Splitting train/val (val_ratio={val_ratio}) ...")
    train, val = stratified_split(valid, val_ratio, seed=seed)
    print(f"  Train: {len(train):,}  |  Val: {len(val):,}")
    write_jsonl(train, str(out_dir / "train.jsonl"))
    write_jsonl(val,   str(out_dir / "val.jsonl"))

    print(f"\n[5/5] Computing statistics ...")
    tokenizer = None
    if HAS_TRANSFORMERS:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            print(f"  Tokenizer: {model_id}")
        except Exception as e:
            print(f"  Tokenizer load failed ({e}) — skipping token stats")

    train_stats = compute_stats(train, tokenizer, max_tokens)
    val_stats   = compute_stats(val,   tokenizer, max_tokens)

    stats = {
        "input_file":             input_path,
        "raw_records":            len(raw),
        "after_normalize":        len(normalized),
        "after_image_validation": len(valid),
        "train":                  train_stats,
        "val":                    val_stats,
        "skipped_bad_format":     len(skipped_norm),
        "skipped_bad_images":     len(invalid),
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f" Raw            : {len(raw):,}")
    print(f" After normalize: {len(normalized):,}")
    print(f" After img valid: {len(valid):,}")
    print(f" Train          : {len(train):,}")
    print(f" Val            : {len(val):,}")
    print(f" Output dir     : {out_dir.resolve()}")
    if tokenizer and isinstance(train_stats.get("token_lengths"), dict):
        tl = train_stats["token_lengths"]
        print(f"\n  Token length (train):")
        print(f"    min={tl['min']}  max={tl['max']}  mean={tl['mean']}  median={tl['median']}")
        print(f"    Over {max_tokens} tokens: {tl['over_max_tokens']} ({tl['over_max_tokens_pct']}%)")
        print(f"    Histogram: {tl['histogram']}")

if __name__ == "__main__":
    main()