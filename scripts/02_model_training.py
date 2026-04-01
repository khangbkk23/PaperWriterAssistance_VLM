import os
import sys
import json
import shutil
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import (
    get_peft_model,
    LoraConfig,
    prepare_model_for_kbit_training,
    PeftModel,
)
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import load_config

class EarlyStopping:
    def __init__(self, patience, min_delta=0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = float("inf")
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            print(f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"  [EarlyStopping] Triggered — stopping training")
        return self.should_stop

class VLMDataset(Dataset):
    def __init__(self, jsonl_path, processor, max_tokens):
        self.processor  = processor
        self.max_tokens = max_tokens
        self.samples    = []
        print(f"Loading dataset: {jsonl_path}")
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.samples.append(json.loads(line))
                    except Exception:
                        continue
        print(f"  Loaded {len(self.samples):,} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            image = Image.open(sample["image"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=(255, 255, 255))
        messages = sample.get("chat_messages") or self._build_messages(sample)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"image": image, "text": text}

    @staticmethod
    def _build_messages(sample):
        messages = []
        first = True
        for turn in sample.get("conversations", []):
            if turn["from"] == "human":
                content = []
                if first:
                    content.append({"type": "image"})
                    first = False
                content.append({"type": "text", "text": turn["value"]})
                messages.append({"role": "user", "content": content})
            else:
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": turn["value"]}]
                })
        return messages

def collate_fn(batch, processor, max_tokens):
    images = [item["image"] for item in batch]
    texts  = [item["text"]  for item in batch]
    inputs = processor(
        text=texts, images=images,
        return_tensors="pt", padding=True,
        truncation=True, max_length=max_tokens,
    )
    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    if hasattr(processor, "image_token_id"):
        labels[labels == processor.image_token_id] = -100
    inputs["labels"] = labels
    return inputs

def load_model_and_processor(cfg, resume_from=None):
    print("\n[Model] Loading processor...")
    processor = AutoProcessor.from_pretrained(
        cfg.model.model_id,
        trust_remote_code=True,
        min_pixels=cfg.model.min_pixels,
        max_pixels=cfg.model.max_pixels,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("[Model] Loading base model (4-bit QLoRA)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg.training.gradient_checkpointing,
    )
    model.enable_input_require_grads()

    if resume_from and os.path.isdir(resume_from):
        print(f"[Model] Resuming LoRA from: {resume_from}")
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
    else:
        print("[Model] Injecting fresh LoRA...")
        lora_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            target_modules=list(cfg.lora.target_modules),
            lora_dropout=cfg.lora.lora_dropout,
            bias=cfg.lora.bias,
            task_type=cfg.lora.task_type,
            modules_to_save=list(cfg.lora.modules_to_save),
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    return processor, model

def save_checkpoint(model, processor, output_dir, tag):
    path = os.path.join(output_dir, tag)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    print(f"  [Checkpoint] Saved → {path}")
    return path

def save_best(model, processor, output_dir, val_loss, best_loss, best_path):
    if val_loss < best_loss:
        if best_path[0] and os.path.isdir(best_path[0]):
            shutil.rmtree(best_path[0], ignore_errors=True)
        path = save_checkpoint(model, processor, output_dir, "best")
        best_path[0] = path
        print(f"  [Best] val_loss {best_loss:.4f} → {val_loss:.4f}")
        return val_loss
    return best_loss

def find_latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    checkpoints = sorted(
        [d for d in Path(output_dir).iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1])
    )
    return str(checkpoints[-1]) if checkpoints else None

def load_training_state(output_dir):
    path = os.path.join(output_dir, "training_state.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_training_state(output_dir, state):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_state.json"), "w") as f:
        json.dump(state, f, indent=2)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss, total_steps = 0.0, 0
    for batch in tqdm(dataloader, desc="  Eval", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        if outputs.loss is not None:
            total_loss  += outputs.loss.item()
            total_steps += 1
    model.train()
    return total_loss / max(total_steps, 1)


def train(cfg):
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = cfg.training.output_dir
    os.makedirs(output_dir, exist_ok=True)

    resume_from = getattr(cfg.training, "resume_from", None)
    if getattr(cfg.training, "auto_resume", False) and not resume_from:
        resume_from = find_latest_checkpoint(output_dir)
        if resume_from:
            print(f"[Resume] Auto-detected: {resume_from}")

    training_state = load_training_state(output_dir) if resume_from else {}
    start_epoch   = training_state.get("epoch", 0)
    global_step   = training_state.get("global_step", 0)
    best_val_loss = training_state.get("best_val_loss", float("inf"))
    best_path     = [training_state.get("best_path", None)]

    patience   = getattr(cfg.training, "early_stopping_patience", 5)
    min_delta  = getattr(cfg.training, "early_stopping_min_delta", 0.001)
    early_stop = EarlyStopping(patience=patience, min_delta=min_delta)
    if resume_from:
        early_stop.counter   = training_state.get("es_counter", 0)
        early_stop.best_loss = training_state.get("es_best_loss", float("inf"))

    use_wandb = HAS_WANDB and getattr(cfg.wandb, "enabled", False)
    if use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name or None,
            config={
                "model_id":              cfg.model.model_id,
                "epochs":                cfg.training.num_train_epochs,
                "batch_size":            cfg.training.per_device_train_batch_size,
                "grad_accum":            cfg.training.gradient_accumulation_steps,
                "lr":                    cfg.training.learning_rate,
                "lora_r":                cfg.lora.r,
                "lora_alpha":            cfg.lora.lora_alpha,
                "early_stopping_patience": patience,
            },
            resume="allow" if resume_from else None,
        )
        print("[WandB] Initialized")
    elif not HAS_WANDB:
        print("[WandB] Not installed")

    processor, model = load_model_and_processor(cfg, resume_from)
    model.train()

    max_tokens = cfg.model.max_length
    _collate   = lambda b: collate_fn(b, processor, max_tokens)

    train_ds = VLMDataset(cfg.data.train_jsonl, processor, max_tokens)
    val_ds   = VLMDataset(cfg.data.val_jsonl,   processor, max_tokens)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.per_device_train_batch_size,
        shuffle=True, collate_fn=_collate, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1,
        shuffle=False, collate_fn=_collate, num_workers=2, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    grad_accum   = cfg.training.gradient_accumulation_steps
    total_steps  = (len(train_loader) // grad_accum) * cfg.training.num_train_epochs
    warmup_steps = max(1, int(total_steps * cfg.training.warmup_ratio))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"Model          : {cfg.model.model_id}")
    print(f"Max epochs     : {cfg.training.num_train_epochs}")
    print(f"Early stopping : patience={patience}, min_delta={min_delta}")
    print(f"Train          : {len(train_ds):,}  |  Val: {len(val_ds):,}")
    print(f"Batch          : {cfg.training.per_device_train_batch_size}  x  grad_accum {grad_accum}")
    print(f"Total steps    : {total_steps}  |  Warmup: {warmup_steps}")
    print(f"Start epoch    : {start_epoch}")

    stopped_early = False

    for epoch in range(start_epoch, cfg.training.num_train_epochs):
        print(f"\n─── Epoch {epoch+1}/{cfg.training.num_train_epochs} ───")
        epoch_loss, epoch_steps = 0.0, 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss    = outputs.loss / grad_accum

            loss.backward()
            epoch_loss  += outputs.loss.item()
            epoch_steps += 1

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = epoch_loss / epoch_steps
                lr_now   = scheduler.get_last_lr()[0]
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    lr=f"{lr_now:.2e}",
                    es=f"{early_stop.counter}/{patience}"
                )

                if use_wandb and global_step % cfg.training.logging_steps == 0:
                    wandb.log({
                        "train/loss":  avg_loss,
                        "train/lr":    lr_now,
                        "train/step":  global_step,
                        "train/epoch": epoch + (step / len(train_loader)),
                    }, step=global_step)

                if global_step % cfg.training.save_steps == 0:
                    save_checkpoint(model, processor, output_dir, f"checkpoint-{global_step}")
                    save_training_state(output_dir, {
                        "epoch": epoch, "global_step": global_step,
                        "best_val_loss": best_val_loss, "best_path": best_path[0],
                        "es_counter": early_stop.counter,
                        "es_best_loss": early_stop.best_loss,
                    })

        avg_train_loss = epoch_loss / max(epoch_steps, 1)
        print(f"\n  Train loss : {avg_train_loss:.4f}")
        print(f"  Running validation...")
        val_loss = evaluate(model, val_loader, device)
        print(f"  Val   loss : {val_loss:.4f}")

        if use_wandb:
            wandb.log({
                "epoch/train_loss":  avg_train_loss,
                "epoch/val_loss":    val_loss,
                "epoch/epoch":       epoch + 1,
                "epoch/es_counter":  early_stop.counter,
            }, step=global_step)

        best_val_loss = save_best(model, processor, output_dir, val_loss, best_val_loss, best_path)
        save_checkpoint(model, processor, output_dir, f"epoch-{epoch+1}")
        save_training_state(output_dir, {
            "epoch": epoch + 1, "global_step": global_step,
            "best_val_loss": best_val_loss, "best_path": best_path[0],
            "es_counter": early_stop.counter,
            "es_best_loss": early_stop.best_loss,
        })

        if early_stop.step(val_loss):
            stopped_early = True
            break

    print("\n[Post-train] Merging LoRA into base model...")
    merged_dir = os.path.join(output_dir, "merged")
    try:
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        processor.save_pretrained(merged_dir)
        print(f"Merged model saved → {merged_dir}")
    except Exception as e:
        print(f"Merge failed: {e}")
        print(f"Load adapter từ: {best_path[0] or output_dir}")

    if use_wandb:
        wandb.finish()

    if stopped_early:
        print(f"  Stopped early at epoch {epoch+1} (patience={patience})")
    print(f"Best val loss  : {best_val_loss:.4f}")
    print(f"Best checkpoint: {best_path[0]}")
    print(f"Merged model   : {merged_dir}")
    print(f"{'='*60}\n")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()