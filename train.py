"""
Training loop cho Streaming Binary Scam Detection.

Staged training theo implementation guide:
  Stage A (epoch 1-3): Freeze PhoBERT hoàn toàn
  Stage B (epoch 4-6): Unfreeze top 2 PhoBERT layers
  Stage C (epoch 7+):  Unfreeze top 4 layers (nếu val F1 cải thiện)

Features:
  - Custom training loop (không dùng HuggingFace Trainer vì batching đặc biệt)
  - Param groups với LR khác nhau (encoder vs RNN/head)
  - AdamW + linear warmup + cosine decay
  - Gradient clipping
  - Validation mỗi epoch + early stopping
  - Sample prediction preview mỗi epoch
"""

import json
import math
import os
import random
import sys
import time
import argparse

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Đảm bảo import từ streaming/ thay vì src/
_streaming_dir = os.path.dirname(os.path.abspath(__file__))
if _streaming_dir not in sys.path:
    sys.path.insert(0, _streaming_dir)

from config import StreamingConfig
from dataset import StreamingDialogueDataset, streaming_collate_fn
from model import StreamingScamDetector
from metrics import compute_streaming_metrics, print_streaming_report


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate(model, dataloader, device, threshold=0.5):
    """Evaluate model trên dataloader, trả về metrics dict."""
    model.eval()
    total_loss = 0.0
    total_turns = 0

    all_labels = []
    all_logits = []
    all_masks = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            turn_mask=batch["turn_mask"],
            labels=batch["labels"],
        )

        loss = output["loss"]
        logits = output["logits"]
        mask = batch["turn_mask"]

        total_loss += loss.item() * mask.sum().item()
        total_turns += mask.sum().item()

        # Thu thập per-dialogue
        B = logits.shape[0]
        for i in range(B):
            all_labels.append(batch["labels"][i].cpu().numpy())
            all_logits.append(logits[i].cpu().numpy())
            all_masks.append(mask[i].cpu().numpy())

    avg_loss = total_loss / max(total_turns, 1)
    metrics = compute_streaming_metrics(all_labels, all_logits, all_masks, threshold)
    metrics["loss"] = avg_loss

    return metrics


# ============================================================
# Sample prediction preview
# ============================================================
@torch.no_grad()
def preview_sample(model, dataloader, device, threshold=0.5, max_samples=2):
    """Random preview vài dialogue từ validation để tránh luôn nhìn cùng 1 mẫu."""
    model.eval()
    dataset = dataloader.dataset
    if len(dataset) == 0:
        return

    # Lấy ngẫu nhiên nhiều candidates để tăng khả năng có cả LEGIT và SCAM.
    num_candidates = min(len(dataset), max(max_samples * 8, 8))
    candidate_indices = random.sample(range(len(dataset)), num_candidates)

    selected_indices = []
    selected_labels = set()
    for idx in candidate_indices:
        dlg_label = dataset.dialogues[idx].get("conversation_label", "UNKNOWN")
        if dlg_label not in selected_labels or len(selected_indices) < max_samples:
            selected_indices.append(idx)
            selected_labels.add(dlg_label)
        if len(selected_indices) >= max_samples and len(selected_labels) >= 2:
            break

    selected_indices = selected_indices[:max_samples]

    print(f"\n  -- Random Validation Preview --")
    for preview_idx, dataset_idx in enumerate(selected_indices, start=1):
        item = dataset[dataset_idx]
        batch = streaming_collate_fn([item])
        batch = {k: v.to(device) for k, v in batch.items()}

        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            turn_mask=batch["turn_mask"],
        )

        logits = output["logits"][0].cpu().numpy()
        labels = batch["labels"][0].cpu().numpy()
        mask = batch["turn_mask"][0].cpu().numpy()
        valid_len = int(mask.sum())

        probs = 1 / (1 + np.exp(-logits[:valid_len]))
        preds = (probs >= threshold).astype(int)
        true = labels[:valid_len].astype(int)

        dialogue = dataset.dialogues[dataset_idx]
        match = "[OK]" if np.array_equal(preds, true) else "[X]"
        print(
            f"    Sample {preview_idx}: idx={dataset_idx} "
            f"id={dialogue.get('dialogue_id')} "
            f"label={dialogue.get('conversation_label')} {match}"
        )
        print(f"      Turn:  {list(range(1, valid_len + 1))}")
        print(f"      True:  {true.tolist()}")
        print(f"      Pred:  {preds.tolist()}")
        print(f"      Probs: [{', '.join(f'{p:.2f}' for p in probs)}]")


# ============================================================
# Training
# ============================================================
def train(cfg: StreamingConfig = None):
    """Main training function."""
    if cfg is None:
        cfg = StreamingConfig()

    set_seed(cfg.seed)

    print("=" * 60)
    print("STREAMING BINARY SCAM DETECTION – TRAINING")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── 1. Tokenizer ──
    print(f"\n  Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # ── 2. Datasets ──
    train_path = os.path.join(cfg.streaming_data_dir, "train.json")
    val_path = os.path.join(cfg.streaming_data_dir, "val.json")
    test_path = os.path.join(cfg.streaming_data_dir, "test.json")

    for p in [train_path, val_path]:
        if not os.path.exists(p):
            print(f"  ERROR: {p} not found. Run prepare_streaming_data.py first.")
            return

    train_dataset = StreamingDialogueDataset(train_path, tokenizer, cfg.max_tokens_per_turn)
    val_dataset = StreamingDialogueDataset(val_path, tokenizer, cfg.max_tokens_per_turn)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=True, collate_fn=streaming_collate_fn,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size,
        shuffle=False, collate_fn=streaming_collate_fn,
        num_workers=0, pin_memory=True,
    )

    print(f"  Train: {len(train_dataset)} dialogues ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_dataset)} dialogues ({len(val_loader)} batches)")

    # ── 3. Model ──
    print(f"\n  Loading model: {cfg.model_name}")
    model = StreamingScamDetector(cfg).to(device)
    print(f"  Trainable params: {model.count_trainable_params():,}")

    # ── 4. Optimizer + Scheduler ──
    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    param_groups = model.get_param_groups(cfg.encoder_lr, cfg.rnn_head_lr)
    optimizer = AdamW(
        param_groups,
        weight_decay=cfg.weight_decay,
        eps=cfg.adam_epsilon,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # ── 5. Training loop ──
    best_f1 = 0.0
    best_epoch = 0
    patience = 3
    patience_counter = 0
    os.makedirs(cfg.output_dir, exist_ok=True)

    for epoch in range(1, cfg.num_epochs + 1):
        epoch_start = time.time()

        # ── Train epoch ──
        model.train()
        epoch_loss = 0.0
        epoch_turns = 0

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                turn_mask=batch["turn_mask"],
                labels=batch["labels"],
            )

            loss = output["loss"]

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optimizer.step()
            scheduler.step()

            mask_sum = batch["turn_mask"].sum().item()
            epoch_loss += loss.item() * mask_sum
            epoch_turns += mask_sum

            # Log mỗi 5 batches
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
                lr_current = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                      f"loss={loss.item():.4f} lr={lr_current:.2e}")

        avg_train_loss = epoch_loss / max(epoch_turns, 1)

        # ── Validation ──
        val_metrics = evaluate(model, val_loader, device, cfg.threshold)

        elapsed = time.time() - epoch_start
        print(f"\n  Epoch {epoch}/{cfg.num_epochs} ({elapsed:.1f}s)")
        print(f"    Train loss:  {avg_train_loss:.4f}")
        print(f"    Val loss:    {val_metrics['loss']:.4f}")
        print(f"    Val F1:      {val_metrics['f1']:.4f}")
        print(f"    Val AUROC:   {val_metrics['auroc']:.4f}")
        print(f"    Val Prec:    {val_metrics['precision']:.4f}")
        print(f"    Val Recall:  {val_metrics['recall']:.4f}")
        if not np.isnan(val_metrics['avg_detection_delay']):
            print(f"    Avg delay:   {val_metrics['avg_detection_delay']:.2f} turns")
        print(f"    False alarm: {val_metrics['false_alarm_rate']:.4f}")

        # Sample preview
        preview_sample(model, val_loader, device, cfg.threshold)

        # ── Best model & early stopping ──
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            save_path = os.path.join(cfg.output_dir, "best_model")
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
            # Save config as JSON (compatible with PyTorch 2.6+ weights_only default)
            import dataclasses
            with open(os.path.join(save_path, "config.json"), "w") as f_cfg:
                json.dump(dataclasses.asdict(cfg), f_cfg, indent=2)
            tokenizer.save_pretrained(save_path)

            # Save metrics
            with open(os.path.join(save_path, "val_metrics.json"), "w") as f:
                json.dump({k: float(v) if isinstance(v, (float, np.floating)) else v
                           for k, v in val_metrics.items()}, f, indent=2)

            print(f"    * Best model saved! (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"    No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience and epoch > cfg.stage_a_epochs:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # ── 6. Final evaluation trên test set ──
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED")
    print(f"  Best epoch: {best_epoch} | Best val F1: {best_f1:.4f}")
    print(f"{'='*60}")

    # Load best model
    best_path = os.path.join(cfg.output_dir, "best_model", "model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        print("  Loaded best model for final evaluation")

    # Test set
    if os.path.exists(test_path):
        test_dataset = StreamingDialogueDataset(test_path, tokenizer, cfg.max_tokens_per_turn)
        test_loader = DataLoader(
            test_dataset, batch_size=cfg.batch_size,
            shuffle=False, collate_fn=streaming_collate_fn,
            num_workers=0,
        )
        print(f"\n  Evaluating on test set ({len(test_dataset)} dialogues)...")
        test_metrics = evaluate(model, test_loader, device, cfg.threshold)
        print_streaming_report(test_metrics)

    return model


# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Streaming-Bert from the prepared dataset folder in config.py."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder for training outputs. Defaults to config output_dir.",
    )
    parser.add_argument("--debug", action="store_true", help="2 epochs, batch_size=2")
    parser.add_argument("--small", action="store_true", help="5 epochs, batch_size=2")
    return parser.parse_args()


if __name__ == "__main__":
    cfg = StreamingConfig()
    args = parse_args()

    if args.output_dir:
        cfg.output_dir = args.output_dir

    if args.debug:
        cfg.num_epochs = 2
        cfg.batch_size = 2
        cfg.stage_a_epochs = 1
        cfg.stage_b_epochs = 1
        print("  DEBUG MODE: 2 epochs, batch_size=2")

    if args.small:
        cfg.batch_size = 2
        cfg.num_epochs = 5
        print("  SMALL MODE: batch_size=2, 5 epochs")

    train(cfg)
