"""
Training loop cho Baseline2: Early-Exit with Weighted Loss.

Chiến lược training:
  - Freeze PhoBERT hoàn toàn
  - Train CrossTurnAttention + Linear classifier
  - AdamW + linear warmup + cosine decay
  - Gradient clipping
  - Validation mỗi epoch + early stopping
  - Sample prediction preview mỗi epoch
"""

import json
import os
import random
import sys
import time
import argparse
import dataclasses

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Fix import path
_baseline2_dir = os.path.dirname(os.path.abspath(__file__))
if _baseline2_dir not in sys.path:
    sys.path.insert(0, _baseline2_dir)

from config import EarlyExitConfig, LABEL_NAMES
from dataset import EarlyExitDataset, early_exit_collate_fn
from models.early_exit_model import EarlyExitWeightedModel
from metrics import compute_early_exit_metrics, print_early_exit_report


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
def evaluate(model, dataloader, device):
    """Evaluate model trên dataloader, trả về metrics dict."""
    model.eval()
    total_loss = 0.0
    num_dialogues = 0

    all_turn_preds = []   # list of list[int] per dialogue
    all_turn_labels = []  # list of list[int] per dialogue

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            turn_mask=batch["turn_mask"],
            labels=batch["labels"],
        )

        loss = output["loss"]
        B = batch["labels"].shape[0]
        total_loss += loss.item() * B
        num_dialogues += B

        # Thu thập per-turn predictions and labels
        for b in range(B):
            N = int(batch["turn_mask"][b].sum().item())
            preds = [
                int(lg.argmax().item()) for lg in output["all_turn_logits"][b]
            ]
            labels = batch["labels"][b, :N].cpu().tolist()
            all_turn_preds.append(preds)
            all_turn_labels.append(labels)

    avg_loss = total_loss / max(num_dialogues, 1)
    metrics = compute_early_exit_metrics(
        all_turn_preds, all_turn_labels, LABEL_NAMES,
    )
    metrics["loss"] = avg_loss

    return metrics


# ============================================================
# Sample prediction preview
# ============================================================
@torch.no_grad()
def preview_sample(model, dataloader, device, max_samples=2):
    """Random preview vài dialogue từ validation."""
    model.eval()
    dataset = dataloader.dataset
    if len(dataset) == 0:
        return

    num_candidates = min(len(dataset), max(max_samples * 8, 8))
    candidate_indices = random.sample(range(len(dataset)), num_candidates)

    # Cố gắng chọn các labels khác nhau
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
        batch = early_exit_collate_fn([item])
        batch = {k: v.to(device) for k, v in batch.items()}

        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            turn_mask=batch["turn_mask"],
        )

        dialogue = dataset.dialogues[dataset_idx]
        true_dlg_label = dialogue["conversation_label"]
        onset = dialogue.get("scam_onset")
        N = item["num_turns"]

        print(f"    Sample {preview_idx}: idx={dataset_idx} "
              f"id={dialogue.get('dialogue_id')} "
              f"dlg_label={true_dlg_label} onset={onset} ({N} turns)")

        # Show per-turn: true label vs predicted
        for t, logits_t in enumerate(output["all_turn_logits"][0]):
            pred_id = int(logits_t.argmax().item())
            pred_name = LABEL_NAMES.get(pred_id, "?")
            true_lbl = item["turn_labels"][t].item()
            true_name = LABEL_NAMES.get(true_lbl, "?")
            probs = torch.softmax(logits_t, dim=0).cpu().numpy()
            prob_str = " ".join(f"{p:.2f}" for p in probs)
            marker = "OK" if pred_id == true_lbl else "XX"
            print(f"      T{t+1}: true={true_name:10s} pred={pred_name:10s} "
                  f"[{prob_str}] {marker}")


# ============================================================
# Training
# ============================================================
def train(cfg: EarlyExitConfig = None):
    """Main training function."""
    if cfg is None:
        cfg = EarlyExitConfig()

    # Fix Windows console encoding
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    set_seed(cfg.seed)

    print("=" * 60)
    print("EARLY-EXIT WITH WEIGHTED LOSS – TRAINING")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── 1. Tokenizer ──
    print(f"\n  Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # ── 2. Datasets ──
    train_path = os.path.join(cfg.data_dir, "train.json")
    val_path = os.path.join(cfg.data_dir, "val.json")
    test_path = os.path.join(cfg.data_dir, "test.json")

    for p in [train_path, val_path]:
        if not os.path.exists(p):
            print(f"  ERROR: {p} not found. Run prepare_data.py first.")
            return

    train_dataset = EarlyExitDataset(train_path, tokenizer, cfg.max_tokens_per_turn)
    val_dataset = EarlyExitDataset(val_path, tokenizer, cfg.max_tokens_per_turn)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=True, collate_fn=early_exit_collate_fn,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size,
        shuffle=False, collate_fn=early_exit_collate_fn,
        num_workers=0, pin_memory=True,
    )

    print(f"  Train: {len(train_dataset)} dialogues ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_dataset)} dialogues ({len(val_loader)} batches)")

    # ── 3. Model ──
    print(f"\n  Loading model: {cfg.model_name}")
    model = EarlyExitWeightedModel(cfg).to(device)
    print(f"  Trainable params: {model.count_trainable_params():,}")
    print(f"  Freeze encoder:   {cfg.freeze_encoder}")
    print(f"  Num classes:      {cfg.num_classes}")
    print(f"  Attention heads:  {cfg.attn_num_heads}")

    # ── 4. Optimizer + Scheduler ──
    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    param_groups = model.get_param_groups(cfg.head_lr)
    optimizer = AdamW(
        param_groups,
        weight_decay=cfg.weight_decay,
        eps=cfg.adam_epsilon,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── 5. Training loop ──
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    os.makedirs(cfg.output_dir, exist_ok=True)

    for epoch in range(1, cfg.num_epochs + 1):
        epoch_start = time.time()

        # ── Train epoch ──
        model.train()
        epoch_loss = 0.0
        epoch_dialogues = 0

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

            B = batch["labels"].shape[0]
            epoch_loss += loss.item() * B
            epoch_dialogues += B

            # Log mỗi 5 batches
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
                lr_current = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                      f"loss={loss.item():.4f} lr={lr_current:.2e}")

        avg_train_loss = epoch_loss / max(epoch_dialogues, 1)

        # ── Validation ──
        val_metrics = evaluate(model, val_loader, device)

        elapsed = time.time() - epoch_start
        print(f"\n  Epoch {epoch}/{cfg.num_epochs} ({elapsed:.1f}s)")
        print(f"    Train loss:       {avg_train_loss:.4f}")
        print(f"    Val loss:         {val_metrics['loss']:.4f}")
        print(f"    Val final acc:    {val_metrics['final_accuracy']:.4f}")
        print(f"    Val macro F1:     {val_metrics['final_macro_f1']:.4f}")
        print(f"    Val weighted F1:  {val_metrics['final_weighted_f1']:.4f}")
        if not np.isnan(val_metrics['avg_detection_delay']):
            print(f"    Avg delay:        {val_metrics['avg_detection_delay']:.2f} turns")
        print(f"    False alarm:      {val_metrics['false_alarm_rate']:.4f}")

        # Sample preview
        preview_sample(model, val_loader, device)

        # ── Best model & early stopping ──
        current_f1 = val_metrics["final_macro_f1"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            save_path = os.path.join(cfg.output_dir, "best_model")
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))

            # Save config as JSON
            with open(os.path.join(save_path, "config.json"), "w") as f_cfg:
                json.dump(dataclasses.asdict(cfg), f_cfg, indent=2)
            tokenizer.save_pretrained(save_path)

            # Save metrics
            with open(os.path.join(save_path, "val_metrics.json"), "w") as f:
                json.dump(
                    {k: float(v) if isinstance(v, (float, np.floating)) else v
                     for k, v in val_metrics.items()},
                    f, indent=2,
                )

            print(f"    * Best model saved! (macro F1={best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"    No improvement ({patience_counter}/{cfg.patience})")

        if patience_counter >= cfg.patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # ── 6. Final evaluation ──
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED")
    print(f"  Best epoch: {best_epoch} | Best val macro F1: {best_f1:.4f}")
    print(f"{'='*60}")

    # Load best model
    best_path = os.path.join(cfg.output_dir, "best_model", "model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True)
        )
        print("  Loaded best model for final evaluation")

    # Test set
    if os.path.exists(test_path):
        test_dataset = EarlyExitDataset(test_path, tokenizer, cfg.max_tokens_per_turn)
        test_loader = DataLoader(
            test_dataset, batch_size=cfg.batch_size,
            shuffle=False, collate_fn=early_exit_collate_fn,
            num_workers=0,
        )
        print(f"\n  Evaluating on test set ({len(test_dataset)} dialogues)...")
        test_metrics = evaluate(model, test_loader, device)
        print_early_exit_report(test_metrics, LABEL_NAMES)

    return model


# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Early-Exit with Weighted Loss baseline."
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Folder for training outputs.",
    )
    parser.add_argument("--debug", action="store_true",
                        help="2 epochs, batch_size=2")
    parser.add_argument("--small", action="store_true",
                        help="5 epochs, batch_size=2")
    return parser.parse_args()


if __name__ == "__main__":
    cfg = EarlyExitConfig()
    args = parse_args()

    if args.output_dir:
        cfg.output_dir = args.output_dir

    if args.debug:
        cfg.num_epochs = 2
        cfg.batch_size = 2
        print("  DEBUG MODE: 2 epochs, batch_size=2")

    if args.small:
        cfg.batch_size = 2
        cfg.num_epochs = 5
        print("  SMALL MODE: batch_size=2, 5 epochs")

    train(cfg)
