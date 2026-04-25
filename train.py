"""
Training loop cho Baseline2: Early-Exit with Noisy-OR Loss.

Chiến lược training:
  - Freeze PhoBERT hoàn toàn
  - Train CrossTurnAttention + Evidence Head (Linear 2d → 1)
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
from metrics import compute_noisy_or_metrics, print_noisy_or_report


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
    num_dialogues = 0

    all_p_dialogue = []   # list of float
    all_labels = []       # list of int
    all_turn_q = []       # list of list[float]
    all_turn_p_agg = []   # list of list[float]

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

        # Thu thập dialogue-level predictions và per-turn data
        for b in range(B):
            p_dlg = output["p_dialogue"][b].item()
            label = int(batch["labels"][b].item())
            all_p_dialogue.append(p_dlg)
            all_labels.append(label)

            # Per-turn data
            q_list = [q.item() for q in output["all_turn_q"][b]]
            p_agg_list = [p.item() for p in output["all_turn_p_agg"][b]]
            all_turn_q.append(q_list)
            all_turn_p_agg.append(p_agg_list)

    avg_loss = total_loss / max(num_dialogues, 1)
    metrics = compute_noisy_or_metrics(
        all_p_dialogue, all_labels,
        all_turn_q=all_turn_q,
        all_turn_p_agg=all_turn_p_agg,
        threshold=threshold,
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

    # Cố gắng chọn cả scam và harmless
    selected_indices = []
    selected_labels = set()
    for idx in candidate_indices:
        dlg_label = dataset.dialogues[idx].get("dialogue_label", -1)
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
        true_label_id = dialogue["dialogue_label"]
        true_label_name = dialogue.get("dialogue_label_name",
                                        LABEL_NAMES.get(true_label_id, "?"))
        N = item["num_turns"]
        p_dlg = output["p_dialogue"][0].item()

        print(f"    Sample {preview_idx}: idx={dataset_idx} "
              f"id={dialogue.get('dialogue_id')} "
              f"true={true_label_name} p_dialogue={p_dlg:.4f} ({N} turns)")

        # Show per-turn: q_t (evidence) and p_agg (cumulative)
        for t in range(len(output["all_turn_q"][0])):
            q_t = output["all_turn_q"][0][t].item()
            p_agg = output["all_turn_p_agg"][0][t].item()

            # Turn text preview
            turn_text = ""
            if t < len(dialogue["turns"]):
                turn = dialogue["turns"][t]
                role = turn.get("role", "?")
                text = turn.get("text", turn.get("text_segmented", ""))[:50]
                turn_text = f" ({role}: {text}...)"

            alert = " [!]" if p_agg >= 0.5 else ""
            print(f"      T{t+1}: q={q_t:.4f}  p_agg={p_agg:.4f}{alert}{turn_text}")


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
    print("EARLY-EXIT WITH NOISY-OR LOSS – TRAINING")
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

    # Label distribution
    train_labels = [d["dialogue_label"] for d in train_dataset.dialogues]
    print(f"  Train labels: harmless={train_labels.count(0)} scam={train_labels.count(1)}")

    # ── 3. Model ──
    print(f"\n  Loading model: {cfg.model_name}")
    model = EarlyExitWeightedModel(cfg).to(device)
    print(f"  Trainable params: {model.count_trainable_params():,}")
    print(f"  Freeze encoder:   {cfg.freeze_encoder}")
    print(f"  Evidence head:    Linear(2×{model.hidden_dim} → 1)")
    print(f"  Attention heads:  {cfg.attn_num_heads}")
    print(f"  Noisy-OR eps:     {cfg.eps}")

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
        print(f"    Train loss:    {avg_train_loss:.4f}")
        print(f"    Val loss:      {val_metrics['loss']:.4f}")
        print(f"    Val accuracy:  {val_metrics['accuracy']:.4f}")
        print(f"    Val F1:        {val_metrics['f1']:.4f}")
        print(f"    Val precision: {val_metrics['precision']:.4f}")
        print(f"    Val recall:    {val_metrics['recall']:.4f}")
        if not np.isnan(val_metrics.get('auroc', float('nan'))):
            print(f"    Val AUROC:     {val_metrics['auroc']:.4f}")
        if not np.isnan(val_metrics.get('avg_detection_delay', float('nan'))):
            print(f"    Avg 1st alert: turn {val_metrics['avg_first_alert_turn']:.1f}")
        print(f"    False alarm:   {val_metrics.get('false_alarm_rate', 0):.4f}")

        # Sample preview
        preview_sample(model, val_loader, device)

        # ── Best model & early stopping ──
        current_f1 = val_metrics["f1"]
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

            print(f"    * Best model saved! (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"    No improvement ({patience_counter}/{cfg.patience})")

        if patience_counter >= cfg.patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # ── 6. Final evaluation ──
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED")
    print(f"  Best epoch: {best_epoch} | Best val F1: {best_f1:.4f}")
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
        print_noisy_or_report(test_metrics)

    return model


# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Early-Exit with Noisy-OR Loss baseline."
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
