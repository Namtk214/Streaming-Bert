"""
Training loop – Streaming Binary Scam Detection (Noisy-OR MIL).

Loss: BCE( Noisy-OR(p_1..p_T), y_dialogue )
PhoBERT được fine-tune end-to-end từ epoch đầu.
"""

import json
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


# ── Evaluation ──────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, device, threshold=0.5):
    model.eval()
    total_loss, n_dialogues = 0.0, 0
    all_dialogue_labels, all_dialogue_probs, all_turn_probs = [], [], []

    for batch in dataloader:
        batch  = {k: v.to(device) for k, v in batch.items()}
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            turn_mask=batch["turn_mask"],
            dialogue_labels=batch["dialogue_labels"],
        )
        B            = batch["dialogue_labels"].shape[0]
        total_loss   += output["loss"].item() * B
        n_dialogues  += B

        for i in range(B):
            vlen = int(batch["turn_mask"][i].sum())
            all_dialogue_labels.append(int(batch["dialogue_labels"][i].item()))
            all_dialogue_probs.append(float(output["dialogue_probs"][i].item()))
            all_turn_probs.append(output["turn_probs"][i, :vlen].cpu().numpy())

    avg_loss = total_loss / max(n_dialogues, 1)
    metrics  = compute_streaming_metrics(
        all_dialogue_labels, all_dialogue_probs, all_turn_probs, threshold
    )
    metrics["loss"] = avg_loss
    return metrics


# ── Sample preview ──────────────────────────────────────────────

@torch.no_grad()
def preview_sample(model, dataloader, device, threshold=0.5, max_samples=2):
    model.eval()
    dataset = dataloader.dataset
    if not dataset:
        return

    n_cand   = min(len(dataset), max(max_samples * 8, 8))
    cands    = random.sample(range(len(dataset)), n_cand)
    selected, seen = [], set()
    for idx in cands:
        lbl = dataset.dialogues[idx].get("conversation_label", "?")
        if lbl not in seen or len(selected) < max_samples:
            selected.append(idx)
            seen.add(lbl)
        if len(selected) >= max_samples and len(seen) >= 2:
            break
    selected = selected[:max_samples]

    import math as _math

    print("\n  -- Random Validation Preview --")
    for n, idx in enumerate(selected, 1):
        item  = dataset[idx]
        batch = streaming_collate_fn([item])
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            turn_mask=batch["turn_mask"],
        )
        vlen    = int(batch["turn_mask"][0].sum())
        d_prob  = float(output["dialogue_probs"][0].item())
        t_probs = output["turn_probs"][0, :vlen].cpu().tolist()

        dlg      = dataset.dialogues[idx]
        true_lbl = dlg.get("conversation_label", "?")
        turns    = dlg.get("turns", [])[:vlen]

        print(
            f"    Sample {n}: idx={idx} id={dlg.get('dialogue_id')} "
            f"true={true_lbl} p_dialogue={d_prob:.4f} ({vlen} turns)"
        )

        log_comp = 0.0
        for t, (q, turn) in enumerate(zip(t_probs, turns), 1):
            log_comp += _math.log(max(1.0 - q, 1e-7))
            p_agg     = 1.0 - _math.exp(log_comp)
            alert     = " [!]" if p_agg >= threshold else ""
            speaker   = turn.get("speaker", "")
            # map speaker id → label
            spk_label = {0: "người gọi", 1: "người nghe"}.get(speaker, str(speaker))
            text_prev = turn.get("text", "")[:50]
            print(
                f"      T{t:<2}: q={q:.4f}  p_agg={p_agg:.4f}{alert}"
                f" ({spk_label}: {text_prev}...)"
            )


# ── Training helpers ────────────────────────────────────────────

def _run_epoch(model, loader, optimizer, scheduler, device, cfg):
    """One training epoch. Returns average loss per dialogue."""
    model.train()
    total_loss, n_dlg = 0.0, 0

    for batch_idx, batch in enumerate(loader):
        batch  = {k: v.to(device) for k, v in batch.items()}
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            turn_mask=batch["turn_mask"],
            dialogue_labels=batch["dialogue_labels"],
        )
        loss = output["loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        B          = batch["dialogue_labels"].shape[0]
        total_loss += loss.item() * B
        n_dlg      += B

        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(loader):
            print(
                f"  [{batch_idx+1}/{len(loader)}] "
                f"loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}"
            )

    return total_loss / max(n_dlg, 1)


def _print_val_metrics(epoch, elapsed, train_loss, m):
    print(f"\n  Epoch {epoch} ({elapsed:.1f}s):")
    print(f"    Train loss:       {train_loss:.4f}")
    print(f"    Val loss:         {m['loss']:.4f}")
    print(f"    Dialogue acc:     {m['dialogue_accuracy']:.4f}")
    print(f"    Dialogue F1:      {m['dialogue_f1']:.4f}")
    if not np.isnan(m.get("auroc", float("nan"))):
        print(f"    AUROC:            {m['auroc']:.4f}")
    print(f"    Detection rate:   {m['detection_rate']:.4f}")
    if not np.isnan(m["avg_detection_delay"]):
        print(f"    Avg delay:        {m['avg_detection_delay']:.2f} turns")
    print(f"    False alarm rate: {m['false_alarm_rate']:.4f}")


def _save_best_model(model, tokenizer, cfg, val_metrics):
    save_path = os.path.join(cfg.output_dir, "best_model")
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
    import dataclasses
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
    tokenizer.save_pretrained(save_path)
    with open(os.path.join(save_path, "val_metrics.json"), "w") as f:
        json.dump(
            {k: float(v) if isinstance(v, (float, np.floating)) else v
             for k, v in val_metrics.items()}, f, indent=2
        )


# ── Training ────────────────────────────────────────────────────

def train(cfg: StreamingConfig = None):
    if cfg is None:
        cfg = StreamingConfig()

    set_seed(cfg.seed)

    print("=" * 60)
    print("STREAMING BINARY SCAM DETECTION – NOISY-OR MIL TRAINING")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Tokenizer ──
    print(f"\n  Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # ── Datasets ──
    train_path = os.path.join(cfg.streaming_data_dir, "train.json")
    val_path   = os.path.join(cfg.streaming_data_dir, "val.json")
    test_path  = os.path.join(cfg.streaming_data_dir, "test.json")

    for p in [train_path, val_path]:
        if not os.path.exists(p):
            print(f"  ERROR: {p} not found. Run prepare_data.py first.")
            return

    train_dataset = StreamingDialogueDataset(train_path, tokenizer, cfg.max_tokens_per_turn)
    val_dataset   = StreamingDialogueDataset(val_path,   tokenizer, cfg.max_tokens_per_turn)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=True, collate_fn=streaming_collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size,
        shuffle=False, collate_fn=streaming_collate_fn, num_workers=0,
    )

    print(f"  Train: {len(train_dataset)} dialogues ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_dataset)} dialogues ({len(val_loader)} batches)")

    # ── Model ──
    print(f"\n  Loading model: {cfg.model_name}")
    model = StreamingScamDetector(cfg).to(device)
    print(f"  Trainable params: {model.count_trainable_params():,}")

    # ── Optimizer + Scheduler ──
    total_steps  = len(train_loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    param_groups = model.get_param_groups(cfg.rnn_head_lr)
    optimizer = AdamW(
        param_groups, lr=cfg.rnn_head_lr,
        weight_decay=cfg.weight_decay, eps=cfg.adam_epsilon,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # ── Training loop ──
    best_f1, best_epoch = 0.0, 0
    patience, patience_counter = 3, 0
    os.makedirs(cfg.output_dir, exist_ok=True)

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        print(f"\n  Epoch {epoch}/{cfg.num_epochs}")

        avg_train_loss = _run_epoch(model, train_loader, optimizer, scheduler, device, cfg)
        val_metrics    = evaluate(model, val_loader, device, cfg.threshold)

        _print_val_metrics(epoch, time.time() - t0, avg_train_loss, val_metrics)
        preview_sample(model, val_loader, device, cfg.threshold)

        if val_metrics["f1"] > best_f1:
            best_f1, best_epoch = val_metrics["f1"], epoch
            patience_counter = 0
            _save_best_model(model, tokenizer, cfg, val_metrics)
            print(f"    * Best model saved (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"    No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # ── Final test evaluation ──
    sep = "=" * 60
    print(f"\n{sep}")
    print("TRAINING COMPLETED")
    print(f"  Best epoch: {best_epoch} | Best val F1: {best_f1:.4f}")
    print(sep)

    best_pt = os.path.join(cfg.output_dir, "best_model", "model.pt")
    if os.path.exists(best_pt):
        model.load_state_dict(torch.load(best_pt, map_location=device, weights_only=True))

    if os.path.exists(test_path):
        test_dataset = StreamingDialogueDataset(test_path, tokenizer, cfg.max_tokens_per_turn)
        test_loader  = DataLoader(
            test_dataset, batch_size=cfg.batch_size,
            shuffle=False, collate_fn=streaming_collate_fn, num_workers=0,
        )
        print(f"\n  Evaluating on test set ({len(test_dataset)} dialogues)...")
        test_metrics = evaluate(model, test_loader, device, cfg.threshold)
        print_streaming_report(test_metrics)

    return model


# ── CLI ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Streaming-Bert (Noisy-OR MIL, binary scam detection)."
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--debug", action="store_true",
                        help="2 epochs, batch_size=2")
    parser.add_argument("--small", action="store_true",
                        help="5 epochs, batch_size=2")
    return parser.parse_args()


if __name__ == "__main__":
    cfg  = StreamingConfig()
    args = parse_args()

    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.debug:
        cfg.num_epochs = 2
        cfg.batch_size = 2
        print("  DEBUG MODE: 2 epochs, batch_size=2")
    if args.small:
        cfg.num_epochs = 5
        cfg.batch_size = 2
        print("  SMALL MODE: 5 epochs, batch_size=2")

    train(cfg)
