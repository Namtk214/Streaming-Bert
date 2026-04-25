"""
Test Early-Exit Noisy-OR model trên dữ liệu đã convert từ Excel.

Đọc file JSON (output của convert_excel.py), chạy evaluate()
qua DataLoader, báo cáo đầy đủ metrics dialogue-level + streaming.

Usage:
    python test.py
    python test.py --data data/excel_test.json --threshold 0.5
    python test.py --verbose   # in turn probs từng sample
"""

import argparse
import json
import math
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

_streaming_dir = os.path.dirname(os.path.abspath(__file__))
if _streaming_dir not in sys.path:
    sys.path.insert(0, _streaming_dir)

from config import EarlyExitConfig, LABEL_NAMES
from dataset import EarlyExitDataset, early_exit_collate_fn
from models.early_exit_model import EarlyExitWeightedModel
from metrics import compute_noisy_or_metrics, print_noisy_or_report


# ── Evaluate ─────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, device, threshold):
    """Evaluate model, return metrics + per-dialogue data."""
    model.eval()
    total_loss, n_dlg = 0.0, 0

    all_labels = []       # list of int
    all_p_dialogue = []   # list of float
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

        B = batch["labels"].shape[0]
        if output["loss"] is not None:
            total_loss += output["loss"].item() * B
        n_dlg += B

        for i in range(B):
            all_labels.append(int(batch["labels"][i].item()))
            all_p_dialogue.append(float(output["p_dialogue"][i].item()))

            # Per-turn evidence and cumulative probabilities
            q_list = [q.item() for q in output["all_turn_q"][i]]
            p_agg_list = [p.item() for p in output["all_turn_p_agg"][i]]
            all_turn_q.append(q_list)
            all_turn_p_agg.append(p_agg_list)

    metrics = compute_noisy_or_metrics(
        all_p_dialogue, all_labels,
        all_turn_q=all_turn_q,
        all_turn_p_agg=all_turn_p_agg,
        threshold=threshold,
    )
    metrics["loss"] = total_loss / max(n_dlg, 1)

    return metrics, all_labels, all_p_dialogue, all_turn_q, all_turn_p_agg


# ── Per-dialogue detail ───────────────────────────────────────────

@torch.no_grad()
def print_verbose(model, dataset, device, threshold, max_print=20):
    """In chi tiết per-turn evidence + cumulative cho từng dialogue."""
    model.eval()
    print(f"\n  -- Per-dialogue detail (first {max_print}) --")

    for idx in range(min(len(dataset), max_print)):
        item = dataset[idx]
        dlg = dataset.dialogues[idx]

        batch = early_exit_collate_fn([item])
        batch = {k: v.to(device) for k, v in batch.items()}

        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            turn_mask=batch["turn_mask"],
        )

        N = item["num_turns"]
        p_dlg = float(output["p_dialogue"][0].item())
        true_label_name = dlg.get("dialogue_label_name", LABEL_NAMES.get(dlg.get("dialogue_label", -1), "?"))
        pred_label = "scam" if p_dlg >= threshold else "harmless"
        match = "[OK]" if true_label_name == pred_label else "[X] "

        turns = dlg.get("turns", [])[:N]
        print(
            f"\n    {match} {dlg.get('dialogue_id', '?'):25s}  "
            f"true={true_label_name}  p_dialogue={p_dlg:.4f}  ({N} turns)"
        )

        for t in range(len(output["all_turn_q"][0])):
            q_t = output["all_turn_q"][0][t].item()
            p_agg = output["all_turn_p_agg"][0][t].item()
            alert = " [!]" if p_agg >= threshold else ""

            role = "?"
            text = ""
            if t < len(turns):
                role = turns[t].get("role", "?")
                text = turns[t].get("text", turns[t].get("text_segmented", ""))[:50]

            print(
                f"      T{t+1:<2}: q={q_t:.4f}  p_agg={p_agg:.4f}{alert}"
                f" ({role}: {text}...)"
            )


# ── CLI ──────────────────────────────────────────────────────────

def parse_args():
    cfg = EarlyExitConfig()
    default_data = os.path.join(cfg.data_dir, "excel_test.json")
    default_model = os.path.join(cfg.output_dir, "best_model")

    parser = argparse.ArgumentParser(description="Test Early-Exit Noisy-OR on Excel data.")
    parser.add_argument("--data",      default=default_data,
                        help="JSON từ convert_excel.py")
    parser.add_argument("--model",     default=default_model)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch",     type=int,   default=cfg.batch_size)
    parser.add_argument("--verbose",   action="store_true",
                        help="In turn probs từng dialogue")
    parser.add_argument("--max-verbose", type=int, default=20,
                        help="Số dialogues tối đa khi --verbose")
    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 65)
    print("EARLY-EXIT NOISY-OR — TEST ON EXCEL DATA")
    print("=" * 65)
    print(f"  Data:       {args.data}")
    print(f"  Model:      {args.model}")
    print(f"  Threshold:  {args.threshold}")

    if not os.path.exists(args.data):
        print(f"\n  [ERROR] File không tồn tại: {args.data}")
        print("  Chạy convert_excel.py trước để tạo file này.")
        sys.exit(1)

    if not os.path.exists(os.path.join(args.model, "model.pt")):
        print(f"\n  [ERROR] Không tìm thấy model: {args.model}/model.pt")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device:     {device}")

    # ── Load model ──
    config_json = os.path.join(args.model, "config.json")
    if os.path.exists(config_json):
        import dataclasses
        with open(config_json) as f:
            cfg_dict = json.load(f)
        cfg = EarlyExitConfig(**{
            k: v for k, v in cfg_dict.items()
            if k in {f.name for f in dataclasses.fields(EarlyExitConfig)}
        })
    else:
        cfg = EarlyExitConfig()

    print(f"\nLoading tokenizer & model from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = EarlyExitWeightedModel(cfg).to(device)
    model.load_state_dict(
        torch.load(os.path.join(args.model, "model.pt"), map_location=device, weights_only=True)
    )
    model.eval()
    print(f"  Trainable params: {model.count_trainable_params():,}")

    # ── Dataset ──
    print(f"\nLoading test data: {args.data}")
    dataset = EarlyExitDataset(args.data, tokenizer, cfg.max_tokens_per_turn)
    with open(args.data, encoding="utf-8") as f:
        raw = json.load(f)
    scam_n = sum(1 for d in raw if d.get("dialogue_label") == 1)
    harm_n = sum(1 for d in raw if d.get("dialogue_label") == 0)
    print(f"  {len(dataset)} conversations (scam={scam_n}, harmless={harm_n})")

    loader = DataLoader(
        dataset, batch_size=args.batch,
        shuffle=False, collate_fn=early_exit_collate_fn, num_workers=0,
    )

    # ── Evaluate ──
    print(f"\nEvaluating ({len(loader)} batches)...")
    metrics, all_labels, all_p_dialogue, all_turn_q, all_turn_p_agg = evaluate(
        model, loader, device, args.threshold
    )
    print_noisy_or_report(metrics)

    # ── Verbose per-dialogue ──
    if args.verbose:
        print_verbose(model, dataset, device, args.threshold, args.max_verbose)

    # ── Error summary ──
    preds = [1 if p >= args.threshold else 0 for p in all_p_dialogue]
    errors = [(i, l, p) for i, (l, p) in enumerate(zip(all_labels, preds)) if l != p]
    fn = sum(1 for l, p in zip(all_labels, preds) if l == 1 and p == 0)
    fp = sum(1 for l, p in zip(all_labels, preds) if l == 0 and p == 1)
    print(f"\n  Errors: {len(errors)} total  (FN={fn} scam bị bỏ sót, FP={fp} harmless báo nhầm)")

    print("\nDone!")


if __name__ == "__main__":
    main()
