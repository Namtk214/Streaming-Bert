"""
Test StreamingScamDetector trên dữ liệu đã convert từ Excel.

Đọc file JSON (output của convert_excel.py), chạy evaluate()
qua DataLoader giống hệt train.py, báo cáo đầy đủ metrics.

Usage:
    python test.py
    python test.py --data data/excel_test.json --threshold 0.4
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

from config import StreamingConfig
from dataset import StreamingDialogueDataset, streaming_collate_fn
from model import StreamingScamDetector
from metrics import compute_streaming_metrics, print_streaming_report


# ── Evaluate ─────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, device, threshold):
    model.eval()
    total_loss, n_dlg = 0.0, 0
    all_labels, all_d_probs, all_t_probs = [], [], []

    for batch in dataloader:
        batch  = {k: v.to(device) for k, v in batch.items()}
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            turn_mask=batch["turn_mask"],
            dialogue_labels=batch["dialogue_labels"],
        )
        B = batch["dialogue_labels"].shape[0]
        if output["loss"] is not None:
            total_loss += output["loss"].item() * B
        n_dlg += B

        for i in range(B):
            vlen = int(batch["turn_mask"][i].sum())
            all_labels.append(int(batch["dialogue_labels"][i].item()))
            all_d_probs.append(float(output["dialogue_probs"][i].item()))
            all_t_probs.append(output["turn_probs"][i, :vlen].cpu().numpy())

    metrics = compute_streaming_metrics(all_labels, all_d_probs, all_t_probs, threshold)
    metrics["loss"] = total_loss / max(n_dlg, 1)
    return metrics, all_labels, all_d_probs, all_t_probs


# ── Per-dialogue detail ───────────────────────────────────────────

@torch.no_grad()
def print_verbose(model, dataset, device, threshold, max_print=20):
    import math as _math
    model.eval()
    print(f"\n  -- Per-dialogue detail (first {max_print}) --")

    for idx in range(min(len(dataset), max_print)):
        item  = dataset[idx]
        dlg   = dataset.dialogues[idx]
        from dataset import streaming_collate_fn
        batch = streaming_collate_fn([item])
        batch = {k: v.to(device) for k, v in batch.items()}

        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            turn_mask=batch["turn_mask"],
        )
        vlen    = int(batch["turn_mask"][0].sum())
        t_probs = output["turn_probs"][0, :vlen].cpu().tolist()
        d_prob  = float(output["dialogue_probs"][0].item())
        true_lbl = dlg.get("conversation_label", "?")
        pred_lbl = "scam" if d_prob >= threshold else "harmless"
        match    = "[OK]" if true_lbl == pred_lbl else "[X] "

        turns = dlg.get("turns", [])[:vlen]
        print(
            f"\n    {match} {dlg.get('dialogue_id'):25s}  "
            f"true={true_lbl}  p_dialogue={d_prob:.4f}  ({vlen} turns)"
        )

        log_comp = 0.0
        for t, (q, turn) in enumerate(zip(t_probs, turns), 1):
            log_comp += _math.log(max(1.0 - q, 1e-7))
            p_agg = 1.0 - _math.exp(log_comp)
            alert = " [!]" if p_agg >= threshold else ""
            spk = {0: "người gọi", 1: "người nghe"}.get(turn.get("speaker"), "?")
            print(
                f"      T{t:<2}: q={q:.4f}  p_agg={p_agg:.4f}{alert}"
                f" ({spk}: {turn.get('text', '')[:50]}...)"
            )


# ── CLI ──────────────────────────────────────────────────────────

def parse_args():
    cfg = StreamingConfig()
    default_data = os.path.join(cfg.streaming_data_dir, "excel_test.json")
    default_model = os.path.join(cfg.output_dir, "best_model")

    parser = argparse.ArgumentParser(description="Test Streaming Scam Detector on Excel data.")
    parser.add_argument("--data",      default=default_data,
                        help="JSON từ convert_excel.py")
    parser.add_argument("--model",     default=default_model)
    parser.add_argument("--threshold", type=float, default=cfg.threshold)
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
    print("STREAMING SCAM DETECTOR — TEST ON EXCEL DATA")
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
        cfg = StreamingConfig(**{
            k: v for k, v in cfg_dict.items()
            if k in {f.name for f in dataclasses.fields(StreamingConfig)}
        })
    else:
        cfg = StreamingConfig()

    print(f"\nLoading tokenizer & model from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = StreamingScamDetector(cfg).to(device)
    model.load_state_dict(
        torch.load(os.path.join(args.model, "model.pt"), map_location=device, weights_only=True)
    )
    model.eval()
    print(f"  Trainable params: {model.count_trainable_params():,}")

    # ── Dataset ──
    print(f"\nLoading test data: {args.data}")
    dataset = StreamingDialogueDataset(args.data, tokenizer, cfg.max_tokens_per_turn)
    with open(args.data, encoding="utf-8") as f:
        raw = json.load(f)
    scam_n = sum(1 for d in raw if d["conversation_label"] == "scam")
    harm_n = sum(1 for d in raw if d["conversation_label"] == "harmless")
    print(f"  {len(dataset)} conversations (scam={scam_n}, harmless={harm_n})")

    loader = DataLoader(
        dataset, batch_size=args.batch,
        shuffle=False, collate_fn=streaming_collate_fn, num_workers=0,
    )

    # ── Evaluate ──
    print(f"\nEvaluating ({len(loader)} batches)...")
    metrics, all_labels, all_d_probs, all_t_probs = evaluate(
        model, loader, device, args.threshold
    )
    print_streaming_report(metrics)

    # ── Verbose per-dialogue ──
    if args.verbose:
        print_verbose(model, dataset, device, args.threshold, args.max_verbose)

    # ── Error analysis ──
    preds      = [1 if p >= args.threshold else 0 for p in all_d_probs]
    fn_indices = [i for i, (l, p) in enumerate(zip(all_labels, preds)) if l == 1 and p == 0]
    fp_indices = [i for i, (l, p) in enumerate(zip(all_labels, preds)) if l == 0 and p == 1]
    n_errors   = len(fn_indices) + len(fp_indices)

    print(f"\n  Errors: {n_errors} total  "
          f"(FN={len(fn_indices)} scam bị bỏ sót, FP={len(fp_indices)} harmless báo nhầm)")

    def _print_error_cases(indices, label_str, limit=10):
        for i in indices[:limit]:
            dlg    = dataset.dialogues[i]
            d_prob = all_d_probs[i]
            turns  = dlg.get("turns", [])
            print(
                f"    [{label_str}] {dlg.get('dialogue_id'):25s}  "
                f"p={d_prob:.3f}  ({len(turns)} turns)"
            )
            for t in turns[:3]:
                spk = {0: "gọi", 1: "nghe"}.get(t.get("speaker"), "?")
                print(f"           [{spk}] {t.get('text', '')[:70]}")

    if fn_indices:
        print(f"\n  ── False Negatives (scam bị bỏ sót, top {min(20, len(fn_indices))}) ──")
        _print_error_cases(fn_indices, "FN", limit=20)

    if fp_indices:
        print(f"\n  ── False Positives (harmless báo nhầm, top {min(20, len(fp_indices))}) ──")
        _print_error_cases(fp_indices, "FP", limit=20)

    print("\nDone!")


if __name__ == "__main__":
    main()
