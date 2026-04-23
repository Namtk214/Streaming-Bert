"""
Metrics cho Baseline2: Early-Exit with Weighted Loss.

Bao gồm:
  - Turn-level metrics: per-turn accuracy, macro F1
  - Final-turn metrics: accuracy, precision, recall, F1 ở turn cuối
  - Streaming-specific:
      + Average detection delay (turns tới khi predict SCAM đúng)
      + False alarm rate (predict SCAM ở dialogue LEGIT)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)


def compute_early_exit_metrics(
    all_turn_logits: list,
    all_labels: list,
    all_num_turns: list,
    label_names: dict = None,
) -> dict:
    """
    Tính metrics cho Early-Exit model.

    Parameters
    ----------
    all_turn_logits : list of list of np.ndarray
        Mỗi phần tử ngoài = 1 dialogue.
        Mỗi phần tử trong = logits [C] cho 1 turn.
    all_labels : list of int
        Dialogue-level label.
    all_num_turns : list of int
        Số turn thật mỗi dialogue.
    label_names : dict {int: str}
        Tên label (optional).

    Returns
    -------
    dict chứa tất cả metrics.
    """
    if label_names is None:
        label_names = {0: "LEGIT", 1: "SCAM", 2: "AMBIGUOUS"}

    # ── Final-turn predictions ──
    final_preds = []
    final_true = []

    # ── All-turn predictions ──
    all_turn_preds = []
    all_turn_true = []

    # ── Streaming-specific ──
    detection_delays = []
    false_alarm_dialogues = 0
    total_legit_dialogues = 0
    total_scam_dialogues = 0
    detected_scam_dialogues = 0

    scam_label_id = 1  # SCAM

    for dlg_idx, (turn_logits, label, N) in enumerate(
        zip(all_turn_logits, all_labels, all_num_turns)
    ):
        # Final turn prediction
        final_logits = turn_logits[-1]
        final_pred = int(np.argmax(final_logits))
        final_preds.append(final_pred)
        final_true.append(label)

        # All-turn predictions
        for t, logits_t in enumerate(turn_logits):
            pred_t = int(np.argmax(logits_t))
            all_turn_preds.append(pred_t)
            all_turn_true.append(label)

        # Streaming metrics
        if label == scam_label_id:
            total_scam_dialogues += 1
            # Tìm turn đầu tiên model predict SCAM
            first_scam_pred = None
            for t, logits_t in enumerate(turn_logits):
                if int(np.argmax(logits_t)) == scam_label_id:
                    first_scam_pred = t
                    break
            if first_scam_pred is not None:
                detected_scam_dialogues += 1
                detection_delays.append(first_scam_pred)

        elif label == 0:  # LEGIT
            total_legit_dialogues += 1
            # Kiểm tra false alarm: model predict SCAM ở bất kỳ turn nào
            any_scam_pred = any(
                int(np.argmax(logits_t)) == scam_label_id
                for logits_t in turn_logits
            )
            if any_scam_pred:
                false_alarm_dialogues += 1

    # ── Compute metrics ──
    final_true = np.array(final_true)
    final_preds = np.array(final_preds)

    num_classes = len(label_names)
    labels_range = list(range(num_classes))

    metrics = {
        # Final-turn
        "final_accuracy": accuracy_score(final_true, final_preds),
        "final_macro_f1": f1_score(
            final_true, final_preds, average="macro",
            labels=labels_range, zero_division=0,
        ),
        "final_weighted_f1": f1_score(
            final_true, final_preds, average="weighted",
            labels=labels_range, zero_division=0,
        ),

        # All-turn
        "turn_accuracy": accuracy_score(all_turn_true, all_turn_preds),
        "turn_macro_f1": f1_score(
            all_turn_true, all_turn_preds, average="macro",
            labels=labels_range, zero_division=0,
        ),

        # Per-class final-turn
        "final_precision_per_class": precision_score(
            final_true, final_preds, average=None,
            labels=labels_range, zero_division=0,
        ).tolist(),
        "final_recall_per_class": recall_score(
            final_true, final_preds, average=None,
            labels=labels_range, zero_division=0,
        ).tolist(),

        # Streaming
        "avg_detection_delay": (
            float(np.mean(detection_delays)) if detection_delays else float("nan")
        ),
        "detection_rate": (
            detected_scam_dialogues / total_scam_dialogues
            if total_scam_dialogues > 0 else 0.0
        ),
        "false_alarm_rate": (
            false_alarm_dialogues / total_legit_dialogues
            if total_legit_dialogues > 0 else 0.0
        ),
        "total_scam_dialogues": total_scam_dialogues,
        "detected_scam_dialogues": detected_scam_dialogues,
        "total_legit_dialogues": total_legit_dialogues,
        "false_alarm_dialogues": false_alarm_dialogues,
    }

    return metrics


def print_early_exit_report(metrics: dict, label_names: dict = None):
    """In báo cáo metrics."""
    if label_names is None:
        label_names = {0: "LEGIT", 1: "SCAM", 2: "AMBIGUOUS"}

    print("\n" + "=" * 60)
    print("EARLY-EXIT WEIGHTED LOSS – EVALUATION REPORT")
    print("=" * 60)

    print("\n  -- Final-Turn Metrics --")
    print(f"    Accuracy:     {metrics['final_accuracy']:.4f}")
    print(f"    Macro F1:     {metrics['final_macro_f1']:.4f}")
    print(f"    Weighted F1:  {metrics['final_weighted_f1']:.4f}")

    print("\n  -- Per-Class (Final Turn) --")
    for i, name in label_names.items():
        p = metrics["final_precision_per_class"][i]
        r = metrics["final_recall_per_class"][i]
        print(f"    {name:12s}: Prec={p:.4f}  Recall={r:.4f}")

    print("\n  -- All-Turn Metrics --")
    print(f"    Accuracy:     {metrics['turn_accuracy']:.4f}")
    print(f"    Macro F1:     {metrics['turn_macro_f1']:.4f}")

    print("\n  -- Streaming Metrics --")
    print(f"    Detection rate:       {metrics['detection_rate']:.4f}"
          f"  ({metrics['detected_scam_dialogues']}/{metrics['total_scam_dialogues']})")
    if not np.isnan(metrics['avg_detection_delay']):
        print(f"    Avg detection delay:  {metrics['avg_detection_delay']:.2f} turns")
    print(f"    False alarm rate:     {metrics['false_alarm_rate']:.4f}"
          f"  ({metrics['false_alarm_dialogues']}/{metrics['total_legit_dialogues']})")
    print("=" * 60)
