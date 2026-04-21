"""
Metrics cho Streaming Binary Scam Detection.

Bao gồm:
  - Metrics chuẩn: Accuracy, Precision, Recall, F1, AUROC
  - Streaming-specific:
      + First-alert turn: turn đầu tiên model dự đoán scam
      + Average detection delay: delay từ turn scam đầu tiên đến first-alert
      + False alarm rate before scam onset
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_streaming_metrics(
    all_labels: list,
    all_logits: list,
    all_turn_masks: list,
    threshold: float = 0.5,
) -> dict:
    """
    Tính toán metrics trên toàn bộ dialogues.

    Parameters
    ----------
    all_labels : list of np.ndarray
        Mỗi phần tử là labels [T] của 1 dialogue.
    all_logits : list of np.ndarray
        Mỗi phần tử là logits [T] của 1 dialogue.
    all_turn_masks : list of np.ndarray
        Mỗi phần tử là turn_mask [T] của 1 dialogue.
    threshold : float
        Ngưỡng sigmoid để quyết định scam.

    Returns
    -------
    dict chứa tất cả metrics.
    """
    # ── Flatten tất cả turn thật (bỏ padding) ──
    flat_labels = []
    flat_probs = []

    # ── Streaming-specific accumulators ──
    first_alert_delays = []
    false_alarm_count = 0
    total_pre_scam_turns = 0
    num_scam_dialogues = 0
    num_detected_dialogues = 0

    for labels, logits, mask in zip(all_labels, all_logits, all_turn_masks):
        probs = _sigmoid(logits)
        valid_len = int(mask.sum())

        # Thu thập flat turn-level metrics
        for t in range(valid_len):
            flat_labels.append(labels[t])
            flat_probs.append(probs[t])

        valid_labels = labels[:valid_len]
        valid_probs = probs[:valid_len]
        valid_preds = (valid_probs >= threshold).astype(int)

        # ── Tìm scam onset (turn đầu tiên label=1) ──
        scam_onset = _find_first(valid_labels, 1)

        # ── Tìm first alert (turn đầu tiên model dự đoán 1) ──
        first_alert = _find_first(valid_preds, 1)

        # ── Detection delay ──
        if scam_onset is not None:
            num_scam_dialogues += 1
            if first_alert is not None:
                delay = max(0, first_alert - scam_onset)
                first_alert_delays.append(delay)
                num_detected_dialogues += 1

        # ── False alarms trước scam onset ──
        pre_scam_end = scam_onset if scam_onset is not None else valid_len
        for t in range(pre_scam_end):
            total_pre_scam_turns += 1
            if valid_preds[t] == 1:
                false_alarm_count += 1

    # ── Turn-level metrics ──
    flat_labels = np.array(flat_labels)
    flat_probs = np.array(flat_probs)
    flat_preds = (flat_probs >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(flat_labels, flat_preds),
        "precision": precision_score(flat_labels, flat_preds, zero_division=0),
        "recall": recall_score(flat_labels, flat_preds, zero_division=0),
        "f1": f1_score(flat_labels, flat_preds, zero_division=0),
    }

    # AUROC (cần cả 2 class mới tính được)
    try:
        metrics["auroc"] = roc_auc_score(flat_labels, flat_probs)
    except ValueError:
        metrics["auroc"] = 0.0

    # ── Streaming-specific metrics ──
    metrics["avg_detection_delay"] = (
        float(np.mean(first_alert_delays)) if first_alert_delays else float("nan")
    )
    metrics["detection_rate"] = (
        num_detected_dialogues / num_scam_dialogues
        if num_scam_dialogues > 0
        else 0.0
    )
    metrics["false_alarm_rate"] = (
        false_alarm_count / total_pre_scam_turns
        if total_pre_scam_turns > 0
        else 0.0
    )
    metrics["num_scam_dialogues"] = num_scam_dialogues
    metrics["num_detected"] = num_detected_dialogues

    return metrics


def print_streaming_report(metrics: dict):
    """In báo cáo metrics streaming."""
    print("\n" + "=" * 60)
    print("STREAMING BINARY SCAM DETECTION – EVALUATION REPORT")
    print("=" * 60)

    print("\n  -- Turn-level Metrics --")
    print(f"    Accuracy:   {metrics['accuracy']:.4f}")
    print(f"    Precision:  {metrics['precision']:.4f}")
    print(f"    Recall:     {metrics['recall']:.4f}")
    print(f"    F1:         {metrics['f1']:.4f}")
    print(f"    AUROC:      {metrics['auroc']:.4f}")

    print("\n  -- Streaming Metrics --")
    print(f"    Detection rate:       {metrics['detection_rate']:.4f}"
          f"  ({metrics['num_detected']}/{metrics['num_scam_dialogues']})")
    print(f"    Avg detection delay:  {metrics['avg_detection_delay']:.2f} turns")
    print(f"    False alarm rate:     {metrics['false_alarm_rate']:.4f}")
    print("=" * 60)


# ── Helpers ──────────────────────────────────────────────────
def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def _find_first(array, value):
    """Trả về index đầu tiên có giá trị == value, hoặc None."""
    indices = np.where(array == value)[0]
    return int(indices[0]) if len(indices) > 0 else None
