"""
Evaluation Metrics cho Streaming Binary Scam Detection (Noisy-OR MIL).

Input từ evaluate():
  - all_dialogue_labels : List[int]          – 0/1 mỗi dialogue
  - all_dialogue_probs  : List[float]        – Noisy-OR aggregated prob
  - all_turn_probs      : List[np.ndarray]   – p_t per real turn [T]

Metrics:
  1. Dialogue-level: accuracy, F1, AUROC  (dựa vào dialogue_probs)
  2. Streaming detection (dựa vào turn_probs):
       - detection_rate     : % scam dialogues được alert ít nhất 1 turn
       - avg_detection_delay: trung bình turn đầu tiên có p_t ≥ threshold (0-based)
       - false_alarm_rate   : % harmless dialogues bị alert ít nhất 1 turn
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from typing import Dict, List


def compute_streaming_metrics(
    all_dialogue_labels: List[int],
    all_dialogue_probs: List[float],
    all_turn_probs: List[np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Parameters
    ----------
    all_dialogue_labels : List[int]          0/1, N dialogues
    all_dialogue_probs  : List[float]        Noisy-OR prob, N dialogues
    all_turn_probs      : List[np.ndarray]   per-turn probs, real turns only
    threshold           : float
    """
    d_labels = np.array(all_dialogue_labels)
    d_probs  = np.array(all_dialogue_probs)
    d_preds  = (d_probs >= threshold).astype(int)

    # Dialogue-level
    try:
        auroc = float(roc_auc_score(d_labels, d_probs))
    except ValueError:
        auroc = float("nan")

    detection_delays = []
    num_scam, num_detected = 0, 0
    num_harmless, num_false_alarms = 0, 0

    for label, turn_probs in zip(all_dialogue_labels, all_turn_probs):
        first_alert = _first_alert_turn(turn_probs, threshold)

        if label == 1:  # scam
            num_scam += 1
            if first_alert is not None:
                num_detected += 1
                detection_delays.append(first_alert)   # 0-based turn index
        else:           # harmless
            num_harmless += 1
            if first_alert is not None:
                num_false_alarms += 1

    metrics = {
        # Dialogue-level
        "dialogue_accuracy": float(accuracy_score(d_labels, d_preds)),
        "dialogue_f1":       float(f1_score(d_labels, d_preds, zero_division=0)),
        "auroc":             auroc,
        # Streaming
        "detection_rate":         num_detected / max(num_scam, 1),
        "avg_detection_delay":    float(np.mean(detection_delays)) if detection_delays else float("nan"),
        "false_alarm_rate":       num_false_alarms / max(num_harmless, 1),
        "num_scam":               num_scam,
        "num_harmless":           num_harmless,
        "num_detected":           num_detected,
        "num_false_alarms":       num_false_alarms,
        # aliases for train.py early stopping
        "accuracy": float(accuracy_score(d_labels, d_preds)),
        "f1":       float(f1_score(d_labels, d_preds, zero_division=0)),
    }
    return metrics


def print_streaming_report(metrics: Dict[str, float]):
    print("\n" + "=" * 60)
    print("STREAMING BINARY EVALUATION REPORT")
    print("=" * 60)

    print("\n  Dialogue-Level Metrics (Noisy-OR aggregation):")
    print(f"    Accuracy: {metrics['dialogue_accuracy']:.4f}")
    print(f"    F1:       {metrics['dialogue_f1']:.4f}")
    if not np.isnan(metrics.get("auroc", float("nan"))):
        print(f"    AUROC:    {metrics['auroc']:.4f}")

    print("\n  Streaming Detection (per-turn threshold):")
    print(
        f"    Detection rate:   {metrics['detection_rate']:.4f} "
        f"({metrics['num_detected']}/{metrics['num_scam']})"
    )
    if not np.isnan(metrics["avg_detection_delay"]):
        print(f"    Avg delay:        {metrics['avg_detection_delay']:.2f} turns")
    print(
        f"    False alarm rate: {metrics['false_alarm_rate']:.4f} "
        f"({metrics['num_false_alarms']}/{metrics['num_harmless']})"
    )

    if "loss" in metrics:
        print(f"\n  Loss: {metrics['loss']:.4f}")
    print("=" * 60)


def _first_alert_turn(turn_probs: np.ndarray, threshold: float):
    """0-based index của turn đầu tiên có p_t ≥ threshold, hoặc None."""
    for i, p in enumerate(turn_probs):
        if float(p) >= threshold:
            return i
    return None
