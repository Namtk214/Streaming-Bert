"""
Dialogue-Level Binary Evaluation Metrics cho Baseline2 — Noisy-OR Loss.

Metrics:
  1. Dialogue-level: accuracy, precision, recall, F1, AUROC
  2. Streaming: first alert turn, average detection delay, false alarm rate
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from typing import Dict, List, Optional


# ============================================================
# Main metrics
# ============================================================
def compute_noisy_or_metrics(
    all_p_dialogue: List[float],
    all_labels: List[int],
    all_turn_q: Optional[List[List[float]]] = None,
    all_turn_p_agg: Optional[List[List[float]]] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Tính metrics cho Noisy-OR model với dialogue-level labels.

    Parameters
    ----------
    all_p_dialogue : list of float
        p_dialogue cho mỗi dialogue (Noisy-OR aggregated probability).
    all_labels : list of int
        True dialogue-level label (0=harmless, 1=scam).
    all_turn_q : list of list of float (optional)
        Per-turn evidence probabilities q_t.
    all_turn_p_agg : list of list of float (optional)
        Per-turn cumulative probabilities p_t_agg.
    threshold : float
        Threshold cho binary classification.

    Returns
    -------
    dict: metrics.
    """
    # Binary predictions
    preds = [1 if p >= threshold else 0 for p in all_p_dialogue]

    # Dialogue-level metrics
    accuracy = accuracy_score(all_labels, preds)
    precision = precision_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)

    # AUROC (cần ít nhất 2 classes trong labels)
    try:
        auroc = roc_auc_score(all_labels, all_p_dialogue)
    except ValueError:
        auroc = float("nan")

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
    }

    # Streaming metrics (nếu có turn-level data)
    if all_turn_p_agg is not None:
        num_scam = 0
        num_harmless = 0
        detection_delays = []
        false_alarms = 0
        first_alert_turns = []

        for i, (p_agg_list, label) in enumerate(zip(all_turn_p_agg, all_labels)):
            # Tìm first alert turn (first turn where p_agg >= threshold)
            first_alert = None
            for t, p_agg in enumerate(p_agg_list):
                if p_agg >= threshold:
                    first_alert = t + 1  # 1-based
                    break

            if label == 1:
                # SCAM dialogue
                num_scam += 1
                if first_alert is not None:
                    first_alert_turns.append(first_alert)
                    # Detection delay = which turn it was first detected
                    detection_delays.append(first_alert)
            else:
                # HARMLESS dialogue
                num_harmless += 1
                if first_alert is not None:
                    false_alarms += 1

        detection_rate = len(detection_delays) / max(num_scam, 1)
        avg_detection_delay = float(np.mean(detection_delays)) if detection_delays else float("nan")
        avg_first_alert = float(np.mean(first_alert_turns)) if first_alert_turns else float("nan")
        false_alarm_rate = false_alarms / max(num_harmless, 1)

        metrics.update({
            "detection_rate": detection_rate,
            "avg_detection_delay": avg_detection_delay,
            "avg_first_alert_turn": avg_first_alert,
            "false_alarm_rate": false_alarm_rate,
            "num_scam_dialogues": num_scam,
            "num_harmless_dialogues": num_harmless,
            "num_detected": len(detection_delays),
            "num_false_alarms": false_alarms,
        })

    # Turn-level evidence analysis (nếu có)
    if all_turn_q is not None:
        scam_q_means = []
        harmless_q_means = []
        for q_list, label in zip(all_turn_q, all_labels):
            mean_q = float(np.mean(q_list))
            if label == 1:
                scam_q_means.append(mean_q)
            else:
                harmless_q_means.append(mean_q)

        metrics["mean_q_scam"] = float(np.mean(scam_q_means)) if scam_q_means else float("nan")
        metrics["mean_q_harmless"] = float(np.mean(harmless_q_means)) if harmless_q_means else float("nan")

    return metrics


# ============================================================
# Pretty print report
# ============================================================
def print_noisy_or_report(metrics: Dict[str, float]):
    """In báo cáo metrics đẹp."""
    print("\n" + "=" * 60)
    print("NOISY-OR EVALUATION REPORT (Dialogue-Level Binary)")
    print("=" * 60)

    print("\n  Dialogue-Level Binary Metrics:")
    print(f"    Accuracy:    {metrics['accuracy']:.4f}")
    print(f"    Precision:   {metrics['precision']:.4f}")
    print(f"    Recall:      {metrics['recall']:.4f}")
    print(f"    F1:          {metrics['f1']:.4f}")
    if not np.isnan(metrics.get('auroc', float('nan'))):
        print(f"    AUROC:       {metrics['auroc']:.4f}")

    if 'detection_rate' in metrics:
        print("\n  Streaming Detection:")
        print(f"    Detection rate:   {metrics['detection_rate']:.4f} "
              f"({metrics['num_detected']}/{metrics['num_scam_dialogues']})")
        if not np.isnan(metrics.get('avg_detection_delay', float('nan'))):
            print(f"    Avg 1st alert:    turn {metrics['avg_first_alert_turn']:.1f}")
            print(f"    Avg delay:        {metrics['avg_detection_delay']:.2f} turns")
        print(f"    False alarm rate: {metrics['false_alarm_rate']:.4f} "
              f"({metrics['num_false_alarms']}/{metrics['num_harmless_dialogues']})")

    if 'mean_q_scam' in metrics:
        print("\n  Evidence Analysis:")
        if not np.isnan(metrics.get('mean_q_scam', float('nan'))):
            print(f"    Mean q (scam):     {metrics['mean_q_scam']:.4f}")
        if not np.isnan(metrics.get('mean_q_harmless', float('nan'))):
            print(f"    Mean q (harmless): {metrics['mean_q_harmless']:.4f}")

    if "loss" in metrics:
        print(f"\n  Loss: {metrics['loss']:.4f}")

    print("=" * 60)
