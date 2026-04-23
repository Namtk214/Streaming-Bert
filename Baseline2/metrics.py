"""
Multi-class Evaluation Metrics cho Baseline2 — Turn-Level Onset Labels.

Metrics:
  1. Turn-level: accuracy, macro F1, weighted F1 (mỗi turn = 1 sample)
  2. Onset detection: accuracy on SCAM dialogues (predicted onset vs true onset)
  3. Streaming: detection delay, false alarm rate
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from typing import Dict, List


# ============================================================
# Main metrics
# ============================================================
def compute_early_exit_metrics(
    all_turn_preds: List[List[int]],
    all_turn_labels: List[List[int]],
    label_names: Dict[int, str],
) -> Dict[str, float]:
    """
    Tính metrics cho Early-Exit model với turn-level labels.

    Parameters
    ----------
    all_turn_preds : list of list[int]
        Predicted label per turn, per dialogue.
    all_turn_labels : list of list[int]
        True label per turn, per dialogue.
    label_names : dict
        {0: "LEGIT", 1: "SCAM", 2: "AMBIGUOUS"}.

    Returns
    -------
    dict: metrics.
    """
    # 1. Flatten all turns for turn-level metrics
    flat_preds = []
    flat_labels = []
    for preds, labels in zip(all_turn_preds, all_turn_labels):
        for p, l in zip(preds, labels):
            if l >= 0:  # skip padding (-100)
                flat_preds.append(p)
                flat_labels.append(l)

    turn_acc = accuracy_score(flat_labels, flat_preds)
    turn_macro_f1 = f1_score(flat_labels, flat_preds, average="macro", zero_division=0)
    turn_weighted_f1 = f1_score(flat_labels, flat_preds, average="weighted", zero_division=0)

    # 2. Final-turn metrics (last turn of each dialogue)
    final_preds = []
    final_labels = []
    for preds, labels in zip(all_turn_preds, all_turn_labels):
        valid = [(p, l) for p, l in zip(preds, labels) if l >= 0]
        if valid:
            final_preds.append(valid[-1][0])
            final_labels.append(valid[-1][1])

    final_acc = accuracy_score(final_labels, final_preds) if final_labels else 0.0
    final_macro_f1 = f1_score(final_labels, final_preds, average="macro", zero_division=0) if final_labels else 0.0
    final_weighted_f1 = f1_score(final_labels, final_preds, average="weighted", zero_division=0) if final_labels else 0.0

    # 3. Onset detection metrics (SCAM dialogues only)
    # True onset = first turn with label SCAM(1) or AMBIGUOUS(2) in true labels
    # Predicted onset = first turn with predicted label != LEGIT(0)
    scam_label_id = None
    ambig_label_id = None
    legit_label_id = None
    for lid, lname in label_names.items():
        if lname == "SCAM":
            scam_label_id = lid
        elif lname == "AMBIGUOUS":
            ambig_label_id = lid
        elif lname == "LEGIT":
            legit_label_id = lid

    onset_errors = []
    detection_delays = []
    num_scam_dialogues = 0
    num_detected = 0
    num_legit_dialogues = 0
    num_false_alarms = 0

    for preds, labels in zip(all_turn_preds, all_turn_labels):
        valid_labels = [l for l in labels if l >= 0]
        valid_preds = preds[:len(valid_labels)]

        # Find true onset (first non-LEGIT turn)
        true_onset = None
        for t, l in enumerate(valid_labels):
            if l != legit_label_id:
                true_onset = t + 1  # 1-based
                break

        # Find predicted onset (first non-LEGIT prediction)
        pred_onset = None
        for t, p in enumerate(valid_preds):
            if p != legit_label_id:
                pred_onset = t + 1  # 1-based
                break

        if true_onset is not None:
            # This is a SCAM dialogue
            num_scam_dialogues += 1

            if pred_onset is not None:
                num_detected += 1
                onset_errors.append(abs(pred_onset - true_onset))
                # Detection delay: how many turns after true onset was it detected?
                delay = max(0, pred_onset - true_onset)
                detection_delays.append(delay)
        else:
            # This is a LEGIT dialogue
            num_legit_dialogues += 1
            if pred_onset is not None:
                num_false_alarms += 1

    detection_rate = num_detected / max(num_scam_dialogues, 1)
    mean_onset_error = float(np.mean(onset_errors)) if onset_errors else float("nan")
    avg_detection_delay = float(np.mean(detection_delays)) if detection_delays else float("nan")
    false_alarm_rate = num_false_alarms / max(num_legit_dialogues, 1)

    return {
        # Turn-level
        "turn_accuracy": turn_acc,
        "turn_macro_f1": turn_macro_f1,
        "turn_weighted_f1": turn_weighted_f1,
        # Final-turn
        "final_accuracy": final_acc,
        "final_macro_f1": final_macro_f1,
        "final_weighted_f1": final_weighted_f1,
        # Onset detection
        "detection_rate": detection_rate,
        "mean_onset_error": mean_onset_error,
        "avg_detection_delay": avg_detection_delay,
        "false_alarm_rate": false_alarm_rate,
        # Counts
        "num_scam_dialogues": num_scam_dialogues,
        "num_legit_dialogues": num_legit_dialogues,
        "num_detected": num_detected,
        "num_false_alarms": num_false_alarms,
    }


# ============================================================
# Pretty print report
# ============================================================
def print_early_exit_report(
    metrics: Dict[str, float],
    label_names: Dict[int, str],
):
    """In báo cáo metrics đẹp."""
    print("\n" + "=" * 60)
    print("EARLY-EXIT EVALUATION REPORT (Turn-Level Onset)")
    print("=" * 60)

    print("\n  Turn-Level Metrics:")
    print(f"    Accuracy:    {metrics['turn_accuracy']:.4f}")
    print(f"    Macro F1:    {metrics['turn_macro_f1']:.4f}")
    print(f"    Weighted F1: {metrics['turn_weighted_f1']:.4f}")

    print("\n  Final-Turn Metrics:")
    print(f"    Accuracy:    {metrics['final_accuracy']:.4f}")
    print(f"    Macro F1:    {metrics['final_macro_f1']:.4f}")
    print(f"    Weighted F1: {metrics['final_weighted_f1']:.4f}")

    print("\n  Onset Detection:")
    print(f"    Detection rate:   {metrics['detection_rate']:.4f} "
          f"({metrics['num_detected']}/{metrics['num_scam_dialogues']})")
    if not np.isnan(metrics['mean_onset_error']):
        print(f"    Mean onset error: {metrics['mean_onset_error']:.2f} turns")
    if not np.isnan(metrics['avg_detection_delay']):
        print(f"    Avg delay:        {metrics['avg_detection_delay']:.2f} turns")
    print(f"    False alarm rate: {metrics['false_alarm_rate']:.4f} "
          f"({metrics['num_false_alarms']}/{metrics['num_legit_dialogues']})")

    if "loss" in metrics:
        print(f"\n  Loss: {metrics['loss']:.4f}")

    print("=" * 60)
