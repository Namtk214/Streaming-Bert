"""
Multi-class Evaluation Metrics cho Streaming 3-class Scam Detection.
Classes: 0=LEGIT, 1=SCAM, 2=AMBIGUOUS

Metrics:
  1. Turn-level: accuracy, macro F1
  2. Final-turn: accuracy, macro F1 (prediction tại turn cuối của dialogue)
  3. Onset detection: detection rate, mean onset error, avg delay
  4. False alarm: tỷ lệ LEGIT dialogues bị flag nhầm
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List

LEGIT = 0
SCAM = 1
AMBIGUOUS = 2
LABEL_NAMES = {LEGIT: "LEGIT", SCAM: "SCAM", AMBIGUOUS: "AMBIGUOUS"}


def compute_streaming_metrics(
    all_labels: List[np.ndarray],
    all_logits: List[np.ndarray],
    all_turn_masks: List[np.ndarray],
    threshold: float = 0.5,   # unused, kept for API compat
) -> Dict[str, float]:
    """
    Parameters
    ----------
    all_labels     : list of [T]   – class index, có padding
    all_logits     : list of [T,C] – raw logits 3-class, có padding
    all_turn_masks : list of [T]   – 1=real turn, 0=padding
    """
    flat_preds, flat_labels = [], []
    final_preds, final_labels = [], []

    onset_errors = []
    detection_delays = []
    num_scam_dialogues = 0
    num_detected = 0
    num_legit_dialogues = 0
    num_false_alarms = 0

    for labels, logits, mask in zip(all_labels, all_logits, all_turn_masks):
        valid_len = int(mask.sum())
        vlabels = labels[:valid_len].astype(int)
        probs = _softmax(logits[:valid_len])          # [T, C]
        vpreds = probs.argmax(axis=-1).astype(int)    # [T]

        flat_preds.extend(vpreds.tolist())
        flat_labels.extend(vlabels.tolist())

        if valid_len > 0:
            final_preds.append(int(vpreds[-1]))
            final_labels.append(int(vlabels[-1]))

        true_onset = _first_nonlegit(vlabels)
        pred_onset = _first_nonlegit(vpreds)

        if true_onset is not None:
            num_scam_dialogues += 1
            if pred_onset is not None:
                num_detected += 1
                onset_errors.append(abs(pred_onset - true_onset))
                detection_delays.append(max(0, pred_onset - true_onset))
        else:
            num_legit_dialogues += 1
            if pred_onset is not None:
                num_false_alarms += 1

    flat_labels = np.array(flat_labels)
    flat_preds = np.array(flat_preds)

    metrics = {
        "turn_accuracy":  accuracy_score(flat_labels, flat_preds),
        "turn_macro_f1":  f1_score(flat_labels, flat_preds, average="macro", zero_division=0),
        "final_accuracy": accuracy_score(final_labels, final_preds) if final_labels else 0.0,
        "final_macro_f1": f1_score(final_labels, final_preds, average="macro", zero_division=0) if final_labels else 0.0,
        "detection_rate":      num_detected / max(num_scam_dialogues, 1),
        "mean_onset_error":    float(np.mean(onset_errors))     if onset_errors     else float("nan"),
        "avg_detection_delay": float(np.mean(detection_delays)) if detection_delays else float("nan"),
        "false_alarm_rate":    num_false_alarms / max(num_legit_dialogues, 1),
        "num_scam_dialogues":  num_scam_dialogues,
        "num_legit_dialogues": num_legit_dialogues,
        "num_detected":        num_detected,
        "num_false_alarms":    num_false_alarms,
        # Aliases cho train.py
        "accuracy": accuracy_score(flat_labels, flat_preds),
        "f1":       f1_score(flat_labels, flat_preds, average="macro", zero_division=0),
        "auroc":    0.0,
    }

    return metrics


def print_streaming_report(metrics: Dict[str, float]):
    print("\n" + "=" * 60)
    print("STREAMING 3-CLASS EVALUATION REPORT")
    print("=" * 60)

    print("\n  Turn-Level Metrics:")
    print(f"    Accuracy:  {metrics['turn_accuracy']:.4f}")
    print(f"    Macro F1:  {metrics['turn_macro_f1']:.4f}")

    print("\n  Final-Turn Metrics:")
    print(f"    Accuracy:  {metrics['final_accuracy']:.4f}")
    print(f"    Macro F1:  {metrics['final_macro_f1']:.4f}")

    print("\n  Onset Detection:")
    print(f"    Detection rate:   {metrics['detection_rate']:.4f} "
          f"({metrics['num_detected']}/{metrics['num_scam_dialogues']})")
    if not np.isnan(metrics["mean_onset_error"]):
        print(f"    Mean onset error: {metrics['mean_onset_error']:.2f} turns")
    if not np.isnan(metrics["avg_detection_delay"]):
        print(f"    Avg delay:        {metrics['avg_detection_delay']:.2f} turns")
    print(f"    False alarm rate: {metrics['false_alarm_rate']:.4f} "
          f"({metrics['num_false_alarms']}/{metrics['num_legit_dialogues']})")

    if "loss" in metrics:
        print(f"\n  Loss: {metrics['loss']:.4f}")
    print("=" * 60)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _first_nonlegit(array) -> int | None:
    """Trả về 1-based index của turn non-LEGIT đầu tiên, hoặc None."""
    for i, v in enumerate(array):
        if v != LEGIT:
            return i + 1
    return None
