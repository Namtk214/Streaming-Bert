"""
Chuyển đổi raw conversations → Baseline2 format cho Early-Exit with Weighted Loss.

Quy trình:
  1. Đọc raw conversations JSON
  2. Clean text + word segmentation (VnCoreNLP)
  3. Gán dialogue-level label (LEGIT=0, SCAM=1, AMBIGUOUS=2)
  4. Chia train/val/test theo conversation_id (stratified)
  5. Lưu JSON format

Output format mỗi dialogue:
  {
    "dialogue_id": "...",
    "conversation_label": "SCAM",
    "label_id": 1,
    "turns": [
      {"turn_id": 1, "speaker": 0, "text": "...", "text_segmented": "..."},
      ...
    ]
  }
"""

import json
import os
import argparse
import random
import re
import sys
import unicodedata
from typing import Dict, List, Tuple

# Import config
_baseline2_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _baseline2_dir)

from config import EarlyExitConfig, LABEL_MAP

SPEAKER_MAP = {"normal": 0, "scammer": 1, "unknown": 2}


# ============================================================
# Text Cleaning
# ============================================================
def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def clean_text(text: str) -> str:
    text = normalize_unicode(text)
    text = re.sub(r"[\x00-\x09\x0b-\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# Word Segmentation
# ============================================================
class WordSegmenter:
    """
    Wrapper cho VnCoreNLP word segmentation.

    PhoBERT yêu cầu input phải được word-segment bằng VnCoreNLP.
    Nếu không load được VnCoreNLP → raise error.
    """

    def __init__(self, vncorenlp_dir: str):
        import py_vncorenlp

        if not os.path.exists(os.path.join(vncorenlp_dir, "VnCoreNLP-1.2.jar")):
            print(f"  Đang tải VnCoreNLP models vào {vncorenlp_dir}...")
            os.makedirs(vncorenlp_dir, exist_ok=True)
            py_vncorenlp.download_model(save_dir=vncorenlp_dir)

        self.segmenter = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"],
            save_dir=vncorenlp_dir,
        )
        print("  VnCoreNLP loaded OK")

    def segment(self, text: str) -> str:
        result = self.segmenter.word_segment(text)
        if isinstance(result, list):
            return " ".join(result)
        return result


# ============================================================
# Convert raw → Baseline2 format
# ============================================================
def assign_turn_labels(t1_label: str, scam_onset: int, num_turns: int) -> list:
    """
    Gán label per-turn dựa trên scam_onset_hint.

    Quy tắc:
      - LEGIT dialogue: tất cả turns → LEGIT (0)
      - SCAM dialogue:
          turns trước onset  → LEGIT (0)
          turn onset         → AMBIGUOUS (2)
          turns sau onset    → SCAM (1)

    Parameters
    ----------
    t1_label : str
        Dialogue-level label ("LEGIT" hoặc "SCAM").
    scam_onset : int or None
        Turn index (1-based) bắt đầu lừa đảo. None cho LEGIT.
    num_turns : int
        Tổng số turn.

    Returns
    -------
    list of int: label cho từng turn.
    """
    if t1_label == "LEGIT" or scam_onset is None:
        return [LABEL_MAP["LEGIT"]] * num_turns

    labels = []
    for t in range(1, num_turns + 1):  # 1-based
        if t < scam_onset:
            labels.append(LABEL_MAP["LEGIT"])
        elif t == scam_onset:
            labels.append(LABEL_MAP["AMBIGUOUS"])
        else:
            labels.append(LABEL_MAP["SCAM"])
    return labels


def convert_to_baseline2(conversation: Dict, segmenter: WordSegmenter) -> Dict:
    """Chuyển 1 conversation thành Baseline2 format với turn-level labels."""
    messages = conversation["messages"]
    t1_label = conversation["t1_label"]
    scam_onset = conversation.get("scam_onset_hint", None)

    num_turns = len(messages)
    turn_labels = assign_turn_labels(t1_label, scam_onset, num_turns)

    turns = []
    for i, msg in enumerate(messages):
        text_clean = clean_text(msg["text"])
        text_segmented = segmenter.segment(text_clean)

        speaker_role = msg.get("speaker_role", "unknown")
        speaker_id = SPEAKER_MAP.get(speaker_role, SPEAKER_MAP["unknown"])

        turns.append({
            "turn_id": i + 1,
            "speaker": speaker_id,
            "text": text_clean,
            "text_segmented": text_segmented,
            "label": turn_labels[i],
        })

    return {
        "dialogue_id": conversation["conversation_id"],
        "conversation_label": t1_label,
        "scam_onset": scam_onset,
        "turns": turns,
    }


# ============================================================
# Data split (stratified)
# ============================================================
def split_dialogues(
    dialogues: List[Dict],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Chia theo dialogue_id, stratified theo conversation_label."""
    random.seed(seed)

    by_label = {}
    for dlg in dialogues:
        label = dlg["conversation_label"]
        by_label.setdefault(label, []).append(dlg)

    train, val, test = [], [], []

    for label, items in by_label.items():
        random.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))

        test.extend(items[:n_test])
        val.extend(items[n_test:n_test + n_val])
        train.extend(items[n_test + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


# ============================================================
# Main pipeline
# ============================================================
def parse_args() -> argparse.Namespace:
    cfg = EarlyExitConfig()
    parser = argparse.ArgumentParser(
        description="Convert raw conversation JSON to Baseline2 train/val/test JSON files."
    )
    parser.add_argument(
        "input_json",
        nargs="?",
        default=cfg.raw_data_path,
        help="Path to raw conversations JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=cfg.data_dir,
        help="Output dataset folder.",
    )
    parser.add_argument(
        "--vncorenlp-dir",
        default=cfg.vncorenlp_dir,
        help="VnCoreNLP model directory.",
    )
    parser.add_argument("--val-ratio", type=float, default=cfg.val_ratio)
    parser.add_argument("--test-ratio", type=float, default=cfg.test_ratio)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    return parser.parse_args()


def load_conversations(input_json: str) -> List[Dict]:
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    with open(input_json, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    if not isinstance(conversations, list):
        raise ValueError("Input JSON must be a list of conversations.")

    required_keys = {"conversation_id", "t1_label", "messages"}
    for idx, conversation in enumerate(conversations):
        missing = required_keys - set(conversation.keys())
        if missing:
            raise ValueError(
                f"Conversation at index {idx} is missing keys: {sorted(missing)}"
            )

    return conversations


def main():
    args = parse_args()

    # Fix Windows console encoding
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("PREPARE BASELINE2 DATA — Early-Exit with Weighted Loss")
    print("=" * 60)

    # 1. Đọc raw data
    print(f"  Input JSON: {args.input_json}")
    print(f"  Output dir: {args.output_dir}")
    all_conversations = load_conversations(args.input_json)
    print(f"  Total: {len(all_conversations)} conversations")

    # Thống kê
    labels = [c["t1_label"] for c in all_conversations]
    for lbl in sorted(set(labels)):
        print(f"    {lbl}: {labels.count(lbl)}")

    # Filter: chỉ giữ conversations có label hợp lệ
    valid_labels = set(LABEL_MAP.keys())
    all_conversations = [c for c in all_conversations if c["t1_label"] in valid_labels]
    print(f"  After filtering: {len(all_conversations)} conversations")

    # 2. Word segmentation
    print("\nInitializing word segmenter...")
    segmenter = WordSegmenter(args.vncorenlp_dir)

    # 3. Convert
    print("\nConverting to Baseline2 format...")
    dialogues = []
    for i, conv in enumerate(all_conversations):
        dlg = convert_to_baseline2(conv, segmenter)
        dialogues.append(dlg)
        if (i + 1) % 50 == 0:
            print(f"  Converted {i + 1}/{len(all_conversations)}")
    print(f"  Done: {len(dialogues)} dialogues")

    # 4. Split
    print("\nSplitting data...")
    train, val, test = split_dialogues(
        dialogues, args.val_ratio, args.test_ratio, args.seed
    )
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # Thống kê labels per split
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        split_labels = [d["conversation_label"] for d in split]
        counts = {lbl: split_labels.count(lbl) for lbl in sorted(set(split_labels))}
        print(f"    {name}: {counts}")

    # 5. Lưu
    os.makedirs(args.output_dir, exist_ok=True)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(args.output_dir, f"{split_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  Saved {split_name}: {path}")

    # 6. Preview
    LABEL_NAMES_REV = {v: k for k, v in LABEL_MAP.items()}
    print("\n-- Sample dialogue (train) --")
    if train:
        sample = train[0]
        print(f"  ID: {sample['dialogue_id']} | Label: {sample['conversation_label']} "
              f"| onset: {sample.get('scam_onset')}")
        for turn in sample["turns"][:6]:
            lbl_name = LABEL_NAMES_REV.get(turn['label'], '?')
            print(f"    Turn {turn['turn_id']} (speaker={turn['speaker']}): "
                  f"[{lbl_name}] {turn['text'][:50]}")

    # 7. Turn-level label stats
    all_turn_labels = [t['label'] for d in train for t in d['turns']]
    print(f"\n  Turn-level label distribution (train):")
    for lbl_id, lbl_name in sorted(LABEL_NAMES_REV.items()):
        cnt = all_turn_labels.count(lbl_id)
        print(f"    {lbl_name}: {cnt} ({cnt/len(all_turn_labels)*100:.1f}%)")

    print("\nDone!")


if __name__ == "__main__":
    main()
