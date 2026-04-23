"""
Chuyển đổi raw conversations → streaming format cho training.

Quy trình:
  1. Đọc raw conversations JSON từ đường dẫn input
  2. Clean text + word segmentation (VnCoreNLP)
  3. Gán turn-level labels theo onset:
     - LEGIT: 0
     - SCAM: 1
     - AMBIGUOUS: 2
     - scam_label binary vẫn được giữ để train model hiện tại
  4. Chia train/val/test theo conversation_id
  5. Lưu streaming format JSON

Output format mỗi dialogue:
  {
    "dialogue_id": "...",
    "turns": [
      {"turn_id": 1, "speaker": 0, "text": "...",
       "text_segmented": "...", "scam_label": 0,
       "turn_label": 0, "turn_label_name": "LEGIT"},
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

# Import streaming config (explicit path to avoid conflict with src/config.py)
_streaming_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _streaming_dir)

import importlib.util
_spec = importlib.util.spec_from_file_location("streaming_config", os.path.join(_streaming_dir, "config.py"))
_cfg_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg_mod)
StreamingConfig = _cfg_mod.StreamingConfig
SPEAKER_MAP = _cfg_mod.SPEAKER_MAP

TURN_LABEL_MAP = {"LEGIT": 0, "SCAM": 1, "AMBIGUOUS": 2}
TURN_LABEL_NAMES = {v: k for k, v in TURN_LABEL_MAP.items()}
GENERIC_T4_LABELS = {"SCAM_INDICATOR"}


# ============================================================
# Text Cleaning (tái sử dụng logic từ src/preprocessing.py)
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
    """Wrapper cho VnCoreNLP. Fallback nếu không cài được."""

    def __init__(self, vncorenlp_dir: str):
        self.segmenter = None
        try:
            import py_vncorenlp
            if not os.path.exists(os.path.join(vncorenlp_dir, "VnCoreNLP-1.2.jar")):
                print(f"Đang tải VnCoreNLP models vào {vncorenlp_dir}...")
                os.makedirs(vncorenlp_dir, exist_ok=True)
                py_vncorenlp.download_model(save_dir=vncorenlp_dir)
            self.segmenter = py_vncorenlp.VnCoreNLP(
                annotators=["wseg"],
                save_dir=vncorenlp_dir,
            )
            print("  VnCoreNLP loaded OK")
        except Exception as e:
            print(f"  WARNING: Cannot load VnCoreNLP ({e}). Using raw text as fallback.")

    def segment(self, text: str) -> str:
        if self.segmenter is None:
            return text  # fallback: dùng text gốc
        result = self.segmenter.word_segment(text)
        if isinstance(result, list):
            return " ".join(result)
        return result


# ============================================================
# Turn label generation theo onset
# ============================================================
def find_scam_onset(conversation: Dict) -> int:
    """
    Tìm scam onset dạng 0-based index.

    Ưu tiên:
      1. scam_onset_hint nếu có trong raw JSON. Field này được xem là 1-based.
      2. t4_labels thật, bỏ qua marker generic do converter thêm.
      3. turn thứ 2 của scammer để tránh coi greeting là onset.
      4. giữa conversation.
    """
    messages = conversation["messages"]
    num_turns = len(messages)

    hint = conversation.get("scam_onset_hint")
    if hint is not None:
        hint = int(hint)
        return max(0, min(num_turns - 1, hint - 1 if hint > 0 else hint))

    for i, msg in enumerate(messages):
        t4_labels = set(msg.get("t4_labels") or [])
        if t4_labels and not t4_labels.issubset(GENERIC_T4_LABELS):
            return i

    scammer_turns = [
        i for i, msg in enumerate(messages)
        if msg.get("speaker_role") == "scammer"
    ]
    if len(scammer_turns) >= 2:
        return scammer_turns[1]
    if scammer_turns:
        return scammer_turns[0]

    return num_turns // 2


def assign_turn_labels(conversation: Dict) -> List[int]:
    """
    Gán turn-level labels theo onset.

    LEGIT:
      - Toàn bộ turns = LEGIT.

    SCAM/AMBIGUOUS:
      - Trước onset = LEGIT
      - Turn onset = AMBIGUOUS
      - Sau onset = SCAM
    """
    messages = conversation["messages"]
    t1_label = conversation["t1_label"]
    num_turns = len(messages)

    if t1_label == "LEGIT":
        return [TURN_LABEL_MAP["LEGIT"]] * num_turns

    scam_onset = find_scam_onset(conversation)
    labels = []
    for i in range(num_turns):
        if i < scam_onset:
            labels.append(TURN_LABEL_MAP["LEGIT"])
        elif i == scam_onset:
            labels.append(TURN_LABEL_MAP["AMBIGUOUS"])
        else:
            labels.append(TURN_LABEL_MAP["SCAM"])
    return labels


def turn_label_to_binary(label: int) -> int:
    """Binary target cho model hiện tại: chỉ SCAM là positive."""
    return 1 if label == TURN_LABEL_MAP["SCAM"] else 0


# ============================================================
# Convert raw → streaming format
# ============================================================
def convert_to_streaming(conversation: Dict, segmenter: WordSegmenter) -> Dict:
    """Chuyển 1 conversation thành streaming format."""
    messages = conversation["messages"]
    turn_labels = assign_turn_labels(conversation)
    binary_labels = [turn_label_to_binary(label) for label in turn_labels]
    onset = next((i + 1 for i, label in enumerate(turn_labels)
                  if label != TURN_LABEL_MAP["LEGIT"]), None)

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
            "scam_label": binary_labels[i],
            "turn_label": turn_labels[i],
            "turn_label_name": TURN_LABEL_NAMES[turn_labels[i]],
        })

    return {
        "dialogue_id": conversation["conversation_id"],
        "conversation_label": conversation["t1_label"],
        "scam_onset": onset,
        "turns": turns,
    }


# ============================================================
# Data split
# ============================================================
def split_dialogues(
    dialogues: List[Dict],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Chia theo dialogue_id, stratified theo conversation_label."""
    random.seed(seed)

    # Group by label
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
    cfg = StreamingConfig()
    parser = argparse.ArgumentParser(
        description="Convert raw conversation JSON to Streaming-Bert train/val/test JSON files."
    )
    parser.add_argument(
        "input_json",
        nargs="?",
        default=cfg.raw_data_path,
        help="Path to raw conversations JSON. Defaults to config raw_data_path.",
    )
    parser.add_argument(
        "--output-dir",
        default=cfg.streaming_data_dir,
        help="Output dataset folder containing train.json, val.json, and test.json.",
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

    print("=" * 60)
    print("PREPARE STREAMING DATA")
    print("=" * 60)

    # ── 1. Đọc raw data ──
    print(f"  Input JSON: {args.input_json}")
    print(f"  Output dir: {args.output_dir}")
    all_conversations = load_conversations(args.input_json)
    print(f"  Total: {len(all_conversations)} conversations")

    # Thống kê
    labels = [c["t1_label"] for c in all_conversations]
    for lbl in sorted(set(labels)):
        print(f"    {lbl}: {labels.count(lbl)}")

    # ── 2. Word segmentation ──
    print("\nInitializing word segmenter...")
    segmenter = WordSegmenter(args.vncorenlp_dir)

    # ── 3. Convert to streaming format ──
    print("\nConverting to streaming format...")
    streaming_dialogues = []
    for i, conv in enumerate(all_conversations):
        dlg = convert_to_streaming(conv, segmenter)
        streaming_dialogues.append(dlg)
        if (i + 1) % 10 == 0:
            print(f"  Converted {i + 1}/{len(all_conversations)}")
    print(f"  Done: {len(streaming_dialogues)} dialogues")

    # ── 4. Split ──
    print("\nSplitting data...")
    train, val, test = split_dialogues(
        streaming_dialogues, args.val_ratio, args.test_ratio, args.seed
    )
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # Thống kê labels per split
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        split_labels = [d["conversation_label"] for d in split]
        counts = {lbl: split_labels.count(lbl) for lbl in sorted(set(split_labels))}
        print(f"    {name}: {counts}")

    # Thống kê turn-level labels per split
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        turn_labels = [turn["turn_label"] for dlg in split for turn in dlg["turns"]]
        counts = {
            TURN_LABEL_NAMES[label_id]: turn_labels.count(label_id)
            for label_id in sorted(TURN_LABEL_NAMES)
        }
        print(f"    {name} turn labels: {counts}")

    # ── 5. Lưu ──
    os.makedirs(args.output_dir, exist_ok=True)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(args.output_dir, f"{split_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  Saved {split_name}: {path}")

    # ── 6. Preview ──
    print("\n-- Sample dialogue (train) --")
    if train:
        sample = train[0]
        print(f"  ID: {sample['dialogue_id']} | Label: {sample['conversation_label']} "
              f"| onset: {sample.get('scam_onset')}")
        for turn in sample["turns"][:6]:
            print(f"    Turn {turn['turn_id']} (speaker={turn['speaker']}, "
                  f"binary={turn['scam_label']}, "
                  f"label={turn['turn_label_name']}): {turn['text'][:60]}...")

    print("\nDone!")


if __name__ == "__main__":
    main()
