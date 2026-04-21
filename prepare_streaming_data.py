"""
Chuyển đổi raw conversations → streaming format cho training.

Quy trình:
  1. Đọc raw_conversations.json + synthetic_conversations.json
  2. Clean text + word segmentation (VnCoreNLP)
  3. Gán binary scam label theo prefix rule:
     - SCAM/AMBIGUOUS: label=1 từ turn scammer có tactic đầu tiên
     - LEGIT: toàn bộ label=0
  4. Chia train/val/test theo conversation_id
  5. Lưu streaming format JSON

Output format mỗi dialogue:
  {
    "dialogue_id": "...",
    "turns": [
      {"turn_id": 1, "speaker": 0, "text": "...",
       "text_segmented": "...", "scam_label": 0},
      ...
    ]
  }
"""

import json
import os
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
# Binary label generation theo prefix rule
# ============================================================
def assign_prefix_labels(conversation: Dict) -> List[int]:
    """
    Gán binary scam label theo prefix rule.

    SCAM/AMBIGUOUS:
      - Tìm turn đầu tiên có t4_labels (bằng chứng scam)
      - Nếu không có t4_labels, tìm turn đầu tiên speaker=scammer
      - Từ turn đó trở đi: label=1

    LEGIT:
      - Toàn bộ label=0
    """
    messages = conversation["messages"]
    t1_label = conversation["t1_label"]
    num_turns = len(messages)

    if t1_label == "LEGIT":
        return [0] * num_turns

    # Tìm scam onset: ưu tiên turn có t4_labels
    scam_onset = None

    # Cách 1: turn đầu tiên có t4_labels
    for i, msg in enumerate(messages):
        if msg.get("t4_labels") and len(msg["t4_labels"]) > 0:
            scam_onset = i
            break

    # Cách 2 (fallback): turn đầu tiên speaker=scammer
    if scam_onset is None:
        for i, msg in enumerate(messages):
            if msg["speaker_role"] == "scammer":
                scam_onset = i
                break

    # Cách 3 (fallback cuối): coi như scam từ turn giữa
    if scam_onset is None:
        scam_onset = num_turns // 2

    labels = [0] * scam_onset + [1] * (num_turns - scam_onset)
    return labels


# ============================================================
# Convert raw → streaming format
# ============================================================
def convert_to_streaming(conversation: Dict, segmenter: WordSegmenter) -> Dict:
    """Chuyển 1 conversation thành streaming format."""
    messages = conversation["messages"]
    prefix_labels = assign_prefix_labels(conversation)

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
            "scam_label": prefix_labels[i],
        })

    return {
        "dialogue_id": conversation["conversation_id"],
        "conversation_label": conversation["t1_label"],
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
def main():
    cfg = StreamingConfig()

    print("=" * 60)
    print("PREPARE STREAMING DATA")
    print("=" * 60)

    # ── 1. Đọc raw data ──
    all_conversations = []

    # Original data
    if os.path.exists(cfg.raw_data_path):
        with open(cfg.raw_data_path, "r", encoding="utf-8") as f:
            original = json.load(f)
        all_conversations.extend(original)
        print(f"  Original data: {len(original)} conversations")

    # Synthetic data
    synth_path = os.path.join(cfg.streaming_data_dir, "synthetic_conversations.json")
    if os.path.exists(synth_path):
        with open(synth_path, "r", encoding="utf-8") as f:
            synthetic = json.load(f)
        all_conversations.extend(synthetic)
        print(f"  Synthetic data: {len(synthetic)} conversations")

    print(f"  Total: {len(all_conversations)} conversations")

    # Thống kê
    labels = [c["t1_label"] for c in all_conversations]
    for lbl in sorted(set(labels)):
        print(f"    {lbl}: {labels.count(lbl)}")

    # ── 2. Word segmentation ──
    print("\nInitializing word segmenter...")
    segmenter = WordSegmenter(cfg.vncorenlp_dir)

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
        streaming_dialogues, cfg.val_ratio, cfg.test_ratio, cfg.seed
    )
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # Thống kê labels per split
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        split_labels = [d["conversation_label"] for d in split]
        counts = {lbl: split_labels.count(lbl) for lbl in sorted(set(split_labels))}
        print(f"    {name}: {counts}")

    # ── 5. Lưu ──
    os.makedirs(cfg.streaming_data_dir, exist_ok=True)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(cfg.streaming_data_dir, f"{split_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  Saved {split_name}: {path}")

    # ── 6. Preview ──
    print("\n-- Sample dialogue (train) --")
    if train:
        sample = train[0]
        print(f"  ID: {sample['dialogue_id']} | Label: {sample['conversation_label']}")
        for turn in sample["turns"][:3]:
            print(f"    Turn {turn['turn_id']} (speaker={turn['speaker']}, "
                  f"scam={turn['scam_label']}): {turn['text'][:60]}...")

    print("\nDone!")


if __name__ == "__main__":
    main()
