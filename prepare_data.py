"""
Chuyển đổi raw data → processed format cho Early-Exit with Noisy-OR Loss.

Quy trình:
  1. Đọc raw JSON files (data/train.json, data/val.json, data/test.json)
  2. Clean text + word segmentation (VnCoreNLP) — bắt buộc cho PhoBERT
  3. Map dialogue-level label: "harmless" → 0, "scam" → 1
  4. Ghi processed JSON files vào Streaming-Bert/data/

Input format mỗi dialogue:
  {
    "_id": 2079,
    "turns": [
      {"turn_idx": 0, "role": "người gọi", "content": "A lô?", ...},
      ...
    ],
    "label": "harmless",
    "sample_id": "harmless-2079"
  }

Output format mỗi dialogue:
  {
    "dialogue_id": "harmless-2079",
    "dialogue_label": 0,
    "dialogue_label_name": "harmless",
    "num_turns": 10,
    "turns": [
      {"turn_id": 1, "role": "người gọi", "text": "A lô?", "text_segmented": "A lô ?"},
      ...
    ]
  }
"""

import json
import os
import argparse
import re
import sys
import unicodedata
from typing import Dict, List

# Import config
_baseline2_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _baseline2_dir)

from config import EarlyExitConfig, LABEL_MAP


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
    Uses VNCORENLP_CACHE singleton to avoid JVM double-start in Colab.
    """

    def __init__(self, vncorenlp_dir: str):
        import py_vncorenlp
        from config import VNCORENLP_CACHE

        abs_dir = os.path.abspath(vncorenlp_dir)

        # Reuse cached instance if JVM already started for this dir
        if abs_dir in VNCORENLP_CACHE:
            self.segmenter = VNCORENLP_CACHE[abs_dir]
            print(f"  VnCoreNLP reused from cache ({abs_dir})")
            return

        if not os.path.exists(os.path.join(vncorenlp_dir, "VnCoreNLP-1.2.jar")):
            print(f"  Đang tải VnCoreNLP models vào {vncorenlp_dir}...")
            os.makedirs(vncorenlp_dir, exist_ok=True)
            py_vncorenlp.download_model(save_dir=vncorenlp_dir)

        self.segmenter = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"],
            save_dir=vncorenlp_dir,
        )
        # Store in singleton cache
        VNCORENLP_CACHE[abs_dir] = self.segmenter
        print(f"  VnCoreNLP loaded OK (cached as {abs_dir})")

    def segment(self, text: str) -> str:
        result = self.segmenter.word_segment(text)
        if isinstance(result, list):
            return " ".join(result)
        return result


# ============================================================
# Convert raw → processed format
# ============================================================
def convert_dialogue(raw_dialogue: Dict, segmenter: WordSegmenter) -> Dict:
    """Chuyển 1 dialogue từ raw format thành processed format."""
    raw_turns = raw_dialogue["turns"]
    label_str = raw_dialogue["label"]
    sample_id = raw_dialogue.get("sample_id", str(raw_dialogue.get("_id", "")))

    # Map dialogue-level label
    label_id = LABEL_MAP.get(label_str)
    if label_id is None:
        raise ValueError(f"Unknown label: {label_str}. Expected one of {list(LABEL_MAP.keys())}")

    turns = []
    for i, turn in enumerate(raw_turns):
        text_raw = turn.get("content", "")
        text_clean = clean_text(text_raw)
        text_segmented = segmenter.segment(text_clean)

        role = turn.get("role", "unknown")

        turns.append({
            "turn_id": i + 1,
            "role": role,
            "text": text_clean,
            "text_segmented": text_segmented,
        })

    return {
        "dialogue_id": sample_id,
        "dialogue_label": label_id,
        "dialogue_label_name": label_str,
        "num_turns": len(turns),
        "turns": turns,
    }


# ============================================================
# Main pipeline
# ============================================================
def parse_args() -> argparse.Namespace:
    cfg = EarlyExitConfig()
    parser = argparse.ArgumentParser(
        description="Preprocess raw dialogue data: clean + word segment for PhoBERT."
    )
    parser.add_argument(
        "--raw-data-dir",
        default=cfg.raw_data_dir,
        help="Directory containing raw train.json, val.json, test.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=cfg.data_dir,
        help="Output directory for processed files.",
    )
    parser.add_argument(
        "--vncorenlp-dir",
        default=cfg.vncorenlp_dir,
        help="VnCoreNLP model directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Fix console encoding
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("PREPARE DATA — Early-Exit with Noisy-OR Loss")
    print("=" * 60)

    print(f"  Raw data dir:  {args.raw_data_dir}")
    print(f"  Output dir:    {args.output_dir}")

    # 1. Word segmentation
    print("\nInitializing word segmenter...")
    segmenter = WordSegmenter(args.vncorenlp_dir)

    # 2. Process each split
    splits = ["train", "val", "test"]
    os.makedirs(args.output_dir, exist_ok=True)

    for split_name in splits:
        input_path = os.path.join(args.raw_data_dir, f"{split_name}.json")
        output_path = os.path.join(args.output_dir, f"{split_name}.json")

        if not os.path.exists(input_path):
            print(f"\n  SKIP: {input_path} not found")
            continue

        print(f"\n  Processing {split_name}...")
        with open(input_path, "r", encoding="utf-8") as f:
            raw_dialogues = json.load(f)

        print(f"    Total dialogues: {len(raw_dialogues)}")

        # Thống kê labels
        label_counts = {}
        for d in raw_dialogues:
            lbl = d.get("label", "UNKNOWN")
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        print(f"    Label distribution: {label_counts}")

        # Filter: chỉ giữ dialogues có label hợp lệ
        valid_labels = set(LABEL_MAP.keys())
        valid_dialogues = [d for d in raw_dialogues if d.get("label") in valid_labels]
        if len(valid_dialogues) < len(raw_dialogues):
            print(f"    After filtering: {len(valid_dialogues)} dialogues")

        # Convert
        processed = []
        for i, raw_dlg in enumerate(valid_dialogues):
            dlg = convert_dialogue(raw_dlg, segmenter)
            processed.append(dlg)
            if (i + 1) % 100 == 0:
                print(f"    Converted {i + 1}/{len(valid_dialogues)}")
        print(f"    Done: {len(processed)} dialogues")

        # Thống kê processed labels
        label_dist = {}
        for d in processed:
            name = d["dialogue_label_name"]
            label_dist[name] = label_dist.get(name, 0) + 1
        print(f"    Processed label distribution: {label_dist}")

        # Turn stats
        all_num_turns = [d["num_turns"] for d in processed]
        print(f"    Turns: min={min(all_num_turns)} max={max(all_num_turns)} "
              f"avg={sum(all_num_turns)/len(all_num_turns):.1f}")

        # Save
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)
        print(f"    Saved: {output_path}")

    # Preview
    train_path = os.path.join(args.output_dir, "train.json")
    if os.path.exists(train_path):
        with open(train_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        if train_data:
            sample = train_data[0]
            print(f"\n-- Sample dialogue (train) --")
            print(f"  ID: {sample['dialogue_id']} | "
                  f"Label: {sample['dialogue_label_name']} ({sample['dialogue_label']}) | "
                  f"Turns: {sample['num_turns']}")
            for turn in sample["turns"][:4]:
                print(f"    Turn {turn['turn_id']} ({turn['role']}): "
                      f"{turn['text'][:60]}...")
                print(f"      segmented: {turn['text_segmented'][:60]}...")

    print("\nDone!")


if __name__ == "__main__":
    main()
