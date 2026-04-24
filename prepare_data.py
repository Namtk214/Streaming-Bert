"""
Convert raw data (PROJECT_ROOT/data/) → streaming format (STREAMING_ROOT/data/).

Raw format:
  {_id, label, source_split, turns: [{turn_idx, role, content, tactic_tags, ...}]}

Streaming format:
  {
    "dialogue_id": "1234",
    "conversation_label": "scam" | "harmless",
    "turns": [
      {
        "turn_id": 1,
        "speaker": 0 | 1 | 2,
        "text": "...",
        "text_segmented": "...",
        "turn_label": 0 | 1
      }, ...
    ]
  }

Turn label strategy (binary):
  - harmless conversation → every turn = 0
  - scam conversation     → every turn = 1
"""

import json
import os
import re
import sys
import unicodedata

_streaming_dir = os.path.dirname(os.path.abspath(__file__))
if _streaming_dir not in sys.path:
    sys.path.insert(0, _streaming_dir)

from config import StreamingConfig, LABEL_MAP

cfg = StreamingConfig()

# ── Text utilities ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\x00-\x09\x0b-\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Speaker role normalisation ──────────────────────────────────

def normalize_role(role: str) -> int:
    """
    Map raw role strings to speaker IDs, tolerating OCR / encoding typos.
      0 = caller  (contains 'gọi' or 'goi')
      1 = listener (contains 'nghe')
      2 = unknown
    """
    if not role:
        return 2
    normalized = unicodedata.normalize("NFC", role).lower()
    # tolerate typos: 'nguelle gọi', 'nguoi goi', etc.
    if "gọi" in normalized or "goi" in normalized:
        return 0
    if "nghe" in normalized:
        return 1
    return 2


# ── Word segmenter (VnCoreNLP bắt buộc) ───────────────────────

class WordSegmenter:
    """Wrap VnCoreNLP wseg. Bắt buộc — raise nếu không load được."""

    def __init__(self, vncorenlp_dir: str):
        import py_vncorenlp
        jar = os.path.join(vncorenlp_dir, "VnCoreNLP-1.2.jar")
        if not os.path.exists(jar):
            print(f"  Downloading VnCoreNLP into {vncorenlp_dir} ...")
            os.makedirs(vncorenlp_dir, exist_ok=True)
            py_vncorenlp.download_model(save_dir=vncorenlp_dir)
        self.segmenter = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"], save_dir=vncorenlp_dir
        )
        print("  VnCoreNLP loaded OK")

    def segment(self, text: str) -> str:
        result = self.segmenter.word_segment(text)
        return " ".join(result) if isinstance(result, list) else result


# ── Conversion ──────────────────────────────────────────────────

def convert_dialogue(raw: dict, segmenter: WordSegmenter) -> dict:
    """Convert one raw conversation to streaming format."""
    label_str = raw["label"]           # "scam" or "harmless"
    turn_label = LABEL_MAP[label_str]  # 0 or 1

    turns = []
    for t in raw["turns"]:
        text = clean_text(t["content"])
        turns.append({
            "turn_id":        t["turn_idx"] + 1,   # 1-based
            "speaker":        normalize_role(t["role"]),
            "text":           text,
            "text_segmented": segmenter.segment(text),
            "turn_label":     turn_label,
        })

    return {
        "dialogue_id":        str(raw["_id"]),
        "conversation_label": label_str,
        "turns":              turns,
    }


# ── Main ────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert raw JSON data to Streaming-Bert format."
    )
    parser.add_argument(
        "--raw-dir",
        default=cfg.raw_data_dir,
        help="Folder with raw train.json / val.json / test.json",
    )
    parser.add_argument(
        "--out-dir",
        default=cfg.streaming_data_dir,
        help="Output folder for streaming-format JSON files",
    )
    parser.add_argument(
        "--vncorenlp-dir",
        default=cfg.vncorenlp_dir,
        help="VnCoreNLP model directory (optional)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PREPARE STREAMING DATA")
    print("=" * 60)
    print(f"  Raw data:    {args.raw_dir}")
    print(f"  Output dir:  {args.out_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    print("\nInitialising word segmenter...")
    segmenter = WordSegmenter(args.vncorenlp_dir)

    for split in ("train", "val", "test"):
        src = os.path.join(args.raw_dir, f"{split}.json")
        dst = os.path.join(args.out_dir, f"{split}.json")

        if not os.path.exists(src):
            print(f"  SKIP {split}: {src} not found")
            continue

        with open(src, encoding="utf-8") as f:
            raw_data = json.load(f)

        converted = []
        for raw in raw_data:
            converted.append(convert_dialogue(raw, segmenter))

        with open(dst, "w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)

        labels = [d["conversation_label"] for d in converted]
        scam_n = labels.count("scam")
        harmless_n = labels.count("harmless")
        print(
            f"  {split:5s}: {len(converted):5d} dialogues  "
            f"(scam={scam_n}, harmless={harmless_n})  → {dst}"
        )

    # Show a short sample
    sample_path = os.path.join(args.out_dir, "train.json")
    if os.path.exists(sample_path):
        with open(sample_path, encoding="utf-8") as f:
            samples = json.load(f)
        print("\n-- Sample (train[0]) --")
        s = samples[0]
        print(f"  id={s['dialogue_id']}  label={s['conversation_label']}")
        for t in s["turns"][:4]:
            print(
                f"  Turn {t['turn_id']} (spk={t['speaker']}, lbl={t['turn_label']}): "
                f"{t['text'][:60]}..."
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
