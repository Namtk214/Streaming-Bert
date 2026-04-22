"""
Stateful Streaming Inference cho Scam Detection.

Mô phỏng inference online thực tế:
  - Nhận turn mới → encode → GRU step → dự đoán
  - Cache hidden state theo dialogue_id
  - Không update gradient

Usage:
    engine = StreamingInferenceEngine(model_path="streaming/outputs/best_model")

    # Turn mới đến
    result = engine.predict_turn("dlg_001", "Tôi là công an!", speaker="scammer")
    print(result["probability"], result["is_scam"])

    # Reset dialogue
    engine.reset("dlg_001")
"""

import os
import sys
import json

import torch
from transformers import AutoTokenizer

# Đảm bảo import từ streaming/ thay vì src/
_streaming_dir = os.path.dirname(os.path.abspath(__file__))
if _streaming_dir not in sys.path:
    sys.path.insert(0, _streaming_dir)

from config import StreamingConfig, SPEAKER_MAP
from model import StreamingScamDetector

# [Fix] Allowed safe globals cho PyTorch 2.6+ trên Colab (nơi ép weights_only=True)
try:
    torch.serialization.add_safe_globals([StreamingConfig])
except AttributeError:
    pass


class StreamingInferenceEngine:
    """
    Stateful streaming inference engine.

    Giữ hidden state theo dialogue_id,
    mỗi turn mới chỉ cần forward 1 lần PhoBERT + 1 GRU step.
    """

    def __init__(
        self,
        model_path: str,
        vncorenlp_dir: str = None,
        threshold: float = 0.5,
        device: str = None,
    ):
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load config (JSON format, compatible with PyTorch 2.6+)
        config_json_path = os.path.join(model_path, "config.json")
        config_pt_path = os.path.join(model_path, "config.pt")
        if os.path.exists(config_json_path):
            with open(config_json_path, "r") as f:
                config_dict = json.load(f)
            self.config = StreamingConfig(**{
                k: v for k, v in config_dict.items()
                if k in StreamingConfig.__dataclass_fields__
            })
        elif os.path.exists(config_pt_path):
            # Fallback for old checkpoints saved with torch.save
            self.config = torch.load(config_pt_path, map_location="cpu", weights_only=False)
        else:
            self.config = StreamingConfig()

        self.threshold = threshold

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load model
        self.model = StreamingScamDetector(self.config)
        model_state = os.path.join(model_path, "model.pt")
        if os.path.exists(model_state):
            self.model.load_state_dict(
                torch.load(model_state, map_location=self.device, weights_only=True)
            )
        self.model.to(self.device)
        self.model.eval()

        # Word segmenter (optional)
        self.segmenter = None
        if vncorenlp_dir:
            try:
                from prepare_streaming_data import WordSegmenter
                self.segmenter = WordSegmenter(vncorenlp_dir)
            except Exception:
                pass

        # Hidden state cache per dialogue
        self._state_cache = {}

        print(f"  StreamingInferenceEngine loaded on {self.device}")
        print(f"  Threshold: {self.threshold}")

    def _preprocess(self, text: str) -> str:
        """Clean + word segment text."""
        import re
        import unicodedata

        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[\x00-\x09\x0b-\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        if self.segmenter and self.segmenter.segmenter is not None:
            text = self.segmenter.segment(text)

        return text

    def predict_turn(
        self,
        dialogue_id: str,
        text: str,
        speaker: str = "normal",
    ) -> dict:
        """
        Dự đoán 1 turn mới.

        Parameters
        ----------
        dialogue_id : str
            ID hội thoại (dùng để cache hidden state).
        text : str
            Nội dung turn (raw text, chưa segment).
        speaker : str
            "normal", "scammer", hoặc "unknown".

        Returns
        -------
        dict:
            - probability: float, xác suất scam
            - is_scam: bool, có vượt ngưỡng không
            - turn_index: int, số thứ tự turn trong dialogue này
            - dialogue_id: str
        """
        # Preprocess
        text_processed = self._preprocess(text)

        # Tokenize
        encoding = self.tokenizer(
            text_processed,
            max_length=self.config.max_tokens_per_turn,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Get cached state
        if dialogue_id in self._state_cache:
            h_prev = self._state_cache[dialogue_id]["hidden"]
            turn_idx = self._state_cache[dialogue_id]["turn_index"] + 1
        else:
            h_prev = None
            turn_idx = 1

        # Forward (no grad)
        logit, h_new = self.model.encode_single_turn(
            input_ids, attention_mask, h_prev
        )

        # Sigmoid
        import math
        prob = 1.0 / (1.0 + math.exp(-logit))

        is_scam = prob >= self.threshold

        # Cache state
        self._state_cache[dialogue_id] = {
            "hidden": h_new,
            "turn_index": turn_idx,
        }

        return {
            "dialogue_id": dialogue_id,
            "turn_index": turn_idx,
            "probability": round(prob, 4),
            "is_scam": bool(is_scam),
            "speaker": speaker,
            "text_preview": text,
        }

    def predict_conversation(self, messages: list, dialogue_id: str = "temp") -> list:
        """
        Dự đoán toàn bộ hội thoại theo streaming (turn by turn).

        Parameters
        ----------
        messages : list of dict
            Mỗi dict có keys: "text", "speaker_role"
        dialogue_id : str
            ID hội thoại

        Returns
        -------
        list of dict: kết quả dự đoán cho từng turn
        """
        self.reset(dialogue_id)
        results = []

        for msg in messages:
            result = self.predict_turn(
                dialogue_id=dialogue_id,
                text=msg["text"],
                speaker=msg.get("speaker_role", "unknown"),
            )
            results.append(result)

        return results

    def reset(self, dialogue_id: str):
        """Reset hidden state cho 1 dialogue."""
        if dialogue_id in self._state_cache:
            del self._state_cache[dialogue_id]

    def reset_all(self):
        """Reset toàn bộ cache."""
        self._state_cache.clear()

    def get_active_dialogues(self) -> list:
        """Danh sách dialogue đang active."""
        return list(self._state_cache.keys())


# ============================================================
# Demo
# ============================================================
def demo():
    """Demo streaming inference."""
    print("=" * 60)
    print("STREAMING INFERENCE DEMO")
    print("=" * 60)

    # Check model exists
    cfg = StreamingConfig()
    model_path = os.path.join(cfg.output_dir, "best_model")

    if not os.path.exists(os.path.join(model_path, "model.pt")):
        print(f"\n  Model chưa train! File không tồn tại: {model_path}/model.pt")
        print("  Chạy train.py trước.")
        print("\n  Demo với fake predictions thay thế...\n")

        # Fake demo
        print("  Ví dụ sử dụng:")
        print("""
    engine = StreamingInferenceEngine(
        model_path="streaming/outputs/best_model",
        vncorenlp_dir="vncorenlp",
        threshold=0.5,
    )

    # Scam conversation
    messages = [
        {"speaker_role": "normal", "text": "Alo ai đấy?"},
        {"speaker_role": "scammer", "text": "Tôi là công an, bạn đang bị điều tra!"},
        {"speaker_role": "normal", "text": "Cái gì ạ?"},
        {"speaker_role": "scammer", "text": "Chuyển tiền ngay!"},
    ]

    results = engine.predict_conversation(messages, "dlg_001")
    for r in results:
        alert = "[!] SCAM!" if r["is_scam"] else "[OK]"
        print(f"  Turn {r['turn_index']}: prob={r['probability']:.3f} {alert}")
        """)
        return

    # Real demo
    engine = StreamingInferenceEngine(
        model_path=model_path,
        vncorenlp_dir=cfg.vncorenlp_dir,
        threshold=cfg.threshold,
    )

    # Test scam conversation
    scam_messages = [
        {"speaker_role": "normal", "text": "Alo ai đấy ạ?"},
        {"speaker_role": "scammer",
         "text": "Tôi là Đại úy Nguyễn Văn Hùng, Công an thành phố. Bạn đang bị điều tra vì liên quan đến đường dây rửa tiền."},
        {"speaker_role": "normal", "text": "Cái gì ạ? Tôi không biết gì cả."},
        {"speaker_role": "scammer",
         "text": "Chuyển tiền vào tài khoản an toàn ngay trong vòng 30 phút, nếu không sẽ bị bắt."},
    ]

    print("\n  -- Test SCAM conversation --")
    results = engine.predict_conversation(scam_messages, "test_scam")
    for r in results:
        alert = "[!] SCAM!" if r["is_scam"] else "[OK]"
        print(f"  Turn {r['turn_index']}: {r['text_preview'][:40]}... "
              f"prob={r['probability']:.3f} {alert}")

    # Test legit conversation
    legit_messages = [
        {"speaker_role": "normal", "text": "Alo bạn ơi, chiều nay đi cà phê không?"},
        {"speaker_role": "normal", "text": "Oke, mấy giờ?"},
        {"speaker_role": "normal", "text": "3 giờ nhé, quán trên Nguyễn Huệ."},
        {"speaker_role": "normal", "text": "Được rồi, hẹn gặp!"},
    ]

    print("\n  -- Test LEGIT conversation --")
    results = engine.predict_conversation(legit_messages, "test_legit")
    for r in results:
        alert = "[!] SCAM!" if r["is_scam"] else "[OK]"
        print(f"  Turn {r['turn_index']}: {r['text_preview'][:40]}... "
              f"prob={r['probability']:.3f} {alert}")


if __name__ == "__main__":
    demo()
