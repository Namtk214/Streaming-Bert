"""
Stateful Streaming Inference cho Baseline2: Early-Exit with Weighted Loss.

Mô phỏng inference online thực tế:
  - Nhận turn mới → encode PhoBERT → cross-attn với history → classify
  - Cache history embeddings [h_1...h_{t-1}] theo dialogue_id
  - Không update gradient

Streaming state chỉ cần lưu:
    H_prev = [h_1, ..., h_{t-1}]

Usage:
    engine = EarlyExitInferenceEngine(model_path="Baseline2/outputs/best_model")

    result = engine.predict_turn("dlg_001", "Tôi là công an!")
    print(result["prediction"], result["probabilities"])

    engine.reset("dlg_001")
"""

import os
import sys
import json
import math

import torch
from transformers import AutoTokenizer

# Fix import path
_baseline2_dir = os.path.dirname(os.path.abspath(__file__))
if _baseline2_dir not in sys.path:
    sys.path.insert(0, _baseline2_dir)

from config import EarlyExitConfig, LABEL_MAP, LABEL_NAMES


class EarlyExitInferenceEngine:
    """
    Stateful streaming inference engine cho Early-Exit model.

    Giữ history embeddings [h_1...h_{t-1}] theo dialogue_id.
    Mỗi turn mới chỉ cần 1 PhoBERT forward + 1 cross-attn step.
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

        # Load config
        config_json_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_json_path):
            with open(config_json_path, "r") as f:
                config_dict = json.load(f)
            self.config = EarlyExitConfig(**{
                k: v for k, v in config_dict.items()
                if k in EarlyExitConfig.__dataclass_fields__
            })
        else:
            self.config = EarlyExitConfig()

        self.threshold = threshold

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load model
        from models.early_exit_model import EarlyExitWeightedModel
        self.model = EarlyExitWeightedModel(self.config)
        model_state = os.path.join(model_path, "model.pt")
        if os.path.exists(model_state):
            self.model.load_state_dict(
                torch.load(model_state, map_location=self.device, weights_only=True)
            )
        self.model.to(self.device)
        self.model.eval()

        # Word segmenter (BẮT BUỘC — PhoBERT yêu cầu input word-segmented)
        if vncorenlp_dir is None:
            vncorenlp_dir = self.config.vncorenlp_dir
        from prepare_data import WordSegmenter
        self.segmenter = WordSegmenter(vncorenlp_dir)

        # History cache per dialogue: {dialogue_id: [h_1, h_2, ...]}
        self._history_cache = {}

        print(f"  EarlyExitInferenceEngine loaded on {self.device}")
        print(f"  Threshold: {self.threshold}")
        print(f"  Num classes: {self.config.num_classes}")

    def _preprocess(self, text: str) -> str:
        """Clean + word segment text (bắt buộc cho PhoBERT)."""
        import re
        import unicodedata

        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[\x00-\x09\x0b-\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Word segmentation (bắt buộc)
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
            ID hội thoại (dùng để cache history).
        text : str
            Nội dung turn (raw text, chưa segment).
        speaker : str
            "normal", "scammer", hoặc "unknown".

        Returns
        -------
        dict:
            - prediction: str (LEGIT/SCAM/AMBIGUOUS)
            - prediction_id: int
            - probabilities: dict {label_name: probability}
            - is_scam: bool
            - turn_index: int
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

        # Get cached history
        if dialogue_id in self._history_cache:
            h_prev_list = self._history_cache[dialogue_id]
            turn_idx = len(h_prev_list) + 1
        else:
            h_prev_list = []
            turn_idx = 1

        # Forward (no grad)
        logits, probs, h_t = self.model.encode_single_turn(
            input_ids, attention_mask, h_prev_list if h_prev_list else None
        )

        # Prediction
        pred_id = int(logits.argmax().item())
        pred_name = LABEL_NAMES.get(pred_id, "UNKNOWN")

        # Probabilities dict
        probs_np = probs.cpu().numpy()
        prob_dict = {
            LABEL_NAMES[i]: round(float(probs_np[i]), 4)
            for i in range(len(probs_np))
        }

        # Is scam?
        scam_prob = prob_dict.get("SCAM", 0.0)
        is_scam = scam_prob >= self.threshold

        # Update history cache
        if dialogue_id not in self._history_cache:
            self._history_cache[dialogue_id] = []
        self._history_cache[dialogue_id].append(h_t.detach())

        return {
            "dialogue_id": dialogue_id,
            "turn_index": turn_idx,
            "prediction": pred_name,
            "prediction_id": pred_id,
            "probabilities": prob_dict,
            "is_scam": bool(is_scam),
            "speaker": speaker,
            "text_preview": text[:80],
        }

    def predict_conversation(self, messages: list,
                             dialogue_id: str = "temp") -> list:
        """
        Dự đoán toàn bộ hội thoại theo streaming (turn by turn).

        Parameters
        ----------
        messages : list of dict
            Mỗi dict có keys: "text", "speaker_role"
        dialogue_id : str

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
        """Reset history cho 1 dialogue."""
        if dialogue_id in self._history_cache:
            del self._history_cache[dialogue_id]

    def reset_all(self):
        """Reset toàn bộ cache."""
        self._history_cache.clear()

    def get_active_dialogues(self) -> list:
        """Danh sách dialogue đang active."""
        return list(self._history_cache.keys())


# ============================================================
# Demo
# ============================================================
def demo():
    """Demo streaming inference cho Early-Exit model."""
    print("=" * 60)
    print("EARLY-EXIT STREAMING INFERENCE DEMO")
    print("=" * 60)

    cfg = EarlyExitConfig()
    model_path = os.path.join(cfg.output_dir, "best_model")

    if not os.path.exists(os.path.join(model_path, "model.pt")):
        print(f"\n  Model chưa train! File không tồn tại: {model_path}/model.pt")
        print("  Chạy train.py trước.")
        print("\n  Ví dụ sử dụng:")
        print("""
    engine = EarlyExitInferenceEngine(
        model_path="Baseline2/outputs/best_model",
        vncorenlp_dir="vncorenlp",
        threshold=0.5,
    )

    messages = [
        {"speaker_role": "normal", "text": "Alo ai đấy?"},
        {"speaker_role": "scammer", "text": "Tôi là công an, bạn đang bị điều tra!"},
        {"speaker_role": "normal", "text": "Cái gì ạ?"},
        {"speaker_role": "scammer", "text": "Chuyển tiền ngay!"},
    ]

    results = engine.predict_conversation(messages, "dlg_001")
    for r in results:
        print(f"  Turn {r['turn_index']}: {r['prediction']} "
              f"scam_prob={r['probabilities'].get('SCAM', 0):.3f}")
        """)
        return

    # Real demo
    engine = EarlyExitInferenceEngine(
        model_path=model_path,
        vncorenlp_dir=cfg.vncorenlp_dir,
        threshold=0.5,
    )

    # Test scam conversation
    scam_messages = [
        {"speaker_role": "normal", "text": "Alo ai đấy ạ?"},
        {"speaker_role": "scammer",
         "text": "Tôi là Đại úy Nguyễn Văn Hùng, Công an thành phố. "
                 "Bạn đang bị điều tra vì liên quan đến đường dây rửa tiền."},
        {"speaker_role": "normal", "text": "Cái gì ạ? Tôi không biết gì cả."},
        {"speaker_role": "scammer",
         "text": "Chuyển tiền vào tài khoản an toàn ngay trong vòng 30 phút, "
                 "nếu không sẽ bị bắt."},
    ]

    print("\n  -- Test SCAM conversation --")
    results = engine.predict_conversation(scam_messages, "test_scam")
    for r in results:
        alert = "[!] SCAM!" if r["is_scam"] else "[OK]"
        print(f"  Turn {r['turn_index']}: {r['prediction']:10s} "
              f"scam_p={r['probabilities'].get('SCAM', 0):.3f} {alert}")

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
        print(f"  Turn {r['turn_index']}: {r['prediction']:10s} "
              f"scam_p={r['probabilities'].get('SCAM', 0):.3f} {alert}")


if __name__ == "__main__":
    demo()
