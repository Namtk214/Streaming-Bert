"""
Stateful Streaming Inference cho Binary Scam Detection.

Mô phỏng inference online thực tế:
  - Nhận turn mới → VnCoreNLP segment → PhoBERT encode → GRU step → p_t
  - Cache hidden state theo dialogue_id
  - is_scam = p_t ≥ threshold  (real-time alert tại mỗi turn)
  - VnCoreNLP bắt buộc

Usage:
    engine = StreamingInferenceEngine(
        model_path="Streaming-Bert/outputs/best_model",
        vncorenlp_dir="vncorenlp",
    )

    result = engine.predict_turn("dlg_001", "Tôi là công an!", speaker=ROLE_CALLER)
    print(result["prob_scam"], result["is_scam"])

    engine.reset("dlg_001")
"""

import json
import os
import sys

import torch
from transformers import AutoTokenizer

_streaming_dir = os.path.dirname(os.path.abspath(__file__))
if _streaming_dir not in sys.path:
    sys.path.insert(0, _streaming_dir)

from config import StreamingConfig
from model import StreamingScamDetector
from prepare_data import WordSegmenter, clean_text

ROLE_CALLER   = "người gọi"
ROLE_LISTENER = "người nghe"

try:
    torch.serialization.add_safe_globals([StreamingConfig])
except AttributeError:
    pass


class StreamingInferenceEngine:
    """
    Stateful streaming inference (binary: harmless / scam).

    Giữ GRU hidden state theo dialogue_id.
    Mỗi turn chỉ cần 1 lần forward PhoBERT + 1 GRU step.
    """

    def __init__(
        self,
        model_path: str,
        vncorenlp_dir: str,
        threshold: float = 0.5,
        device: str = None,
    ):
        if device:
            resolved_device = device
        elif torch.cuda.is_available():
            resolved_device = "cuda"
        else:
            resolved_device = "cpu"
        self.device = torch.device(resolved_device)

        # Load config
        config_json = os.path.join(model_path, "config.json")
        config_pt   = os.path.join(model_path, "config.pt")
        if os.path.exists(config_json):
            with open(config_json) as f:
                cfg_dict = json.load(f)
            self.config = StreamingConfig(**{
                k: v for k, v in cfg_dict.items()
                if k in StreamingConfig.__dataclass_fields__
            })
        elif os.path.exists(config_pt):
            self.config = torch.load(config_pt, map_location="cpu", weights_only=False)
        else:
            self.config = StreamingConfig()

        self.threshold = threshold
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = StreamingScamDetector(self.config)
        model_state = os.path.join(model_path, "model.pt")
        if os.path.exists(model_state):
            self.model.load_state_dict(
                torch.load(model_state, map_location=self.device, weights_only=True)
            )
        self.model.to(self.device)
        self.model.eval()

        # VnCoreNLP bắt buộc
        print("  Loading VnCoreNLP word segmenter...")
        self.segmenter = WordSegmenter(vncorenlp_dir)

        self._state_cache: dict = {}
        print(f"  StreamingInferenceEngine ready on {self.device}")
        print(f"  Threshold: {self.threshold}")

    # ── Preprocessing ──────────────────────────────────────────

    def _preprocess(self, text: str) -> str:
        return self.segmenter.segment(clean_text(text))

    # ── Predict single turn ────────────────────────────────────

    def predict_turn(
        self,
        dialogue_id: str,
        text: str,
        speaker: str = ROLE_CALLER,
    ) -> dict:
        """
        Dự đoán 1 turn mới.

        Parameters
        ----------
        dialogue_id : str
        text        : str  – raw text (chưa segment)
        speaker     : str  – ROLE_CALLER | ROLE_LISTENER

        Returns
        -------
        dict:
            prob_scam     float  – σ(logit_t) ∈ [0,1]
            prob_harmless float
            is_scam       bool
            predicted_label str  – "scam" | "harmless"
            turn_index    int    – 1-based
            dialogue_id   str
            text_preview  str
        """
        encoding = self.tokenizer(
            self._preprocess(text),
            max_length=self.config.max_tokens_per_turn,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        cache    = self._state_cache.get(dialogue_id)
        h_prev   = cache["hidden"]      if cache else None
        turn_idx = cache["turn_index"] + 1 if cache else 1

        prob_scam, h_new = self.model.encode_single_turn(input_ids, attention_mask, h_prev)
        is_scam = prob_scam >= self.threshold

        self._state_cache[dialogue_id] = {"hidden": h_new, "turn_index": turn_idx}

        return {
            "dialogue_id":     dialogue_id,
            "turn_index":      turn_idx,
            "predicted_label": "scam" if is_scam else "harmless",
            "prob_scam":       round(prob_scam, 4),
            "prob_harmless":   round(1.0 - prob_scam, 4),
            "probability":     round(prob_scam, 4),  # compat với visualize
            "is_scam":         bool(is_scam),
            "speaker":         speaker,
            "text_preview":    text,
        }

    # ── Predict full conversation ──────────────────────────────

    def predict_conversation(self, messages: list, dialogue_id: str = "temp") -> list:
        """
        Dự đoán toàn bộ hội thoại theo streaming (turn by turn).

        Parameters
        ----------
        messages : list of dict  – mỗi dict có "text" và "speaker_role" (optional)
        """
        self.reset(dialogue_id)
        return [
            self.predict_turn(
                dialogue_id=dialogue_id,
                text=msg["text"],
                speaker=msg.get("speaker_role", ROLE_CALLER),
            )
            for msg in messages
        ]

    # ── State management ───────────────────────────────────────

    def reset(self, dialogue_id: str):
        """Reset hidden state cho 1 dialogue."""
        self._state_cache.pop(dialogue_id, None)

    def reset_all(self):
        """Reset toàn bộ cache."""
        self._state_cache.clear()

    def get_active_dialogues(self) -> list:
        return list(self._state_cache.keys())


# ── Demo ───────────────────────────────────────────────────────

def demo():
    print("=" * 60)
    print("STREAMING INFERENCE DEMO")
    print("=" * 60)

    cfg = StreamingConfig()
    model_path = os.path.join(cfg.output_dir, "best_model")

    if not os.path.exists(os.path.join(model_path, "model.pt")):
        print("\n  Model chưa train! Chạy train.py trước.")
        print(f"  Expected: {model_path}/model.pt")
        return

    engine = StreamingInferenceEngine(
        model_path=model_path,
        vncorenlp_dir=cfg.vncorenlp_dir,
        threshold=cfg.threshold,
    )

    scam_msgs = [
        {"speaker_role": ROLE_LISTENER, "text": "Đây có phải anh Nam không ạ?"},
        {"speaker_role": ROLE_CALLER,  "text": "Đúng rồi, ai đầu dây vậy?"},
        {"speaker_role": ROLE_LISTENER,
         "text": "Em là nhân viên ngân hàng BIDV. Thẻ của anh sắp hết hạn, "
                 "anh cần cung cấp số thẻ và CVV để gia hạn ngay hôm nay."},
        {"speaker_role": ROLE_CALLER,  "text": "Thẻ tôi vẫn dùng được mà?"},
        {"speaker_role": ROLE_LISTENER,
         "text": "Hệ thống mới cập nhật, nếu không gia hạn ngay thẻ sẽ bị khóa."},
    ]

    print("\n  -- Test SCAM --")
    for r in engine.predict_conversation(scam_msgs, "demo_scam"):
        tag = "[SCAM]" if r["is_scam"] else "[OK]  "
        print(f"  Turn {r['turn_index']} {tag} p={r['prob_scam']:.3f} | {r['text_preview'][:55]}")

    harmless_msgs = [
        {"speaker_role": ROLE_LISTENER, "text": "A lô, có bưu phẩm của anh đây ạ."},
        {"speaker_role": ROLE_CALLER,  "text": "Ừ, anh để ở đâu vậy?"},
        {"speaker_role": ROLE_LISTENER, "text": "Em để ở phòng bảo vệ rồi ạ, anh xuống lấy nhé."},
        {"speaker_role": ROLE_CALLER,  "text": "OK cảm ơn em."},
    ]

    print("\n  -- Test HARMLESS --")
    for r in engine.predict_conversation(harmless_msgs, "demo_harmless"):
        tag = "[SCAM]" if r["is_scam"] else "[OK]  "
        print(f"  Turn {r['turn_index']} {tag} p={r['prob_scam']:.3f} | {r['text_preview'][:55]}")


if __name__ == "__main__":
    demo()
