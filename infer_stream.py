"""
Stateful Streaming Inference cho Binary Scam Detection.

Mô phỏng inference online thực tế:
  - Nhận turn mới → VnCoreNLP segment → PhoBERT encode → GRU step → p_t
  - Cache hidden state theo dialogue_id
  - is_scam = p_≤t ≥ threshold  (Noisy-OR prefix score, nhất quán với training)
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
import math
import os
import re
import sys
import unicodedata

import torch
from transformers import AutoTokenizer

_streaming_dir = os.path.dirname(os.path.abspath(__file__))
if _streaming_dir not in sys.path:
    sys.path.insert(0, _streaming_dir)

from config import StreamingConfig, VNCORENLP_CACHE
from model import StreamingScamDetector

ROLE_CALLER   = "người gọi"
ROLE_LISTENER = "người nghe"

try:
    torch.serialization.add_safe_globals([StreamingConfig])
except AttributeError:
    pass


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\x00-\x09\x0b-\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _missing_vncorenlp_files(vncorenlp_dir: str):
    required = [
        os.path.join(vncorenlp_dir, "VnCoreNLP-1.2.jar"),
        os.path.join(vncorenlp_dir, "models", "wordsegmenter", "vi-vocab"),
        os.path.join(vncorenlp_dir, "models", "wordsegmenter", "wordsegmenter.rdr"),
    ]
    return [p for p in required if not os.path.exists(p)]


class InferenceWordSegmenter:
    """Strict VnCoreNLP loader for inference. Never downloads files."""

    def __init__(self, vncorenlp_dir: str):
        abs_dir = os.path.abspath(vncorenlp_dir)
        if abs_dir in VNCORENLP_CACHE:
            self.segmenter = VNCORENLP_CACHE[abs_dir]
            print("  VnCoreNLP reused OK")
            return

        missing = _missing_vncorenlp_files(abs_dir)
        if missing:
            missing_rel = [os.path.relpath(p, abs_dir) for p in missing]
            raise FileNotFoundError(
                "VnCoreNLP directory is missing required files: "
                f"{missing_rel}. This inference loader never downloads files; "
                "pass the folder that already contains VnCoreNLP."
            )

        import py_vncorenlp

        self.segmenter = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"], save_dir=abs_dir
        )
        VNCORENLP_CACHE[abs_dir] = self.segmenter
        print("  VnCoreNLP loaded OK")

    def segment(self, text: str) -> str:
        result = self.segmenter.word_segment(text)
        return " ".join(result) if isinstance(result, list) else result


class StreamingInferenceEngine:
    """
    Stateful streaming inference (binary: harmless / scam).

    Giữ GRU hidden state theo dialogue_id.
    Mỗi turn chỉ cần 1 lần forward PhoBERT + 1 GRU step.
    """

    def __init__(
        self,
        model_path: str,
        vncorenlp_dir: str = None,
        threshold: float = 0.5,
        device: str = None,
        segmenter=None,
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

        if segmenter is not None:
            print("  Using existing VnCoreNLP word segmenter")
            self.segmenter = segmenter
        else:
            if vncorenlp_dir is None:
                vncorenlp_dir = self.config.vncorenlp_dir
            print("  Loading VnCoreNLP word segmenter...")
            self.segmenter = InferenceWordSegmenter(vncorenlp_dir)

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

        _LOG_EPS = 1e-7

        cache        = self._state_cache.get(dialogue_id)
        h_prev       = cache["hidden"]           if cache else None
        turn_idx     = cache["turn_index"] + 1   if cache else 1
        log_comp_prev = cache["log_complement"]  if cache else 0.0  # log(1 - p_≤(t-1))

        q_t, h_new = self.model.encode_single_turn(input_ids, attention_mask, h_prev)

        # Noisy-OR prefix score: p_≤t = 1 − ∏_{i=1}^{t}(1−q_i)
        q_clamped     = min(max(q_t, _LOG_EPS), 1.0 - _LOG_EPS)
        log_comp_new  = log_comp_prev + math.log(1.0 - q_clamped)
        prefix_prob   = 1.0 - math.exp(log_comp_new)

        is_scam = prefix_prob >= self.threshold

        self._state_cache[dialogue_id] = {
            "hidden":         h_new,
            "turn_index":     turn_idx,
            "log_complement": log_comp_new,
        }

        return {
            "dialogue_id":     dialogue_id,
            "turn_index":      turn_idx,
            "predicted_label": "scam" if is_scam else "harmless",
            "prob_scam":       round(prefix_prob, 4),
            "prob_harmless":   round(1.0 - prefix_prob, 4),
            "probability":     round(prefix_prob, 4),  # compat với visualize
            "turn_prob":       round(q_t, 4),          # raw q_t để debug
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
