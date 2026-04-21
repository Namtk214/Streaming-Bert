"""
Cấu hình cho Streaming Binary Scam Detection.

Kiến trúc: PhoBERT per-turn encoder + uni-GRU + binary classifier head.
Hyperparameters theo khuyến nghị từ implementation guide.
"""

from dataclasses import dataclass, field
import os

# ============================================================
# Đường dẫn gốc
# ============================================================
STREAMING_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(STREAMING_ROOT)

# ============================================================
# Speaker encoding
# ============================================================
SPEAKER_MAP = {"normal": 0, "scammer": 1, "unknown": 2}
NUM_SPEAKERS = 3


@dataclass
class StreamingConfig:
    """Hyperparameters cho streaming binary scam detection."""

    # ── PhoBERT Turn Encoder ──
    model_name: str = "vinai/phobert-base-v2"
    max_tokens_per_turn: int = 128

    # ── Conversation Encoder (GRU) ──
    gru_hidden_size: int = 256
    gru_num_layers: int = 1

    # ── Speaker Embedding ──
    num_speakers: int = NUM_SPEAKERS
    speaker_embed_dim: int = 16

    # ── Regularization ──
    head_dropout: float = 0.2

    # ── Optimizer ──
    encoder_lr: float = 1e-5
    rnn_head_lr: float = 1e-4
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    grad_clip: float = 1.0
    warmup_ratio: float = 0.1

    # ── Training ──
    num_epochs: int = 10
    batch_size: int = 4  # số dialogues mỗi batch

    # ── Staged Training ──
    stage_a_epochs: int = 3   # Freeze PhoBERT hoàn toàn
    stage_b_epochs: int = 3   # Unfreeze top 2 layers
    # Epoch còn lại: unfreeze top 4 layers (nếu val cải thiện)

    # ── Data Split ──
    seed: int = 42
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # ── Inference ──
    threshold: float = 0.5

    # ── Paths ──
    raw_data_path: str = field(
        default_factory=lambda: os.path.join(PROJECT_ROOT, "data", "raw_conversations.json")
    )
    streaming_data_dir: str = field(
        default_factory=lambda: os.path.join(STREAMING_ROOT, "data")
    )
    output_dir: str = field(
        default_factory=lambda: os.path.join(STREAMING_ROOT, "outputs")
    )
    vncorenlp_dir: str = field(
        default_factory=lambda: os.path.join(PROJECT_ROOT, "vncorenlp")
    )
