"""
Configuration cho Baseline2: Early-Exit with Weighted Loss.

Kiến trúc:
  turn text → PhoBERT (frozen) → mean pooling → h_t
  → Cross-Turn Attention (cho t >= 2) → c_t
  → concat(h_t, c_t) → Linear classifier → logits
  → Weighted Cumulative CE Loss: L = Σ (2t/N) * CE(p_t, y)
"""

from dataclasses import dataclass, field
import os

BASELINE2_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASELINE2_ROOT)

# Label map: dialogue-level classification
LABEL_MAP = {"LEGIT": 0, "SCAM": 1, "AMBIGUOUS": 2}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


@dataclass
class EarlyExitConfig:
    """Hyperparameters cho Early-Exit with Weighted Loss baseline."""

    # PhoBERT turn encoder
    model_name: str = "vinai/phobert-base-v2"
    max_tokens_per_turn: int = 128
    freeze_encoder: bool = True

    # Cross-turn attention
    attn_num_heads: int = 8
    attn_dropout: float = 0.1

    # Classification
    num_classes: int = 3  # LEGIT, SCAM, AMBIGUOUS

    # Regularization
    head_dropout: float = 0.2

    # Optimizer
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    grad_clip: float = 1.0
    warmup_ratio: float = 0.1

    # Training
    num_epochs: int = 15
    batch_size: int = 4
    patience: int = 5

    # Data split
    seed: int = 42
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Paths
    raw_data_path: str = field(
        default_factory=lambda: os.path.join(
            PROJECT_ROOT, "data", "excel_raw_conversations.json"
        )
    )
    data_dir: str = field(
        default_factory=lambda: os.path.join(BASELINE2_ROOT, "data")
    )
    output_dir: str = field(
        default_factory=lambda: os.path.join(BASELINE2_ROOT, "outputs")
    )
    vncorenlp_dir: str = field(
        default_factory=lambda: os.path.join(PROJECT_ROOT, "vncorenlp")
    )
