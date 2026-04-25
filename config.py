"""
Configuration cho Baseline2: Early-Exit with Noisy-OR Loss.

Kiến trúc:
  turn text → PhoBERT (frozen) → mean pooling → h_t
  → Cross-Turn Attention (cho t >= 2) → c_t
  → concat(h_t, c_t) → Linear(2d → 1) → scalar logit s_t
  → q_t = sigmoid(s_t)  (per-turn evidence probability)
  → Noisy-OR aggregation: p_dialogue = 1 - ∏(1 - q_t)
  → BCE(p_dialogue, y_dialogue)
"""

from dataclasses import dataclass, field
import os

BASELINE2_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASELINE2_ROOT)

# Label map: dialogue-level binary classification
LABEL_MAP = {"harmless": 0, "scam": 1}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}

# VnCoreNLP singleton cache — prevents JVM double-start in Colab/notebooks.
# Key = absolute path to vncorenlp_dir, Value = py_vncorenlp.VnCoreNLP instance.
# Both WordSegmenter (prepare_data.py) and InferenceEngine (infer_stream.py)
# import this dict and reuse the same instance.
VNCORENLP_CACHE: dict = {}


@dataclass
class EarlyExitConfig:
    """Hyperparameters cho Early-Exit with Noisy-OR Loss baseline."""

    # PhoBERT turn encoder
    model_name: str = "vinai/phobert-base-v2"
    max_tokens_per_turn: int = 128
    freeze_encoder: bool = True

    # Cross-turn attention
    attn_num_heads: int = 8
    attn_dropout: float = 0.1

    # Regularization
    head_dropout: float = 0.2

    # Noisy-OR numerical stability
    eps: float = 1e-6

    # Optimizer
    head_lr: float = 2e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    grad_clip: float = 1.0
    warmup_ratio: float = 0.1

    # Training
    num_epochs: int = 15
    batch_size: int = 2
    patience: int = 5

    # Data split (used by prepare_data.py if needed)
    seed: int = 42
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Paths
    raw_data_dir: str = field(
        default_factory=lambda: os.path.join(PROJECT_ROOT, "data")
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
