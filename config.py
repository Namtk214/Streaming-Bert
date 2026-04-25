from dataclasses import dataclass, field
import os

STREAMING_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(STREAMING_ROOT)

# Shared JVM singleton guard — abs_path → py_vncorenlp.VnCoreNLP instance.
# JVM can only be started once per process; all modules must share this dict.
VNCORENLP_CACHE: dict = {}

# Speaker roles in raw data (handle common typos via normalize_role())
# 0 = caller ("người gọi"), 1 = listener ("người nghe"), 2 = unknown
ROLE_MAP = {"người gọi": 0, "người nghe": 1}

# Conversation-level labels
LABEL_MAP = {"harmless": 0, "scam": 1}
LABEL_NAMES = {0: "harmless", 1: "scam"}


@dataclass
class StreamingConfig:
    """Hyperparameters cho streaming binary scam detection."""

    # PhoBERT turn encoder
    model_name: str = "vinai/phobert-base-v2"
    max_tokens_per_turn: int = 256

    # GRU conversation encoder
    gru_hidden_size: int = 256
    gru_num_layers: int = 1

    # Classifier — 1 logit/turn, max-pool → dialogue logit, BCEWithLogitsLoss
    num_classes: int = 1

    # Regularization
    head_dropout: float = 0.1

    # Optimizer
    encoder_lr: float = 2e-5
    rnn_head_lr: float = 1e-4
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    grad_clip: float = 1.0
    warmup_ratio: float = 0.1

    # Training
    num_epochs: int = 10
    batch_size: int = 4

    # Inference
    threshold: float = 0.5
    seed: int = 42

    # Paths
    raw_data_dir: str = field(
        default_factory=lambda: os.path.join(PROJECT_ROOT, "data")
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
