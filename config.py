from dataclasses import dataclass, field
import os

STREAMING_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(STREAMING_ROOT)

SPEAKER_MAP = {"normal": 0, "scammer": 1, "unknown": 2}


@dataclass
class StreamingConfig:
    """Hyperparameters cho streaming binary scam detection."""

    # PhoBERT turn encoder
    model_name: str = "vinai/phobert-base-v2"
    max_tokens_per_turn: int = 128

    # GRU conversation encoder
    gru_hidden_size: int = 256
    gru_num_layers: int = 1

    # Classifier
    num_classes: int = 3  # 0=LEGIT, 1=SCAM, 2=AMBIGUOUS

    # Regularization
    head_dropout: float = 0.2

    # Optimizer
    encoder_lr: float = 2e-5
    rnn_head_lr: float = 1e-4
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    grad_clip: float = 1.0
    warmup_ratio: float = 0.1

    # Training
    num_epochs: int = 10
    batch_size: int = 2

    # Data split
    seed: int = 42
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Inference
    threshold: float = 0.5

    # Paths
    streaming_data_dir: str = field(
        default_factory=lambda: os.path.join(STREAMING_ROOT, "data")
    )
    output_dir: str = field(
        default_factory=lambda: os.path.join(STREAMING_ROOT, "outputs")
    )
    vncorenlp_dir: str = field(
        default_factory=lambda: os.path.join(PROJECT_ROOT, "vncorenlp")
    )
