"""
Noisy-OR Loss (dialogue-level supervision).

Gộp per-turn evidence probabilities q_t bằng Noisy-OR:
    p_dialogue = 1 - ∏_{t=1..T}(1 - q_t)

Loss:
    L = BCE(p_dialogue, y_dialogue)

Numerical stability:
    - Clamp q_t vào [eps, 1-eps]
    - Tính ở log-space: log_not_p = Σ log(1 - q_t)
"""

import torch
import torch.nn.functional as F


def noisy_or_loss(
    turn_evidence_probs: list,
    dialogue_label: torch.Tensor,
    eps: float = 1e-6,
) -> tuple:
    """
    Tính Noisy-OR loss cho 1 dialogue.

    Parameters
    ----------
    turn_evidence_probs : list of Tensor (scalar)
        q_t = sigmoid(s_t) cho từng turn. Mỗi phần tử là scalar tensor.
    dialogue_label : Tensor (scalar)
        Nhãn dialogue-level: 0 (harmless) hoặc 1 (scam).
    eps : float
        Epsilon cho numerical stability.

    Returns
    -------
    loss : Tensor (scalar)
        BCE loss giữa p_dialogue và y.
    p_dialogue : Tensor (scalar)
        Xác suất dialogue-level scam (Noisy-OR aggregated).
    """
    # Stack tất cả q_t → [T]
    q = torch.stack(turn_evidence_probs).clamp(eps, 1 - eps)

    # Log-space computation cho numerical stability
    log_not_p = torch.log1p(-q).sum()
    p_dialogue = 1.0 - torch.exp(log_not_p)

    # Clamp p_dialogue để tránh log(0) trong BCE
    p_dialogue = p_dialogue.clamp(eps, 1 - eps)

    # BCE loss
    label = dialogue_label.float()
    loss = F.binary_cross_entropy(p_dialogue.unsqueeze(0), label.unsqueeze(0))

    return loss, p_dialogue
