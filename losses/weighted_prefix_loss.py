"""
Noisy-OR + Weighted Prefix Loss.

Kết hợp 2 thành phần:

1) Noisy-OR dialogue-level BCE (main):
       L_noisy_or = BCE(p_dialogue, y)

2) Weighted prefix auxiliary trên cumulative probability p_t_agg:
       L_weighted = Σ w_t × BCE(p_t_agg, y)
   trong đó w_t = 2t/N  (t = 1..N, 1-based)

Total:
       L = L_noisy_or + λ × L_weighted

Numerical stability:
    - Clamp q_t vào [eps, 1-eps]
    - Tính Noisy-OR ở log-space
"""

import torch
import torch.nn.functional as F


def noisy_or_loss(
    turn_evidence_probs: list,
    dialogue_label: torch.Tensor,
    eps: float = 1e-6,
    weighted_lambda: float = 0.5,
) -> tuple:
    """
    Tính Noisy-OR loss + Weighted Prefix auxiliary cho 1 dialogue.

    Parameters
    ----------
    turn_evidence_probs : list of Tensor (scalar)
        q_t = sigmoid(s_t) cho từng turn.
    dialogue_label : Tensor (scalar)
        Nhãn dialogue-level: 0 (harmless) hoặc 1 (scam).
    eps : float
        Epsilon cho numerical stability.
    weighted_lambda : float
        Hệ số λ cho weighted prefix auxiliary loss. 0 = pure Noisy-OR.

    Returns
    -------
    loss : Tensor (scalar)
        Total loss = L_noisy_or + λ × L_weighted.
    p_dialogue : Tensor (scalar)
        Xác suất dialogue-level scam (= p_T_agg).
    """
    N = len(turn_evidence_probs)
    label = dialogue_label.float()

    # Stack tất cả q_t → [N]
    q = torch.stack(turn_evidence_probs).clamp(eps, 1 - eps)

    # ── Tính p_t_agg (cumulative Noisy-OR) tại mỗi prefix ──
    log_not_q = torch.log1p(-q)                     # [N]
    cumsum_log = torch.cumsum(log_not_q, dim=0)      # [N]
    p_agg = (1.0 - torch.exp(cumsum_log)).clamp(eps, 1 - eps)  # [N]

    # ── Main: Noisy-OR dialogue-level BCE ──
    p_dialogue = p_agg[-1]  # = p_T_agg
    loss_noisy_or = F.binary_cross_entropy(
        p_dialogue.unsqueeze(0), label.unsqueeze(0)
    )

    # ── Auxiliary: Weighted Prefix Loss ──
    loss_weighted = torch.tensor(0.0, device=q.device)
    if weighted_lambda > 0:
        # w_t = 2t/N,  t = 1..N  (1-based)
        t_indices = torch.arange(1, N + 1, dtype=torch.float32, device=q.device)
        weights = 2.0 * t_indices / N                   # [N]

        per_prefix_bce = F.binary_cross_entropy(
            p_agg, label.expand(N), reduction='none'
        )  # [N]
        loss_weighted = (weights * per_prefix_bce).sum()

    # ── Total Loss ──
    loss = loss_noisy_or + weighted_lambda * loss_weighted

    return loss, p_dialogue
