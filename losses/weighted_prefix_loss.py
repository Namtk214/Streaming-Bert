"""
Weighted Cumulative Cross-Entropy Loss.

L = Σ_{t=1..N} w_t * CE(p_t, y)

trong đó:
  w_t = 2t / N (1-based indexing)
  N = số turn của hội thoại
  p_t = predicted logits tại turn t
  y = dialogue-level label
"""

import torch
import torch.nn.functional as F


def weighted_cumulative_loss(
    turn_logits: list,
    label: torch.Tensor,
    N: int,
) -> torch.Tensor:
    """
    Tính weighted cumulative CE loss cho 1 dialogue.

    Parameters
    ----------
    turn_logits : list of Tensor
        Mỗi phần tử là logits [C] cho 1 turn.
    label : Tensor scalar
        Dialogue-level label (int).
    N : int
        Tổng số turn thật của dialogue.

    Returns
    -------
    Tensor scalar — weighted loss.
    """
    total_loss = torch.tensor(0.0, device=turn_logits[0].device)
    label_expanded = label.unsqueeze(0)  # [1]

    for i, logits_t in enumerate(turn_logits):
        w = 2.0 * (i + 1) / N
        ce = F.cross_entropy(logits_t.unsqueeze(0), label_expanded)
        total_loss = total_loss + w * ce

    return total_loss
