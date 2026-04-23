"""
Weighted Cumulative Cross-Entropy Loss (turn-level labels).

L = Σ_{t=1..N} w_t * CE(p_t, y_t)

trong đó:
  w_t = 2t / N (1-based indexing)
  N = số turn của hội thoại
  p_t = predicted logits tại turn t
  y_t = turn-level label tại turn t (khác nhau mỗi turn)
"""

import torch
import torch.nn.functional as F


def weighted_cumulative_loss(
    turn_logits: list,
    turn_labels: list,
    N: int,
) -> torch.Tensor:
    """
    Tính weighted cumulative CE loss cho 1 dialogue với per-turn labels.

    Parameters
    ----------
    turn_logits : list of Tensor
        Mỗi phần tử là logits [C] cho 1 turn.
    turn_labels : list of int or Tensor
        Label cho từng turn (LEGIT=0, SCAM=1, AMBIGUOUS=2).
    N : int
        Tổng số turn thật của dialogue.

    Returns
    -------
    Tensor scalar — weighted loss.
    """
    total_loss = torch.tensor(0.0, device=turn_logits[0].device)

    for i, logits_t in enumerate(turn_logits):
        w = 2.0 * (i + 1) / N
        label_t = turn_labels[i]
        if isinstance(label_t, int):
            label_t = torch.tensor([label_t], device=logits_t.device)
        elif label_t.dim() == 0:
            label_t = label_t.unsqueeze(0)

        ce = F.cross_entropy(logits_t.unsqueeze(0), label_t)
        total_loss = total_loss + w * ce

    return total_loss
