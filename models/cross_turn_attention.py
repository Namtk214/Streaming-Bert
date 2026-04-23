"""
CrossTurnAttention — Cross-turn attention module.

Với turn t >= 2:
    c_t = MultiheadAttention(query=h_t, key=H_prev, value=H_prev)
    trong đó H_prev = [h_1, ..., h_{t-1}]

Với turn 1:
    c_1 = zero vector (không có history)

Output:
    c_t ∈ R^d (cùng dimension với h_t)
"""

import torch
import torch.nn as nn


class CrossTurnAttention(nn.Module):
    """Multi-head cross-turn attention.

    Query = current turn embedding h_t
    Key/Value = all previous turn embeddings [h_1, ..., h_{t-1}]
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, h_t: torch.Tensor,
                h_prev: torch.Tensor) -> torch.Tensor:
        """
        Cross-turn attention cho 1 turn.

        Parameters
        ----------
        h_t : Tensor [d]
            Current turn embedding.
        h_prev : Tensor [t-1, d]
            Previous turn embeddings (history).

        Returns
        -------
        Tensor [d] — context vector c_t.

        Notes
        -----
        Nếu h_prev rỗng (turn 1), trả về zero vector.
        """
        if h_prev.shape[0] == 0:
            return torch.zeros_like(h_t)

        # Reshape cho nn.MultiheadAttention (batch_first=True)
        query = h_t.unsqueeze(0).unsqueeze(0)      # [1, 1, d]
        kv = h_prev.unsqueeze(0)                    # [1, t-1, d]

        c_t, _ = self.attn(query, kv, kv)           # [1, 1, d]
        return c_t.squeeze(0).squeeze(0)            # [d]
