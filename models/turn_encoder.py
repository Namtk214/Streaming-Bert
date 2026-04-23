"""
TurnEncoder — PhoBERT shared encoder cho từng turn.

Encode mỗi turn riêng bằng PhoBERT, lấy turn embedding
bằng masked mean pooling trên chiều token.

Input:  list turn texts (tokenized)
Output: embeddings h_1...h_N, mỗi h_t có shape [d]
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class TurnEncoder(nn.Module):
    """PhoBERT per-turn encoder với masked mean pooling."""

    def __init__(self, model_name: str = "vinai/phobert-base-v2",
                 freeze: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size  # 768

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def masked_mean_pool(self, token_hidden: torch.Tensor,
                         attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Masked mean pooling trên chiều token.

        Parameters
        ----------
        token_hidden : Tensor [N, L, H]
            Hidden states từ PhoBERT.
        attention_mask : Tensor [N, L]
            1 = token thật, 0 = padding.

        Returns
        -------
        Tensor [N, H]
        """
        mask_expanded = attention_mask.unsqueeze(-1).float()       # [N, L, 1]
        sum_hidden = (token_hidden * mask_expanded).sum(dim=1)     # [N, H]
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)        # [N, 1]
        return sum_hidden / sum_mask

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode batch of turns.

        Parameters
        ----------
        input_ids : Tensor [N, L]
            Token IDs cho N turns.
        attention_mask : Tensor [N, L]

        Returns
        -------
        Tensor [N, H] — turn embeddings.
        """
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        token_hidden = encoder_output.last_hidden_state  # [N, L, H]
        return self.masked_mean_pool(token_hidden, attention_mask)  # [N, H]
