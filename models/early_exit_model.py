"""
EarlyExitWeightedModel — Main model cho Baseline2 (Noisy-OR Loss).

Kiến trúc:
  1. TurnEncoder: PhoBERT shared → mean pooling → h_t ∈ R^d
  2. CrossTurnAttention: query=h_t, kv=[h_1..h_{t-1}] → c_t ∈ R^d
  3. Fusion: z_t = concat(h_t, c_t) ∈ R^{2d}
     (turn 1: c_1 = zeros, nên z_1 = concat(h_1, zeros))
  4. Evidence head: Linear(2d → 1) → scalar logit s_t
  5. q_t = sigmoid(s_t) — per-turn evidence probability
  6. Noisy-OR: p_dialogue = 1 - ∏(1 - q_t)
  7. Loss: BCE(p_dialogue, y_dialogue)

Forward pass:
  - Flatten all turns trong batch → encode bằng PhoBERT
  - Regroup theo dialogue
  - Loop từng dialogue: cross-attn per turn → evidence head → Noisy-OR → BCE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .turn_encoder import TurnEncoder
from .cross_turn_attention import CrossTurnAttention

import os
import sys

_baseline2_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _baseline2_dir not in sys.path:
    sys.path.insert(0, _baseline2_dir)

from losses.weighted_prefix_loss import noisy_or_loss


class EarlyExitWeightedModel(nn.Module):
    """
    Early-Exit with Noisy-OR Loss model.

    PhoBERT (frozen) + Cross-Turn Attention + Binary Evidence Head.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Turn encoder (PhoBERT)
        self.turn_encoder = TurnEncoder(
            model_name=config.model_name,
            freeze=config.freeze_encoder,
        )
        self.hidden_dim = self.turn_encoder.hidden_size  # 768

        # 2. Cross-turn attention
        self.cross_attn = CrossTurnAttention(
            hidden_dim=self.hidden_dim,
            num_heads=config.attn_num_heads,
            dropout=config.attn_dropout,
        )

        # 3. Evidence head: Linear(2d → 1) → scalar logit
        self.dropout = nn.Dropout(config.head_dropout)
        self.evidence_head = nn.Linear(2 * self.hidden_dim, 1)

    def forward(self, input_ids, attention_mask, turn_mask,
                labels=None, num_turns_list=None):
        """
        Forward pass cho batch of dialogues.

        Parameters
        ----------
        input_ids : Tensor [B, T, L]
            Token IDs, padded theo max turns trong batch.
        attention_mask : Tensor [B, T, L]
            Token-level attention mask.
        turn_mask : Tensor [B, T]
            1 = turn thật, 0 = turn padding.
        labels : Tensor [B] (optional)
            Dialogue-level labels. 0 = harmless, 1 = scam.
        num_turns_list : list of int (optional)
            Số turn thật mỗi dialogue (tính từ turn_mask nếu None).

        Returns
        -------
        dict with:
            - 'loss': scalar (nếu labels != None)
            - 'all_turn_q': list of list — q_t (evidence prob) mỗi turn mỗi dialogue
            - 'all_turn_p_agg': list of list — p_t_agg (cumulative prob) mỗi dialogue
            - 'p_dialogue': Tensor [B] — dialogue-level scam probability
        """
        B, T, L = input_ids.shape

        # 1. Flatten all turns → encode
        input_ids_flat = input_ids.view(B * T, L)
        attention_mask_flat = attention_mask.view(B * T, L)

        # Encode tất cả turns (PhoBERT, frozen)
        with torch.no_grad() if self.config.freeze_encoder else _dummy_ctx():
            all_embeddings = self.turn_encoder(
                input_ids_flat, attention_mask_flat
            )  # [B*T, d]

        all_embeddings = all_embeddings.view(B, T, -1)  # [B, T, d]

        # 2. Tính num_turns từ turn_mask nếu chưa có
        if num_turns_list is None:
            num_turns_list = turn_mask.sum(dim=1).long().tolist()

        # 3. Per-dialogue: cross-attn + evidence head + Noisy-OR
        batch_all_q = []        # per-turn evidence probs
        batch_all_p_agg = []    # per-turn cumulative probs
        batch_p_dialogue = []   # dialogue-level probs
        total_loss = torch.tensor(0.0, device=input_ids.device)

        eps = self.config.eps

        for b in range(B):
            N = num_turns_list[b]
            H = all_embeddings[b, :N]  # [N, d]

            turn_q = []      # evidence probabilities
            turn_p_agg = []  # cumulative probabilities

            p_agg = torch.tensor(0.0, device=input_ids.device)

            for t in range(N):
                h_t = H[t]  # [d]

                # Cross-turn attention
                if t == 0:
                    c_t = torch.zeros_like(h_t)
                else:
                    c_t = self.cross_attn(h_t, H[:t])  # [d]

                # Fusion + evidence head
                z_t = torch.cat([h_t, c_t], dim=0)  # [2d]
                z_t = self.dropout(z_t)
                s_t = self.evidence_head(z_t).squeeze(-1)  # scalar logit
                q_t = torch.sigmoid(s_t)  # scalar in (0, 1)

                turn_q.append(q_t)

                # Online Noisy-OR update: p_t_agg = 1 - (1 - p_{t-1}) * (1 - q_t)
                p_agg = 1.0 - (1.0 - p_agg) * (1.0 - q_t)
                turn_p_agg.append(p_agg)

            batch_all_q.append(turn_q)
            batch_all_p_agg.append(turn_p_agg)

            # Dialogue-level probability (= p_T_agg, last cumulative)
            p_dlg = turn_p_agg[-1] if turn_p_agg else torch.tensor(0.0, device=input_ids.device)
            batch_p_dialogue.append(p_dlg)

            # Noisy-OR loss (per-dialogue)
            if labels is not None:
                dlg_loss, _ = noisy_or_loss(turn_q, labels[b], eps=eps)
                total_loss = total_loss + dlg_loss

        # Average loss across batch
        loss = None
        if labels is not None:
            loss = total_loss / B

        p_dialogue = torch.stack(batch_p_dialogue)  # [B]

        return {
            "loss": loss,
            "all_turn_q": batch_all_q,        # list of list[Tensor scalar]
            "all_turn_p_agg": batch_all_p_agg, # list of list[Tensor scalar]
            "p_dialogue": p_dialogue,           # [B]
        }

    @torch.no_grad()
    def encode_single_turn(self, input_ids, attention_mask,
                           h_prev_list=None, p_agg_prev=0.0):
        """
        Streaming inference: encode 1 turn mới.

        Parameters
        ----------
        input_ids : Tensor [1, L]
        attention_mask : Tensor [1, L]
        h_prev_list : list of Tensor [d] hoặc None
            Danh sách embeddings các turn trước.
        p_agg_prev : float
            Cumulative Noisy-OR probability tính đến turn trước.

        Returns
        -------
        q_t : float — evidence probability cho turn hiện tại
        p_agg : float — cumulative scam probability (Noisy-OR)
        h_t : Tensor [d] — embedding turn mới (để append vào history)
        """
        # Encode turn
        h_t = self.turn_encoder(input_ids, attention_mask)  # [1, d]
        h_t = h_t.squeeze(0)  # [d]

        # Cross-turn attention
        if h_prev_list is None or len(h_prev_list) == 0:
            c_t = torch.zeros_like(h_t)
        else:
            H_prev = torch.stack(h_prev_list, dim=0)  # [t-1, d]
            c_t = self.cross_attn(h_t, H_prev)

        # Evidence head
        z_t = torch.cat([h_t, c_t], dim=0)  # [2d]
        s_t = self.evidence_head(z_t).squeeze(-1)  # scalar
        q_t = torch.sigmoid(s_t).item()  # float

        # Online Noisy-OR update
        p_agg = 1.0 - (1.0 - p_agg_prev) * (1.0 - q_t)

        return q_t, p_agg, h_t

    def get_param_groups(self, head_lr: float):
        """
        Tạo param groups. Chỉ bao gồm params có requires_grad = True.

        Khi freeze_encoder=True, chỉ có attention + evidence_head params.
        """
        trainable_params = [
            p for p in self.parameters() if p.requires_grad
        ]
        return [{"params": trainable_params, "lr": head_lr}]

    def count_trainable_params(self) -> int:
        """Đếm số lượng trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _dummy_ctx:
    """Dummy context manager khi không freeze encoder."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
