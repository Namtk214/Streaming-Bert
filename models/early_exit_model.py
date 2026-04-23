"""
EarlyExitWeightedModel — Main model cho Baseline2.

Kiến trúc:
  1. TurnEncoder: PhoBERT shared → mean pooling → h_t ∈ R^d
  2. CrossTurnAttention: query=h_t, kv=[h_1..h_{t-1}] → c_t ∈ R^d
  3. Fusion: z_t = concat(h_t, c_t) ∈ R^{2d}
     (turn 1: c_1 = zeros, nên z_1 = concat(h_1, zeros))
  4. Classifier: Linear(2d → C) → logits_t
  5. Weighted Loss: L = Σ (2t/N) * CE(logits_t, y)

Forward pass:
  - Flatten all turns trong batch → encode bằng PhoBERT
  - Regroup theo dialogue
  - Loop từng dialogue: cross-attn per turn → classifier → weighted loss
"""

import torch
import torch.nn as nn

from .turn_encoder import TurnEncoder
from .cross_turn_attention import CrossTurnAttention

import os
import sys

_baseline2_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _baseline2_dir not in sys.path:
    sys.path.insert(0, _baseline2_dir)

from losses.weighted_prefix_loss import weighted_cumulative_loss


class EarlyExitWeightedModel(nn.Module):
    """
    Early-Exit with Weighted Loss model.

    PhoBERT (frozen) + Cross-Turn Attention + Linear Classifier.
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

        # 3. Classifier: Linear(2d → C)
        self.dropout = nn.Dropout(config.head_dropout)
        self.classifier = nn.Linear(2 * self.hidden_dim, config.num_classes)

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
        labels : Tensor [B, T] (optional)
            Turn-level labels. Padding turns = -100.
        num_turns_list : list of int (optional)
            Số turn thật mỗi dialogue (tính từ turn_mask nếu None).

        Returns
        -------
        dict with:
            - 'loss': scalar (nếu labels != None)
            - 'all_turn_logits': list of list — logits mỗi turn mỗi dialogue
            - 'final_logits': Tensor [B, C] — logits turn cuối
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

        # 3. Per-dialogue: cross-attn + classifier + loss
        batch_all_logits = []
        batch_final_logits = []
        total_loss = torch.tensor(0.0, device=input_ids.device)

        for b in range(B):
            N = num_turns_list[b]
            H = all_embeddings[b, :N]  # [N, d]

            turn_logits = []
            for t in range(N):
                h_t = H[t]  # [d]

                # Cross-turn attention
                if t == 0:
                    c_t = torch.zeros_like(h_t)
                else:
                    c_t = self.cross_attn(h_t, H[:t])  # [d]

                # Fusion + classifier
                z_t = torch.cat([h_t, c_t], dim=0)  # [2d]
                z_t = self.dropout(z_t)
                logits_t = self.classifier(z_t)  # [C]
                turn_logits.append(logits_t)

            batch_all_logits.append(turn_logits)
            batch_final_logits.append(turn_logits[-1])

            # Weighted cumulative loss (per-turn labels)
            if labels is not None:
                dlg_labels = labels[b, :N].tolist()  # [N]
                dlg_loss = weighted_cumulative_loss(
                    turn_logits, dlg_labels, N
                )
                total_loss = total_loss + dlg_loss

        # Average loss across batch
        loss = None
        if labels is not None:
            loss = total_loss / B

        final_logits = torch.stack(batch_final_logits, dim=0)  # [B, C]

        return {
            "loss": loss,
            "all_turn_logits": batch_all_logits,
            "final_logits": final_logits,
        }

    @torch.no_grad()
    def encode_single_turn(self, input_ids, attention_mask,
                           h_prev_list=None):
        """
        Streaming inference: encode 1 turn mới.

        Parameters
        ----------
        input_ids : Tensor [1, L]
        attention_mask : Tensor [1, L]
        h_prev_list : list of Tensor [d] hoặc None
            Danh sách embeddings các turn trước.

        Returns
        -------
        logits : Tensor [C]
        probs : Tensor [C]
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

        # Classify
        z_t = torch.cat([h_t, c_t], dim=0)  # [2d]
        logits = self.classifier(z_t)  # [C]
        probs = torch.softmax(logits, dim=0)

        return logits, probs, h_t

    def get_param_groups(self, head_lr: float):
        """
        Tạo param groups. Chỉ bao gồm params có requires_grad = True.

        Khi freeze_encoder=True, chỉ có attention + classifier params.
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
