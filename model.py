"""
Streaming Scam Detector – PhoBERT + uni-GRU + Noisy-OR MIL loss.

Kiến trúc:
  turn text u_t
    → PhoBERT turn encoder
    → masked mean pooling → e_t (768-dim)
    → uni-GRU → h_t (hidden_dim)
    → Linear(hidden, 1) + sigmoid → p_t ∈ [0,1]

Training (weak supervision với dialogue-level label):
  p_dialogue = 1 − ∏_{t=1}^{T} (1 − p_t)   [Noisy-OR]
  loss = BCE(p_dialogue, y_dialogue)

  Tại sao Noisy-OR thay vì max-pool:
  - Mọi turn đều nhận gradient (không chỉ turn có logit lớn nhất)
  - Soft aggregation: nhiều turn đáng ngờ → xác suất dialogue tăng dần
  - Gradient ∂p_d/∂p_t = ∏_{s≠t}(1−p_s) → turn nào còn "room" đều được update

Inference (streaming):
  Nhận từng turn, trả p_t = σ(logit_t).
  is_scam = p_t ≥ threshold  (real-time alert)
"""

import torch
import torch.nn as nn
from transformers import AutoModel

_LOG_EPS = 1e-7   # clamp để tránh log(0)


class StreamingScamDetector(nn.Module):
    """PhoBERT per-turn encoder + uni-GRU + Noisy-OR binary head."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = AutoModel.from_pretrained(config.model_name)
        self.encoder_hidden_size = self.encoder.config.hidden_size  # 768
        # Freeze PhoBERT parameters (chỉ train GRU + head)
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.gru = nn.GRU(
            input_size=self.encoder_hidden_size,
            hidden_size=config.gru_hidden_size,
            num_layers=config.gru_num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )

        self.dropout    = nn.Dropout(config.head_dropout)
        self.classifier = nn.Linear(config.gru_hidden_size, 1)  # 1 logit/turn

        self.loss_fn = nn.BCELoss()   # input đã là probability

    # ── Helpers ────────────────────────────────────────────────

    def masked_mean_pool(self, token_hidden, attention_mask):
        """[N, L, H] → [N, H] via masked mean."""
        mask_exp = attention_mask.unsqueeze(-1).float()
        return (token_hidden * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)

    @staticmethod
    def noisy_or(turn_probs: torch.Tensor, turn_mask: torch.Tensor) -> torch.Tensor:
        """
        Noisy-OR aggregation (log-space để tránh underflow).

        p_dialogue = 1 − exp( Σ_{real turns} log(1 − p_t) )

        Parameters
        ----------
        turn_probs : [B, T]   sigmoid probabilities per turn
        turn_mask  : [B, T]   1=real turn, 0=padding

        Returns
        -------
        [B]  dialogue-level probability
        """
        p_clamped = turn_probs.clamp(_LOG_EPS, 1.0 - _LOG_EPS)
        log_complement = torch.log(1.0 - p_clamped)    # [B, T]
        # padding turns đóng góp log(1) = 0 → không ảnh hưởng tích
        log_complement = log_complement * turn_mask
        log_prod = log_complement.sum(dim=1)            # [B]
        return 1.0 - torch.exp(log_prod)                # [B]

    # ── Training forward ───────────────────────────────────────

    def forward(self, input_ids, attention_mask, turn_mask, dialogue_labels=None):
        """
        Parameters
        ----------
        input_ids        [B, T, L]
        attention_mask   [B, T, L]
        turn_mask        [B, T]    1=real, 0=padding
        dialogue_labels  [B]       0/1 float, optional

        Returns
        -------
        dict:
          loss           scalar | None
          turn_probs     [B, T]   σ(logit_t) per turn
          dialogue_probs [B]      Noisy-OR aggregated probability
        """
        B, T, L = input_ids.shape

        # ── Encode all turns ──
        ids_flat  = input_ids.view(B * T, L)
        mask_flat = attention_mask.view(B * T, L)

        enc_out = self.encoder(input_ids=ids_flat, attention_mask=mask_flat)
        e_flat  = self.masked_mean_pool(enc_out.last_hidden_state, mask_flat)
        x       = e_flat.view(B, T, -1)                              # [B, T, 768]

        # ── GRU ──
        lengths = turn_mask.sum(dim=1).long().cpu()
        packed  = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        rnn_packed, _ = self.gru(packed)
        rnn_out, _    = nn.utils.rnn.pad_packed_sequence(
            rnn_packed, batch_first=True, total_length=T
        )                                                             # [B, T, H]

        # ── Per-turn probability ──
        rnn_out    = self.dropout(rnn_out)
        logits     = self.classifier(rnn_out).squeeze(-1)            # [B, T]
        turn_probs = torch.sigmoid(logits)                           # [B, T]

        # ── Noisy-OR → dialogue probability ──
        dialogue_probs = self.noisy_or(turn_probs, turn_mask)        # [B]

        # ── Loss ──
        loss = None
        if dialogue_labels is not None:
            loss = self.loss_fn(dialogue_probs, dialogue_labels.float())

        return {
            "loss":           loss,
            "turn_probs":     turn_probs,      # [B, T]
            "dialogue_probs": dialogue_probs,  # [B]
        }

    # ── Streaming inference (1 turn at a time) ─────────────────

    def encode_single_turn(self, input_ids, attention_mask, h_prev=None):
        """
        Encode 1 turn, update GRU hidden state, return p_t.

        Returns
        -------
        prob_scam : float   σ(logit_t) ∈ [0, 1]
        h_new     : Tensor  [num_layers, 1, H]
        """
        device = input_ids.device

        with torch.no_grad():
            enc_out   = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            e         = self.masked_mean_pool(enc_out.last_hidden_state, attention_mask)

        x = e.unsqueeze(1)  # [1, 1, 768]

        if h_prev is None:
            h_prev = torch.zeros(
                self.config.gru_num_layers, 1, self.config.gru_hidden_size,
                device=device,
            )

        with torch.no_grad():
            rnn_out, h_new = self.gru(x, h_prev)                    # [1,1,H], [L,1,H]
            logit          = self.classifier(rnn_out.squeeze(1))     # [1, 1]
            prob_scam      = torch.sigmoid(logit).item()

        return prob_scam, h_new

    # ── Utilities ──────────────────────────────────────────────

    def get_param_groups(self, rnn_head_lr: float):
        trainable = [p for p in self.parameters() if p.requires_grad]
        return [{"params": trainable, "lr": rnn_head_lr}]

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
