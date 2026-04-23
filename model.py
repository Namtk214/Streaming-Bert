"""
Streaming Scam Detector – PhoBERT + uni-GRU.

Kiến trúc:
  turn text u_t
    → PhoBERT turn encoder
    → masked mean pooling → e_t (768-dim)
    → uni-GRU → h_t (hidden_dim)
    → 3-class classifier → logits (num_classes-dim)
    → classes: 0=LEGIT, 1=SCAM, 2=AMBIGUOUS

Forward (training):
  input_ids [B,T,L] → flatten [B*T,L] → PhoBERT → pool → [B,T,768]
  → GRU → [B,T,256] → classifier → [B,T,C]
  masked CrossEntropyLoss theo turn_mask

Forward (inference):
  Từng turn 1: encode → GRU step với h_{t-1} → classify → lưu h_t
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class StreamingScamDetector(nn.Module):
    """PhoBERT per-turn encoder + uni-GRU + 3-class classification head."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # PhoBERT encoder
        self.encoder = AutoModel.from_pretrained(config.model_name)
        self.encoder_hidden_size = self.encoder.config.hidden_size  # 768

        # uni-GRU conversation encoder
        self.gru = nn.GRU(
            input_size=self.encoder_hidden_size,
            hidden_size=config.gru_hidden_size,
            num_layers=config.gru_num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )

        # 3-class classifier head
        self.dropout = nn.Dropout(config.head_dropout)
        self.classifier = nn.Linear(config.gru_hidden_size, config.num_classes)

        # CrossEntropyLoss — reduction="none" để mask padding turns
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def masked_mean_pool(self, token_hidden, attention_mask):
        """Masked mean pooling: [N, L, H] → [N, H]."""
        mask_expanded = attention_mask.unsqueeze(-1).float()       # [N, L, 1]
        sum_hidden = (token_hidden * mask_expanded).sum(dim=1)     # [N, H]
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)        # [N, 1]
        return sum_hidden / sum_mask

    def forward(self, input_ids, attention_mask, turn_mask, labels=None):
        """
        Parameters
        ----------
        input_ids      : [B, T, L]
        attention_mask : [B, T, L]
        turn_mask      : [B, T]   – 1=turn thật, 0=padding
        labels         : [B, T]   – class index (long), optional

        Returns
        -------
        dict with 'loss' (if labels) and 'logits' [B, T, C]
        """
        B, T, L = input_ids.shape

        # Flatten turns → encode
        input_ids_flat = input_ids.view(B * T, L)
        attention_mask_flat = attention_mask.view(B * T, L)

        encoder_output = self.encoder(
            input_ids=input_ids_flat,
            attention_mask=attention_mask_flat,
        )
        token_hidden = encoder_output.last_hidden_state              # [B*T, L, 768]

        e_flat = self.masked_mean_pool(token_hidden, attention_mask_flat)  # [B*T, 768]
        x = e_flat.view(B, T, -1)                                    # [B, T, 768]

        # GRU với packed sequence
        lengths = turn_mask.sum(dim=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        rnn_out_packed, _ = self.gru(packed)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(
            rnn_out_packed, batch_first=True, total_length=T
        )                                                             # [B, T, hidden]

        # Classifier
        rnn_out = self.dropout(rnn_out)
        logits = self.classifier(rnn_out)                            # [B, T, C]

        # Masked CrossEntropyLoss
        loss = None
        if labels is not None:
            logits_flat = logits.view(B * T, -1)                     # [B*T, C]
            labels_flat = labels.view(B * T).long()                  # [B*T]
            loss_raw = self.loss_fn(logits_flat, labels_flat)        # [B*T]
            loss_raw = loss_raw.view(B, T)
            loss = (loss_raw * turn_mask).sum() / turn_mask.sum()

        return {"loss": loss, "logits": logits}

    def encode_single_turn(self, input_ids, attention_mask, h_prev=None):
        """
        Encode 1 turn và update hidden state.

        Returns
        -------
        probs : list[float]  – softmax probabilities [C]
        h_new : Tensor [num_layers, 1, hidden]
        """
        device = input_ids.device

        with torch.no_grad():
            encoder_output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            token_hidden = encoder_output.last_hidden_state          # [1, L, 768]
            e = self.masked_mean_pool(token_hidden, attention_mask)  # [1, 768]

        x = e.unsqueeze(1)                                           # [1, 1, 768]

        if h_prev is None:
            h_prev = torch.zeros(
                self.config.gru_num_layers, 1, self.config.gru_hidden_size,
                device=device,
            )

        with torch.no_grad():
            rnn_out, h_new = self.gru(x, h_prev)                    # [1,1,H], [L,1,H]
            logits = self.classifier(rnn_out.squeeze(1))             # [1, C]
            probs = torch.softmax(logits, dim=-1).squeeze(0)         # [C]

        return probs.cpu().tolist(), h_new

    def get_param_groups(self, encoder_lr: float, rnn_head_lr: float):
        encoder_params = []
        other_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("encoder."):
                encoder_params.append(param)
            else:
                other_params.append(param)
        groups = []
        if encoder_params:
            groups.append({"params": encoder_params, "lr": encoder_lr})
        if other_params:
            groups.append({"params": other_params, "lr": rnn_head_lr})
        return groups

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
