"""
Streaming Scam Detector – PhoBERT + uni-GRU.

Kiến trúc:
  turn text u_t
    → PhoBERT turn encoder
    → masked mean pooling → e_t (768-dim)
    → uni-GRU → h_t (hidden_dim)
    → binary classifier → logits (1-dim)

Forward (training):
  input_ids [B,T,L] → flatten [B*T,L] → PhoBERT → pool → [B,T,768]
  → GRU → [B,T,256] → classifier → [B,T]
  masked BCEWithLogitsLoss theo turn_mask

Forward (inference):
  Từng turn 1: encode → GRU step với h_{t-1} → classify → lưu h_t
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class StreamingScamDetector(nn.Module):
    """PhoBERT per-turn encoder + uni-GRU + binary classification head."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. PhoBERT encoder (pretrained)
        self.encoder = AutoModel.from_pretrained(config.model_name)
        self.encoder_hidden_size = self.encoder.config.hidden_size  # 768

        # 2. uni-GRU (conversation encoder)
        gru_input_size = self.encoder_hidden_size
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=config.gru_hidden_size,
            num_layers=config.gru_num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,  # chỉ dùng dropout ở head
        )

        # 4. Classifier head
        self.dropout = nn.Dropout(config.head_dropout)
        self.classifier = nn.Linear(config.gru_hidden_size, 1)

        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    # ──────────────────────────────────────────────
    # Pooling
    # ──────────────────────────────────────────────
    def masked_mean_pool(self, token_hidden, attention_mask):
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

    # ──────────────────────────────────────────────
    # Forward (training – full batch)
    # ──────────────────────────────────────────────
    def forward(self, input_ids, attention_mask,
                turn_mask, labels=None):
        """
        Parameters
        ----------
        input_ids      : [B, T, L]
        attention_mask  : [B, T, L]
        turn_mask       : [B, T]   – 1 = turn thật, 0 = padding
        labels          : [B, T]   – binary scam labels (optional)

        Returns
        -------
        dict with 'loss' (if labels) and 'logits' [B, T]
        """
        B, T, L = input_ids.shape

        # ── 1. Flatten turns ──
        input_ids_flat = input_ids.view(B * T, L)
        attention_mask_flat = attention_mask.view(B * T, L)

        # ── 2. PhoBERT encode ──
        encoder_output = self.encoder(
            input_ids=input_ids_flat,
            attention_mask=attention_mask_flat,
        )
        token_hidden = encoder_output.last_hidden_state      # [B*T, L, 768]

        # ── 3. Masked mean pooling ──
        e_flat = self.masked_mean_pool(token_hidden, attention_mask_flat)  # [B*T, 768]
        x = e_flat.view(B, T, -1)                            # [B, T, 768]

        # ── 4. GRU (packed sequence – skip padding turns) ──
        lengths = turn_mask.sum(dim=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        rnn_out_packed, _ = self.gru(packed)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(
            rnn_out_packed, batch_first=True, total_length=T
        )                                                     # [B, T, hidden]

        # ── 6. Classifier ──
        rnn_out = self.dropout(rnn_out)
        logits = self.classifier(rnn_out).squeeze(-1)         # [B, T]

        # ── 7. Masked loss ──
        loss = None
        if labels is not None:
            loss_raw = self.loss_fn(logits, labels)            # [B, T]
            loss = (loss_raw * turn_mask).sum() / turn_mask.sum()

        return {"loss": loss, "logits": logits}

    # ──────────────────────────────────────────────
    # Inference (single turn – streaming)
    # ──────────────────────────────────────────────
    def encode_single_turn(self, input_ids, attention_mask, h_prev=None):
        """
        Encode 1 turn mới và update hidden state.

        Parameters
        ----------
        input_ids      : [1, L]
        attention_mask  : [1, L]
        h_prev         : [num_layers, 1, hidden] hoặc None

        Returns
        -------
        logits: float, h_new: Tensor [num_layers, 1, hidden]
        """
        device = input_ids.device

        # PhoBERT encode single turn
        with torch.no_grad():
            encoder_output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            token_hidden = encoder_output.last_hidden_state    # [1, L, 768]
            e = self.masked_mean_pool(token_hidden, attention_mask)  # [1, 768]

        x = e.unsqueeze(1)                                     # [1, 1, 768]

        # GRU step
        if h_prev is None:
            h_prev = torch.zeros(
                self.config.gru_num_layers, 1, self.config.gru_hidden_size,
                device=device,
            )

        with torch.no_grad():
            rnn_out, h_new = self.gru(x, h_prev)               # [1,1,hidden], [layers,1,hidden]
            logits = self.classifier(rnn_out.squeeze(1))        # [1, 1]

        return logits.item(), h_new

    # ──────────────────────────────────────────────
    # Freeze / Unfreeze helpers
    # ──────────────────────────────────────────────
    def freeze_encoder(self):
        """Freeze toàn bộ PhoBERT parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("  [Model] PhoBERT encoder FROZEN")

    def unfreeze_top_layers(self, n: int = 2):
        """Unfreeze top n transformer layers của PhoBERT."""
        # Freeze tất cả trước
        self.freeze_encoder()
        # Unfreeze top n layers
        total_layers = len(self.encoder.encoder.layer)
        for i in range(total_layers - n, total_layers):
            for param in self.encoder.encoder.layer[i].parameters():
                param.requires_grad = True
        print(f"  [Model] PhoBERT top {n}/{total_layers} layers UNFROZEN")

    def unfreeze_all(self):
        """Unfreeze toàn bộ PhoBERT."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("  [Model] PhoBERT encoder fully UNFROZEN")

    def get_param_groups(self, encoder_lr: float, rnn_head_lr: float):
        """
        Tạo param groups với LR khác nhau cho encoder và phần còn lại.
        Chỉ bao gồm params có requires_grad = True.
        """
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
        """Đếm số lượng trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
