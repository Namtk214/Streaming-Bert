"""
Dataset và Collate Function cho Streaming Scam Detection.

Mỗi sample là 1 dialogue (danh sách turns).
Collate function thực hiện padding 2 cấp:
  1) Token-level: pad từng turn tới max_token_len (do tokenizer)
  2) Turn-level: pad số turns tới max_num_turns trong batch
"""

import json
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class StreamingDialogueDataset(Dataset):
    """
    Dataset trả về 1 dialogue mỗi sample.

    Mỗi sample gồm:
    - input_ids [T, L]: token IDs cho từng turn
    - attention_mask [T, L]: attention mask cho từng turn
    - labels [T]: binary scam label theo prefix rule
    - num_turns: số turn thật (dùng để tạo turn_mask khi collate)
    """

    def __init__(self, data_path: str, tokenizer, max_token_len: int = 128):
        with open(data_path, "r", encoding="utf-8") as f:
            self.dialogues = json.load(f)
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self) -> int:
        return len(self.dialogues)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dlg = self.dialogues[idx]
        turns = dlg["turns"]

        input_ids_list = []
        attention_mask_list = []
        labels = []

        for turn in turns:
            # Tokenize từng turn riêng
            encoding = self.tokenizer(
                turn["text_segmented"],
                max_length=self.max_token_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids_list.append(encoding["input_ids"].squeeze(0))
            attention_mask_list.append(encoding["attention_mask"].squeeze(0))
            labels.append(turn["scam_label"])

        return {
            "input_ids": torch.stack(input_ids_list),            # [T, L]
            "attention_mask": torch.stack(attention_mask_list),   # [T, L]
            "labels": torch.tensor(labels, dtype=torch.float),   # [T]
            "num_turns": len(turns),
        }


def streaming_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function với padding 2 cấp.

    Token-level padding: đã thực hiện bởi tokenizer (max_length).
    Turn-level padding: pad tới max_num_turns trong batch hiện tại.
    Tạo turn_mask để phân biệt turn thật vs turn padding.

    Output tensors:
    - input_ids:      [B, T_max, L]
    - attention_mask:  [B, T_max, L]
    - turn_mask:       [B, T_max]
    - labels:          [B, T_max]
    """
    max_turns = max(item["num_turns"] for item in batch)
    token_len = batch[0]["input_ids"].shape[-1]
    B = len(batch)

    # Khởi tạo tensors với zeros (padding)
    input_ids = torch.zeros(B, max_turns, token_len, dtype=torch.long)
    attention_mask = torch.zeros(B, max_turns, token_len, dtype=torch.long)
    turn_mask = torch.zeros(B, max_turns, dtype=torch.float)
    labels = torch.zeros(B, max_turns, dtype=torch.float)

    for i, item in enumerate(batch):
        T = item["num_turns"]
        input_ids[i, :T] = item["input_ids"]
        attention_mask[i, :T] = item["attention_mask"]
        turn_mask[i, :T] = 1.0   # turn thật = 1, padding = 0
        labels[i, :T] = item["labels"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "turn_mask": turn_mask,
        "labels": labels,
    }
