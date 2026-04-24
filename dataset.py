"""
Dataset và Collate Function cho Streaming Binary Scam Detection (Noisy-OR MIL).

Mỗi sample là 1 dialogue.
Label là dialogue-level (0=harmless, 1=scam) — không cần per-turn label.

Collate thực hiện padding 2 cấp:
  1) Token-level: pad từng turn tới max_token_len (do tokenizer)
  2) Turn-level:  pad số turns tới max_num_turns trong batch → turn_mask
"""

import json
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class StreamingDialogueDataset(Dataset):
    """
    Dataset trả về 1 dialogue mỗi sample.

    Mỗi sample gồm:
    - input_ids       [T, L]  token IDs cho từng turn
    - attention_mask  [T, L]  attention mask
    - dialogue_label  int     0=harmless, 1=scam
    - num_turns       int     số turn thật
    """

    def __init__(self, data_path: str, tokenizer, max_token_len: int = 128):
        with open(data_path, "r", encoding="utf-8") as f:
            self.dialogues = json.load(f)
        self.tokenizer   = tokenizer
        self.max_token_len = max_token_len

    def __len__(self) -> int:
        return len(self.dialogues)

    def __getitem__(self, idx: int) -> Dict:
        dlg   = self.dialogues[idx]
        turns = dlg["turns"]

        input_ids_list   = []
        attn_mask_list   = []

        for turn in turns:
            enc = self.tokenizer(
                turn["text_segmented"],
                max_length=self.max_token_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids_list.append(enc["input_ids"].squeeze(0))
            attn_mask_list.append(enc["attention_mask"].squeeze(0))

        # dialogue_label: 1 nếu scam, 0 nếu harmless
        label_str = dlg["conversation_label"]
        dialogue_label = 1 if label_str == "scam" else 0

        return {
            "input_ids":      torch.stack(input_ids_list),    # [T, L]
            "attention_mask": torch.stack(attn_mask_list),    # [T, L]
            "dialogue_label": dialogue_label,                  # int
            "num_turns":      len(turns),
        }


def streaming_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Padding 2 cấp.

    Output:
    - input_ids        [B, T_max, L]
    - attention_mask   [B, T_max, L]
    - turn_mask        [B, T_max]    1=real turn, 0=padding
    - dialogue_labels  [B]           0/1
    """
    max_turns = max(item["num_turns"] for item in batch)
    token_len = batch[0]["input_ids"].shape[-1]
    B = len(batch)

    input_ids      = torch.zeros(B, max_turns, token_len, dtype=torch.long)
    attention_mask = torch.zeros(B, max_turns, token_len, dtype=torch.long)
    turn_mask      = torch.zeros(B, max_turns, dtype=torch.float)
    dialogue_labels = torch.zeros(B, dtype=torch.float)

    for i, item in enumerate(batch):
        T = item["num_turns"]
        input_ids[i, :T]      = item["input_ids"]
        attention_mask[i, :T] = item["attention_mask"]
        turn_mask[i, :T]      = 1.0
        dialogue_labels[i]    = float(item["dialogue_label"])

    return {
        "input_ids":       input_ids,
        "attention_mask":  attention_mask,
        "turn_mask":       turn_mask,
        "dialogue_labels": dialogue_labels,
    }
