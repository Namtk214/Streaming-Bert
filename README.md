# Streaming Binary Scam Detection với PhoBERT + GRU

Hệ thống phát hiện lừa đảo theo thời gian thực (streaming). Input đi vào theo từng turn hội thoại, model cập nhật trạng thái và đưa ra dự đoán scam/not-scam tại mỗi thời điểm.

## Kiến trúc

```
turn text u_t
   → PhoBERT per-turn encoder (pretrained)
   → masked mean pooling → e_t (768-dim)
   → concat speaker embedding (16-dim)
   → uni-GRU (hidden=256)
   → h_t
   → binary classifier (Linear → sigmoid)
   → prediction y_hat_t
```

**Tại sao không đưa cả hội thoại vào PhoBERT?**
- Hội thoại dài dễ vượt giới hạn 256 tokens
- Bài toán streaming cần lưu state theo thời gian
- PhoBERT encode từng turn rẻ hơn khi inference online

## Cấu trúc thư mục

```
Streaming-Bert/
├── README.md
├── config.py                  # Cấu hình hyperparameters
├── generate_data.py           # Tạo dữ liệu synthetic để test
├── prepare_data.py            # Chuyển raw train/val/test JSON → streaming dataset
├── dataset.py                 # Dataset + collate_fn (padding 2 cấp)
├── model.py                   # PhoBERT + GRU + binary head
├── metrics.py                 # Streaming metrics (F1, AUROC, delay...)
├── train.py                   # Training loop (Noisy-OR MIL)
├── infer_stream.py            # Stateful online inference
├── visualize.py               # Gradio demo UI
├── data/
│   ├── train.json              # Training set (streaming format)
│   ├── val.json                # Validation set
│   └── test.json               # Test set
└── outputs/
    └── best_model/                   # Model đã train
        ├── model.pt
        ├── config.json
        └── tokenizer files...
```

## Hướng dẫn sử dụng

### 1. Cài đặt dependencies

```bash
pip install torch transformers scikit-learn numpy py_vncorenlp gradio
apt-get install -y default-jdk
pip install py_vncorenlp
```

### 2. Chuẩn bị dữ liệu

```bash
cd Streaming-Bert

python prepare_data.py
```

Mặc định script đọc raw data từ `../data/train.json`, `../data/val.json`, `../data/test.json` và ghi output vào `Streaming-Bert/data/`.

Có thể truyền path khác nếu cần:

```bash
python prepare_data.py \
  --raw-dir ../data \
  --out-dir data \
  --vncorenlp-dir ../vncorenlp
```

Output:
- `data/train.json`
- `data/val.json`
- `data/test.json`


### 3. Training

```bash
# Debug mode (2 epochs, batch=2, nhanh ~1 phút trên CPU)
python train.py --debug

# Small mode (5 epochs, batch=2)
python train.py --small

# Full training (10 epochs, khuyến nghị chạy trên GPU)
python train.py
```


PhoBERT được fine-tune end-to-end từ epoch đầu. Optimizer dùng learning rate riêng cho encoder và GRU/head.

### 4. Inference (streaming)

```python
from infer_stream import StreamingInferenceEngine

engine = StreamingInferenceEngine(
    model_path="outputs/best_model",
    vncorenlp_dir="vncorenlp",   # optional
    threshold=0.5,
)

# Simulate từng turn đến
engine.predict_turn("dlg_001", "Alo ai đấy?", speaker="normal")
# → {"probability": 0.12, "is_scam": False, ...}

engine.predict_turn("dlg_001", "Tôi là công an, bạn đang bị điều tra!", speaker="scammer")
# → {"probability": 0.87, "is_scam": True, ...}

# Hoặc predict cả conversation
messages = [
    {"speaker_role": "normal", "text": "Alo ai đấy?"},
    {"speaker_role": "scammer", "text": "Tôi là công an!"},
]
results = engine.predict_conversation(messages, "dlg_002")

# In kết quả
for r in results:
    label = "[!] SCAM" if r["is_scam"] else "[OK]"
    print(f"  Turn {r['turn_index']}: prob={r['probability']:.3f} {label} | {r['text_preview']}")

# Reset state
engine.reset("dlg_001")
```

### 5. Visualization (Gradio)

Giao diện trực quan để demo model với 3 chế độ:
- **Chat Mode**: Nhập từng turn, xem kết quả real-time (mô phỏng streaming)
- **Batch Mode**: Paste cả hội thoại, phân tích một lượt
- **Preset Examples**: Chọn ví dụ có sẵn (scam + legit) để test nhanh

```python
# Trên Google Colab
!pip install gradio
from visualize import launch_app
launch_app(model_path="outputs/best_model", share=True)

# Local
python visualize.py
```

## Format dữ liệu

### Input (raw conversations)

`prepare_data.py` đọc các file `train.json`, `val.json`, `test.json` trong raw data folder. Mỗi file là danh sách conversations theo format:

```json
{
  "_id": "conv_0001",
  "label": "scam",
  "source_split": "train",
  "turns": [
    {
      "turn_idx": 0,
      "role": "người gọi",
      "content": "Alo ai đấy?",
      "tactic_tags": []
    },
    {
      "turn_idx": 1,
      "role": "người nghe",
      "content": "Tôi là công an...",
      "tactic_tags": ["AUTHORITY", "THREAT_LEGAL"]
    }
  ]
}
```

### Streaming format (sau prepare)

```json
{
  "dialogue_id": "conv_0001",
  "conversation_label": "scam",
  "turns": [
    {
      "turn_id": 1,
      "speaker": 0,
      "text": "Alo ai đấy?",
      "text_segmented": "Alo ai đấy ?",
      "turn_label": 1
    },
    {
      "turn_id": 2,
      "speaker": 1,
      "text": "Tôi là công an...",
      "text_segmented": "Tôi là công_an ...",
      "turn_label": 1
    }
  ]
}
```

**Conversation label:**
- `scam`: dialogue positive
- `harmless`: dialogue negative

**Turn label hiện tại trong `prepare_data.py`:**
- Dialogue `harmless`: mọi turn có `turn_label=0`
- Dialogue `scam`: mọi turn có `turn_label=1`

Model hiện tại train bằng dialogue-level label qua Noisy-OR MIL; `turn_label` chỉ được giữ trong data để tham khảo.

**Speaker encoding:** `người gọi=0`, `người nghe=1`, `unknown=2`

## Metrics

### Turn-level (chuẩn)
- Accuracy, Precision, Recall, F1, AUROC

### Streaming-specific
| Metric | Ý nghĩa |
|--------|---------|
| **Detection rate** | Tỷ lệ dialogue scam được phát hiện |
| **Avg detection delay** | Trung bình số turn từ scam onset đến first alert |
| **False alarm rate** | Tỷ lệ turn bị báo scam sai trước khi scam xảy ra |

## Hyperparameters

| Param | Giá trị | Ghi chú |
|-------|---------|---------|
| PhoBERT | `vinai/phobert-base-v2` | Pretrained encoder |
| Max tokens/turn | 128 | Truncate turn dài |
| GRU hidden | 256 | 1 layer, unidirectional |
| Speaker embed | 16-dim | 3 speakers |
| Dropout (head) | 0.2 | |
| Encoder LR | 1e-5 | Cho PhoBERT layers |
| RNN/Head LR | 1e-4 | Cho GRU + classifier |
| Weight decay | 0.01 | AdamW |
| Grad clip | 1.0 | |
| Warmup ratio | 0.1 | Cosine schedule |
| Batch size | 4 | Số dialogues/batch |

## Lưu ý quan trọng

1. **Không dùng BiLSTM** ở conversation level — gây leak thông tin tương lai
2. **Phải mask loss** ở turn padding — nếu không metrics và loss sẽ sai
3. **LR khác nhau** cho encoder vs RNN — LR cao phá pretrained weights
4. **Word segment tiếng Việt** trước khi tokenize — cải thiện chất lượng PhoBERT
5. **Không fix threshold=0.5** — nên tune trên validation set




