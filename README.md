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
streaming/
├── README.md
├── config.py                  # Cấu hình hyperparameters
├── generate_data.py           # Tạo dữ liệu synthetic để test
├── prepare_streaming_data.py  # Chuyển raw JSON → streaming format
├── dataset.py                 # Dataset + collate_fn (padding 2 cấp)
├── model.py                   # PhoBERT + GRU + binary head
├── metrics.py                 # Streaming metrics (F1, AUROC, delay...)
├── train.py                   # Training loop (staged freezing)
├── infer_stream.py            # Stateful online inference
├── visualize.py               # Gradio demo UI
├── data/
│   ├── synthetic_conversations.json  # Dữ liệu synthetic
│   ├── train.json                    # Training set (streaming format)
│   ├── val.json                      # Validation set
│   └── test.json                     # Test set
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
```

### 2. Tạo dữ liệu test

```bash
cd streaming

# Tạo 40 hội thoại synthetic (20 scam + 5 ambiguous + 15 legit)
python generate_data.py

# Chuyển sang streaming format + chia train/val/test
python prepare_streaming_data.py
```

Output:
- `data/train.json` (52 dialogues)
- `data/val.json` (9 dialogues)
- `data/test.json` (9 dialogues)

### 3. Training

```bash
# Debug mode (2 epochs, batch=2, nhanh ~1 phút trên CPU)
python train.py --debug

# Small mode (5 epochs, batch=2)
python train.py --small

# Full training (10 epochs, khuyến nghị chạy trên GPU)
python train.py
```

**Staged training:**

| Stage | Epochs | PhoBERT | Trainable params |
|-------|--------|---------|-----------------|
| A | 1-3 | Frozen hoàn toàn | ~800K |
| B | 4-6 | Top 2 layers unfrozen | ~15M |
| C | 7+ | Top 4 layers unfrozen | ~29M |

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

Cùng format với `data/raw_conversations.json`:

```json
{
  "conversation_id": "conv_0001",
  "t1_label": "SCAM",
  "messages": [
    {
      "turn_id": "conv_0001_t01",
      "speaker_role": "normal",
      "text": "Alo ai đấy?"
    },
    {
      "turn_id": "conv_0001_t02",
      "speaker_role": "scammer",
      "text": "Tôi là công an...",
      "t4_labels": ["AUTHORITY", "THREAT_LEGAL"]
    }
  ]
}
```

### Streaming format (sau prepare)

```json
{
  "dialogue_id": "conv_0001",
  "conversation_label": "SCAM",
  "turns": [
    {
      "turn_id": 1,
      "speaker": 0,
      "text": "Alo ai đấy?",
      "text_segmented": "Alo ai đấy ?",
      "scam_label": 0
    },
    {
      "turn_id": 2,
      "speaker": 1,
      "text": "Tôi là công an...",
      "text_segmented": "Tôi là công_an ...",
      "scam_label": 1
    }
  ]
}
```

**Binary label theo prefix rule:**
- `SCAM`/`AMBIGUOUS`: label=0 trước khi có bằng chứng, label=1 từ turn scammer đầu tiên
- `LEGIT`: toàn bộ label=0

**Speaker encoding:** `normal=0`, `scammer=1`, `unknown=2`

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

## Khắc phục lỗi thường gặp (Troubleshooting)

### Lỗi `UnpicklingError: Weights only load failed` trên Google Colab / PyTorch 2.6+
Lỗi này xảy ra khi bạn load mô hình được train từ code cũ (lưu `config.pt` thay vì `config.json`) lên môi trường PyTorch 2.6 (ví dụ: Google Colab) với cơ chế bảo mật tự động chặn các file `.pt` chứa cấu trúc class Python.

**Cách khắc phục:**
- **Cách 1 (Khuyên dùng):** Xóa model cũ và chạy lại lệnh training (`python train.py`). Code mới nhất đã tự động chuyển sang lưu bằng `config.json` để hoàn toàn tương thích với cơ chế an toàn của PyTorch 2.6.
- **Cách 2:** Thêm đoạn code sau vào đầu file Notebook trên Colab của bạn **trước** khi init engine:
  ```python
  import torch
  from config import StreamingConfig
  torch.serialization.add_safe_globals([StreamingConfig])
  ```

### VnCoreNLP không load được trên Google Colab
Do JVM bị Jupyter kernel chiếm trước. Cách fix:
```python
# Chạy TRƯỚC mọi import khác (ô code đầu tiên)
!apt-get install -y default-jdk
!pip install py_vncorenlp
import py_vncorenlp
py_vncorenlp.download_model(save_dir='vncorenlp')
```
Sau đó restart runtime rồi chạy lại. Nếu vẫn lỗi, model sẽ tự fallback dùng raw text (chất lượng giảm nhẹ ~5-10% F1).
