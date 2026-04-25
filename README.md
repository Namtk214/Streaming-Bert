# 🛡️ Early-Exit Scam Detection with Noisy-OR Loss

**Streaming online scam detector** sử dụng PhoBERT + Cross-Turn Attention + Noisy-OR aggregation.

Model đưa ra dự đoán **real-time** sau mỗi turn hội thoại, không cần đợi hết cuộc hội thoại.

---

## Ý tưởng cốt lõi

Cho một dialogue gồm T turns, model sinh cho mỗi turn một **evidence probability**:

```
q_t = sigmoid(s_t)     — "xác suất turn t chứa bằng chứng scam"
```

Các `q_t` được gộp thành xác suất dialogue-level bằng **Noisy-OR**:

```
p_dialogue = 1 − ∏(1 − q_t)
```

Loss chỉ cần **1 nhãn binary** ở mức dialogue:

```
L = BCE(p_dialogue, y)     — y ∈ {0: harmless, 1: scam}
```

### Tại sao Noisy-OR?

- ❌ **Weighted per-turn CE** (cách cũ): ép mọi turn trong dialogue scam → label scam, gây label noise mạnh ở early turns
- ✅ **Noisy-OR**: chỉ cần ít nhất 1 turn có evidence mạnh → dialogue được phân loại scam. Supervision sạch hơn nhiều khi chỉ có dialogue-level label

---

## Kiến trúc

```
Turn text → PhoBERT (frozen) → mean pooling → h_t ∈ R^768
         → Cross-Turn Attention(query=h_t, kv=[h_1..h_{t-1}]) → c_t
         → concat(h_t, c_t) ∈ R^1536
         → Linear(1536 → 1) → sigmoid → q_t
         → Noisy-OR aggregation → p_dialogue
         → BCE(p_dialogue, y_dialogue)
```

| Component | Module | Trainable |
|-----------|--------|-----------|
| Turn encoder | PhoBERT (`vinai/phobert-base-v2`) | ❄️ Frozen |
| Cross-turn attention | `nn.MultiheadAttention` (8 heads) | ✅ |
| Evidence head | `nn.Linear(1536 → 1)` | ✅ |

### Online Inference (Streaming)

Tại runtime, Noisy-OR được cập nhật **online** sau mỗi turn:

```
p_0 = 0
p_t = 1 − (1 − p_{t-1}) × (1 − q_t)
```

`p_t` **luôn không giảm** → rất phù hợp cho early detection.

---

## Cấu trúc project

```
Baseline2/
├── data/                          # Raw data (chưa segment)
│   ├── train.json
│   ├── val.json
│   └── test.json
│
├── Streaming-Bert/
│   ├── config.py                  # Hyperparameters & paths
│   ├── prepare_data.py            # Clean + VnCoreNLP word segment
│   ├── dataset.py                 # Dataset & collate (dialogue-level)
│   │
│   ├── models/
│   │   ├── turn_encoder.py        # PhoBERT + mean pooling
│   │   ├── cross_turn_attention.py # Multi-head cross-turn attention
│   │   └── early_exit_model.py    # Main model (Noisy-OR)
│   │
│   ├── losses/
│   │   └── weighted_prefix_loss.py # Noisy-OR loss function
│   │
│   ├── train.py                   # Training loop
│   ├── metrics.py                 # Dialogue-level binary metrics
│   ├── infer_stream.py            # Streaming inference engine
│   ├── visualize.py               # Gradio demo app
│   │
│   ├── convert_excel.py           # Convert Excel test data → JSON
│   ├── test.py                    # Test model trên data từ Excel
│   │
│   ├── data/                      # Processed data (sau prepare_data.py)
│   └── outputs/                   # Trained models & logs
│
└── instructions.txt               # Chi tiết thiết kế Noisy-OR
```

---

## Data Format

### Raw data (`data/train.json`)

```json
{
  "_id": 2079,
  "turns": [
    {"turn_idx": 0, "role": "người gọi", "content": "A lô?"},
    {"turn_idx": 1, "role": "người nghe", "content": "Dạ anh ơi..."}
  ],
  "label": "harmless",
  "sample_id": "harmless-2079"
}
```

- `label`: `"harmless"` hoặc `"scam"` — **dialogue-level**, không cần nhãn per-turn
- `role`: `"người gọi"` / `"người nghe"`

### Processed data (`Streaming-Bert/data/train.json`)

Sau khi chạy `prepare_data.py`:

```json
{
  "dialogue_id": "harmless-2079",
  "dialogue_label": 0,
  "dialogue_label_name": "harmless",
  "num_turns": 10,
  "turns": [
    {"turn_id": 1, "role": "người gọi", "text": "A lô?", "text_segmented": "A lô ?"}
  ]
}
```

- `text_segmented`: đã qua VnCoreNLP word segmentation (bắt buộc cho PhoBERT)

---

## Hướng dẫn sử dụng

### 1. Cài đặt dependencies

```bash
apt-get install -y default-jdk
pip install torch transformers py_vncorenlp scikit-learn numpy
pip install openpyxl  # cho convert_excel.py
pip install gradio    # optional, cho visualization
```

### 2. Tiền xử lý data

```bash
cd Streaming-Bert

python prepare_data.py \
    --raw-data-dir ../data \
```

Script sẽ:
- Đọc `data/{train,val,test}.json`
- Clean text + word segment (VnCoreNLP)
- Map label: `harmless → 0`, `scam → 1`
- Ghi ra `Streaming-Bert/data/{train,val,test}.json`

### 3. Training

```bash
# Full training
python train.py

# Quick debug (2 epochs, batch_size=2)
python train.py --debug

# Small run (5 epochs, batch_size=2)
python train.py --small
```

Model tốt nhất sẽ được lưu tại `outputs/best_model/`.

### 4. Streaming Inference

```python
from infer_stream import EarlyExitInferenceEngine

engine = EarlyExitInferenceEngine(
    model_path="outputs/best_model",
    vncorenlp_dir="../vncorenlp",
    threshold=0.5,
)

# Predict từng turn (streaming)
r1 = engine.predict_turn("dlg_001", "Alo ai đấy?")
r2 = engine.predict_turn("dlg_001", "Tôi là công an, bạn đang bị điều tra!")

print(f"q_t={r2['q_t']:.3f}  p_agg={r2['p_agg']:.3f}  scam={r2['is_scam']}")
# → q_t=0.823  p_agg=0.891  scam=True

# Hoặc predict cả conversation
messages = [
    {"role": "người gọi", "text": "Alo ai đấy?"},
    {"role": "người nghe", "text": "Chuyển tiền ngay!"},
]
results = engine.predict_conversation(messages, "dlg_002")
```

### 5. Gradio Demo

```bash
python visualize.py
```

Hoặc trong Colab:

```python
from visualize import launch_app
launch_app(model_path="outputs/best_model", share=True)
```

### 6. Convert Excel Test Data

Chuyển file Excel (scam ở Sheet1, harmless ở sheet no_scam) sang JSON:

```bash
python convert_excel.py \
    --excel "../Tổng hợp kịch bản test AI on devices_result_v2.xlsx" \
    --out data/excel_test.json
```

Mỗi row trong Excel: `(index, conversation)` với turns cách nhau bằng `\n`.
Script sẽ clean text, word segment (VnCoreNLP), và ghi ra JSON cùng format với processed data.

### 7. Test trên Excel Data

```bash
# Test cơ bản
python test.py --data data/excel_test.json

# Test với threshold khác
python test.py --data data/excel_test.json --threshold 0.4

# In chi tiết per-turn evidence từng dialogue
python test.py --data data/excel_test.json --verbose
```

Output: dialogue-level metrics (F1, AUROC, ...) + streaming metrics (detection delay, false alarm rate) + error summary (FN/FP).

---

## Output của Model

| Output | Ý nghĩa | Range |
|--------|----------|-------|
| `q_t` | Evidence probability tại turn t | [0, 1] |
| `p_agg` | Cumulative scam probability (Noisy-OR) | [0, 1], không giảm |
| `p_dialogue` | = `p_T_agg` (turn cuối cùng) | [0, 1] |
| `is_scam` | `p_agg ≥ threshold` | bool |

### Metrics được log

**Dialogue-level:**
- Accuracy, Precision, Recall, F1, AUROC

**Streaming:**
- Detection rate, Average first alert turn, Detection delay, False alarm rate

**Evidence analysis:**
- Mean `q_t` cho scam vs harmless dialogues

---

## Hyperparameters chính

| Parameter | Default | Mô tả |
|-----------|---------|-------|
| `model_name` | `vinai/phobert-base-v2` | PhoBERT encoder |
| `max_tokens_per_turn` | 128 | Max tokens mỗi turn |
| `freeze_encoder` | True | Freeze PhoBERT |
| `attn_num_heads` | 8 | Cross-turn attention heads |
| `head_lr` | 2e-5 | Learning rate |
| `batch_size` | 2 | Dialogues per batch |
| `num_epochs` | 15 | Max epochs |
| `patience` | 5 | Early stopping patience |
| `eps` | 1e-6 | Noisy-OR numerical stability |

Chỉnh sửa trong `config.py`.

---

## Lưu ý quan trọng

1. **Không supervise `q_t` trực tiếp** — chỉ supervise sau khi gộp thành `p_dialogue` bằng Noisy-OR
2. **PhoBERT cần word-segmented input** — luôn chạy `prepare_data.py` trước
3. **Noisy-OR tính ở log-space** — tránh underflow cho dialogue dài: `log_not_p = Σ log(1 − q_t)`
4. **Luôn clamp `q_t`** vào `[eps, 1−eps]` để tránh `log(0)`
