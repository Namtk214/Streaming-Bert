# Baseline 2: Early-Exit with Weighted Loss

**PhoBERT + Cross-Turn Attention + Weighted Cumulative Loss** cho bài toán Scam Detection theo kiểu streaming.

---

## Tổng quan kiến trúc

```
Turn x_t  →  PhoBERT (frozen)  →  Mean Pooling  →  h_t ∈ ℝ^768
                                                       │
                   ┌───────────────────────────────────┘
                   │
              t = 1:  c_1 = 0⃗
              t ≥ 2:  c_t = CrossAttention(query=h_t, kv=[h_1…h_{t-1}])
                   │
                   ▼
          z_t = concat(h_t, c_t) ∈ ℝ^1536
                   │
                   ▼
          Classifier: Linear(1536 → 3)  →  logits_t
                   │
                   ▼
          Loss = Σ_{t=1..N}  (2t/N) · CE(logits_t, y)
```


### Ý tưởng chính

- **Early-Exit**: Model có thể đưa ra prediction ở **mọi turn**, không cần chờ hết hội thoại.
- **Weighted Loss**: Turn càng muộn → weight càng lớn (`w_t = 2t/N`), nên prediction cuối cùng quan trọng nhất, nhưng model vẫn học classify sớm dần.
- **Cross-Turn Attention**: Turn hiện tại attend vào các turn trước (chỉ `h_1…h_{t-1}`, không include turn hiện tại), giúp model hiểu ngữ cảnh hội thoại.

---

## Cấu trúc thư mục

```
Baseline2/
├── config.py                       # Config dataclass (hyperparameters)
├── prepare_data.py                 # Convert raw JSON → train/val/test
├── dataset.py                      # Dataset + Collate function
├── train.py                        # Training loop
├── infer_stream.py                 # Streaming inference engine
├── visualize.py                    # Gradio demo UI
├── metrics.py                      # Evaluation metrics
├── models/
│   ├── __init__.py
│   ├── turn_encoder.py             # PhoBERT + masked mean pooling
│   ├── cross_turn_attention.py     # Multi-head cross-turn attention
│   └── early_exit_model.py         # Main model (encoder + attn + classifier)
├── losses/
│   ├── __init__.py
│   └── weighted_prefix_loss.py     # Weighted cumulative CE loss
├── data/                           # Generated data (not committed)
│   ├── train.json
│   ├── val.json
│   └── test.json
└── outputs/                        # Training outputs (not committed)
    └── best_model/
```

---

## Yêu cầu

```bash
pip install torch transformers gradio scikit-learn numpy
pip install py_vncorenlp
apt-get install -y default-jdk   # JDK cần cho VnCoreNLP
```

> **VnCoreNLP là bắt buộc.** PhoBERT yêu cầu input phải được word-segment bằng VnCoreNLP.
> Nếu thiếu `py_vncorenlp` hoặc JDK, pipeline sẽ báo lỗi.

---

## Hướng dẫn sử dụng

### 1. Chuẩn bị dữ liệu

Convert raw conversations sang format Baseline2:

```bash
python prepare_data.py
```

**Input**: `../data/excel_raw_conversations.json` (945 conversations)

**Output**: `data/train.json`, `data/val.json`, `data/test.json`

```
Train: 665 | Val: 140 | Test: 140
Labels: LEGIT (459) + SCAM (486)
```

Mỗi dialogue có format:
```json
{
  "dialogue_id": "excel_sheet1_1",
  "conversation_label": "SCAM",
  "label_id": 1,
  "turns": [
    {"turn_id": 1, "speaker": 0, "text": "...", "text_segmented": "..."}
  ]
}
```

### 2. Training

```bash
# Full training (khuyến nghị GPU)
python train.py

# Debug mode (2 epochs, batch_size=2)
python train.py --debug

# Small mode (5 epochs, batch_size=2)
python train.py --small

# Custom output directory
python train.py --output-dir outputs/experiment_1
```

**Chiến lược training:**
- PhoBERT **frozen** hoàn toàn → chỉ train cross-attention + classifier head
- AdamW + cosine schedule with warmup
- Early stopping theo macro F1 trên validation set
- Gradient clipping (max norm = 1.0)

**Trainable parameters**: ~1.2M (attention + classifier), so với ~135M total PhoBERT params.

### 3. Inference (streaming)

```python
from infer_stream import EarlyExitInferenceEngine

engine = EarlyExitInferenceEngine(
    model_path="outputs/best_model",
    threshold=0.5,
)

# Predict từng turn (streaming)
r1 = engine.predict_turn("dlg_001", "Alo ai đấy?")
r2 = engine.predict_turn("dlg_001", "Tôi là công an, bạn đang bị điều tra!")
print(r2["prediction"])     # "SCAM"
print(r2["probabilities"])  # {"LEGIT": 0.1, "SCAM": 0.8, "AMBIGUOUS": 0.1}

# Hoặc predict cả conversation
messages = [
    {"speaker_role": "normal", "text": "Alo?"},
    {"speaker_role": "scammer", "text": "Chuyển tiền ngay!"},
]
results = engine.predict_conversation(messages, "dlg_002")

# Reset dialogue
engine.reset("dlg_001")
```

### 4. Demo Gradio

```bash
python visualize.py
```

Hoặc trong Colab:
```python
from visualize import launch_app
launch_app(model_path="outputs/best_model", share=True)
```

**3 tabs:**
- **Chat Mode** — nhập từng turn, chọn speaker, xem prediction real-time
- **Batch Mode** — paste cả hội thoại, phân tích toàn bộ
- **Preset Examples** — 5 ví dụ có sẵn (3 SCAM + 2 LEGIT)

---

## Chi tiết kỹ thuật

### Turn Encoder

Dùng `vinai/phobert-base-v2` (shared weights) cho mọi turn.

**Pipeline encode mỗi turn:**
```
raw text → clean → VnCoreNLP word segment → tokenize [L tokens] → PhoBERT → [L, 768] → masked mean pool → h_t [768]
```

- **Word segmentation (bắt buộc)**: PhoBERT yêu cầu input phải qua VnCoreNLP (`py_vncorenlp`)
  - `prepare_data.py`: segment offline khi chuẩn bị data → lưu vào `text_segmented`
  - `infer_stream.py`: segment online cho mỗi turn mới khi inference

Trong training, tất cả turns trong batch được **flatten** và encode cùng một lần (`[B*T, L]`) để tận dụng batch parallelism, sau đó regroup lại theo dialogue.

### Cross-Turn Attention

`nn.MultiheadAttention` (8 heads, `batch_first=True`):

```
Query: h_t           [1, 1, 768]
Key:   H[:t]         [1, t-1, 768]
Value: H[:t]         [1, t-1, 768]
Output: c_t          [768]
```

- Turn 1: `c_1 = 0⃗` (zero vector), không có history.
- Turn t ≥ 2: attend vào `[h_1, …, h_{t-1}]`. **Không** include `h_t` trong key/value.

### Fusion & Classifier

```
z_t = concat(h_t, c_t) ∈ ℝ^{1536}     # cả turn 1 (c_1 = zeros)
logits_t = Linear(1536 → 3)(z_t)       # 3 classes
```

Baseline dùng single Linear layer. Có thể nâng lên MLP nếu cần.

### Weighted Cumulative Loss

```
L = Σ_{t=1}^{N} w_t · CE(logits_t, y)

w_t = 2t / N    (1-indexed)
```

Ví dụ N=5: weights = `[0.4, 0.8, 1.2, 1.6, 2.0]`

**Lưu ý quan trọng**: index trong code là 0-based → `w_i = 2*(i+1)/N`

### Streaming State

Khi inference online, chỉ cần cache:
```python
H_prev = [h_1, h_2, ..., h_{t-1}]  # list of Tensor [768]
```

Mỗi turn mới: encode → cross-attn với history → classify → append `h_t` vào history.

---

## Hyperparameters (mặc định)

| Parameter | Value | Mô tả |
|---|---|---|
| `model_name` | `vinai/phobert-base-v2` | PhoBERT encoder |
| `max_tokens_per_turn` | 128 | Max tokens mỗi turn |
| `freeze_encoder` | True | Freeze PhoBERT |
| `attn_num_heads` | 8 | Số attention heads |
| `attn_dropout` | 0.1 | Dropout trong attention |
| `num_classes` | 3 | LEGIT / SCAM / AMBIGUOUS |
| `head_dropout` | 0.2 | Dropout trước classifier |
| `head_lr` | 1e-3 | Learning rate |
| `num_epochs` | 15 | Max epochs |
| `batch_size` | 4 | Batch size |
| `patience` | 5 | Early stopping patience |
| `warmup_ratio` | 0.1 | Warmup steps ratio |
| `grad_clip` | 1.0 | Gradient clipping |

---

## Metrics

### Final-turn (dialogue-level)
- **Accuracy**, **Macro F1**, **Weighted F1**
- **Per-class Precision / Recall** (LEGIT, SCAM, AMBIGUOUS)

### All-turn
- **Turn Accuracy**, **Turn Macro F1** (mỗi turn đều phải predict đúng)

### Streaming-specific
- **Detection Rate**: tỉ lệ SCAM dialogues được phát hiện
- **Average Detection Delay**: trung bình bao nhiêu turn để phát hiện SCAM
- **False Alarm Rate**: tỉ lệ LEGIT dialogues bị predict sai thành SCAM

---

## Các lỗi thường gặp

1. **Sai weight indexing**: `w = 2*i/N` (sai) → `w = 2*(i+1)/N` (đúng, 1-based)
2. **Turn 1 dimension mismatch**: Luôn dùng `c_1 = zeros(d)` + concat để đảm bảo `z_t` luôn có dim `2d`
3. **Attention include current turn**: `H[:t]` chỉ chứa `h_1…h_{t-1}`, **không** include `h_t`
4. **Dùng chung N cho cả batch**: Mỗi dialogue có N riêng, tính weight riêng

---

## Extending

Sau khi baseline chạy ổn, có thể thử:
- Thay `concat` bằng **gated fusion**
- Thay **Linear** classifier bằng **MLP** (2 layers)
- Thêm **threshold-based early alert** metric
- **Unfreeze** top PhoBERT layers (staged training)
- Thêm **speaker embedding** riêng
