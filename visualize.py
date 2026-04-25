"""
Gradio Visualization cho Baseline2: Early-Exit with Noisy-OR Loss.

Giao diện trực quan để demo model:
  - Chat Mode: nhập từng turn, xem kết quả real-time (streaming)
  - Batch Mode: paste cả hội thoại, xem timeline predictions
  - Preset Examples để test nhanh

Hiển thị binary predictions (Noisy-OR) với:
  - q_t: per-turn evidence probability
  - p_agg: cumulative scam probability
  - Timeline chart theo turn

Usage (Colab):
    !pip install gradio
    from visualize import launch_app
    launch_app(model_path="outputs/best_model")

Usage (local):
    python visualize.py
"""

import os
import sys
import json

import gradio as gr
import numpy as np

# Import Baseline2 modules
_baseline2_dir = os.path.dirname(os.path.abspath(__file__))
if _baseline2_dir not in sys.path:
    sys.path.insert(0, _baseline2_dir)

from config import EarlyExitConfig, LABEL_MAP, LABEL_NAMES
from infer_stream import EarlyExitInferenceEngine

# ============================================================
# Preset examples
# ============================================================
SCAM_EXAMPLES = [
    {
        "name": "Giả mạo công an",
        "messages": [
            {"role": "người gọi", "text": "Alo ai đấy ạ?"},
            {"role": "người nghe", "text": "Tôi là Đại úy Nguyễn Văn Hùng, Công an thành phố. Bạn đang bị điều tra vì liên quan đến đường dây rửa tiền."},
            {"role": "người gọi", "text": "Cái gì ạ? Tôi không biết gì cả."},
            {"role": "người nghe", "text": "Chuyển tiền vào tài khoản an toàn ngay trong vòng 30 phút, nếu không sẽ bị bắt."},
        ],
    },
    {
        "name": "Lừa đảo ngân hàng",
        "messages": [
            {"role": "người gọi", "text": "Alo xin nghe?"},
            {"role": "người nghe", "text": "Xin chào, tôi gọi từ ngân hàng BIDV. Thẻ ATM của bạn sắp hết hạn và cần gia hạn trực tuyến ngay hôm nay."},
            {"role": "người gọi", "text": "Thẻ tôi vẫn dùng được bình thường mà?"},
            {"role": "người nghe", "text": "Hệ thống mới cập nhật ạ. Bạn cung cấp số thẻ, ngày hết hạn và mã CVV để em xử lý nhé."},
        ],
    },
    {
        "name": "Lừa đảo đầu tư crypto",
        "messages": [
            {"role": "người nghe", "text": "Chào bạn, bạn có muốn đầu tư Bitcoin với lãi suất 500% trong 30 ngày không?"},
            {"role": "người gọi", "text": "Nghe hấp dẫn nhỉ, nhưng làm sao được lãi cao vậy?"},
            {"role": "người nghe", "text": "Chúng tôi có đội ngũ chuyên gia giao dịch AI tự động. Bạn chỉ cần nạp tối thiểu 5 triệu vào ví điện tử."},
            {"role": "người gọi", "text": "Có an toàn không?"},
            {"role": "người nghe", "text": "Hoàn toàn an toàn, đã có hàng nghìn người tham gia thành công. Nhưng chương trình chỉ mở thêm 24 giờ nữa thôi."},
        ],
    },
]

LEGIT_EXAMPLES = [
    {
        "name": "Đặt pizza",
        "messages": [
            {"role": "người gọi", "text": "Alo Pizza Hut phải không ạ?"},
            {"role": "người nghe", "text": "Dạ đúng rồi ạ. Anh chị muốn đặt gì ạ?"},
            {"role": "người gọi", "text": "Cho tôi 1 pizza hải sản cỡ lớn và 2 lon Pepsi nhé."},
            {"role": "người nghe", "text": "Dạ tổng 285 nghìn ạ. Anh cho em địa chỉ giao hàng."},
        ],
    },
    {
        "name": "Hẹn đi cà phê",
        "messages": [
            {"role": "người gọi", "text": "Alo bạn ơi, chiều nay đi cà phê không?"},
            {"role": "người nghe", "text": "Oke, mấy giờ?"},
            {"role": "người gọi", "text": "3 giờ nhé, quán trên Nguyễn Huệ."},
            {"role": "người nghe", "text": "Được rồi, hẹn gặp!"},
        ],
    },
]


# ============================================================
# Color & styling helpers
# ============================================================
def scam_prob_to_color(prob):
    """Màu theo xác suất scam."""
    if prob < 0.3:
        return "#22c55e"  # green
    elif prob < 0.5:
        return "#eab308"  # yellow
    elif prob < 0.7:
        return "#f97316"  # orange
    else:
        return "#ef4444"  # red


def evidence_to_color(q):
    """Màu nhẹ hơn cho evidence q_t."""
    if q < 0.2:
        return "#86efac"
    elif q < 0.4:
        return "#fde047"
    elif q < 0.6:
        return "#fdba74"
    else:
        return "#fca5a5"


# ============================================================
# Build HTML result
# ============================================================
def build_result_html(results, messages):
    """Tạo HTML trực quan cho kết quả Noisy-OR."""
    html_parts = []

    html_parts.append("""
    <div style="font-family: 'Segoe UI', sans-serif; max-width: 750px; margin: 0 auto;">
    """)

    for r, msg in zip(results, messages):
        q_t = r["q_t"]
        p_agg = r["p_agg"]
        is_scam = r["is_scam"]
        speaker = msg.get("role", msg.get("speaker_role", "unknown"))
        text = msg.get("text", msg.get("content", ""))

        p_color = scam_prob_to_color(p_agg)
        q_color = evidence_to_color(q_t)

        border_color = "#ef4444" if is_scam else "#22c55e"
        bg_color = "#fef2f2" if is_scam else "#f0fdf4"
        status_icon = "🔴" if is_scam else "🟢"
        status_text = "SCAM" if is_scam else "SAFE"

        # Probability bars
        q_bar_w = int(q_t * 100)
        p_bar_w = int(p_agg * 100)

        html_parts.append(f"""
        <div style="margin-bottom: 14px; padding: 14px 18px; border-radius: 12px;
                    border-left: 5px solid {border_color};
                    background: {bg_color}; box-shadow: 0 1px 3px rgba(0,0,0,0.06);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; color: #374151; font-size: 14px;">
                    Turn {r['turn_index']} ({speaker})
                </span>
                <span style="font-weight: 700; color: {p_color}; font-size: 15px;
                             padding: 2px 10px; border-radius: 6px;
                             background: {p_color}18;">
                    {status_icon} {status_text}
                </span>
            </div>
            <div style="color: #374151; font-size: 14px; margin-bottom: 10px; line-height: 1.5;">
                {text}
            </div>
            <div style="padding: 6px 0;">
                <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
                    <span style="font-size: 11px; color: #6b7280; width: 72px; text-align: right;">Evidence q</span>
                    <div style="flex: 1; background: #e5e7eb; border-radius: 3px; height: 8px; overflow: hidden;">
                        <div style="background: {q_color}; width: {q_bar_w}%; height: 100%; border-radius: 3px;
                                    transition: width 0.3s;"></div>
                    </div>
                    <span style="font-size: 11px; color: #374151; width: 40px;">{q_t:.1%}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="font-size: 11px; color: #6b7280; width: 72px; text-align: right;">Cumul. p</span>
                    <div style="flex: 1; background: #e5e7eb; border-radius: 3px; height: 8px; overflow: hidden;">
                        <div style="background: {p_color}; width: {p_bar_w}%; height: 100%; border-radius: 3px;
                                    transition: width 0.3s;"></div>
                    </div>
                    <span style="font-size: 11px; color: #374151; width: 40px;">{p_agg:.1%}</span>
                </div>
            </div>
        </div>
        """)

    # Summary
    final = results[-1]
    p_final = final["p_agg"]
    is_scam = final["is_scam"]

    # First alert turn
    first_alert = None
    for r in results:
        if r["is_scam"]:
            first_alert = r["turn_index"]
            break

    if is_scam:
        summary_color = "#dc2626"
        summary_bg = "#fef2f2"
        summary_icon = "🚨"
        summary_text = f"CẢNH BÁO: Phát hiện lừa đảo! (từ Turn {first_alert})"
    else:
        summary_color = "#16a34a"
        summary_bg = "#f0fdf4"
        summary_icon = "✅"
        summary_text = "AN TOÀN: Không phát hiện dấu hiệu lừa đảo"

    html_parts.append(f"""
    <div style="margin-top: 18px; padding: 16px 20px; border-radius: 12px;
                background: {summary_bg}; border: 2px solid {summary_color};
                box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
        <div style="font-weight: 700; color: {summary_color}; font-size: 17px;">
            {summary_icon} {summary_text}
        </div>
        <div style="color: #6b7280; font-size: 13px; margin-top: 6px;">
            Cumulative scam probability: <b>{p_final:.1%}</b>
            &nbsp;|&nbsp; {len(results)} turns analyzed
        </div>
    </div>
    """)

    html_parts.append("</div>")
    return "\n".join(html_parts)


# ============================================================
# Build probability chart (HTML/CSS)
# ============================================================
def build_prob_chart(results):
    """Tạo chart q_t (evidence) và p_agg (cumulative) theo turn bằng HTML/CSS."""
    chart_html = """
    <div style="font-family: 'Segoe UI', sans-serif; max-width: 750px; margin: 18px auto 0;">
        <div style="font-weight: 600; color: #374151; margin-bottom: 12px; font-size: 15px;">
            📊 Evidence & Cumulative Probability Timeline
        </div>
        <div style="display: flex; align-items: flex-end; gap: 8px; height: 220px;
                    padding: 12px; background: #f9fafb; border-radius: 12px;
                    border: 1px solid #e5e7eb;">
    """

    for r in results:
        q_t = r["q_t"]
        p_agg = r["p_agg"]

        bar_height = 170
        q_h = int(q_t * bar_height)
        p_h = int(p_agg * bar_height)

        q_color = evidence_to_color(q_t)
        p_color = scam_prob_to_color(p_agg)

        chart_html += f"""
            <div style="flex: 1; display: flex; flex-direction: column; align-items: center;
                        justify-content: flex-end; gap: 2px;">
                <div style="font-size: 9px; color: #6b7280; font-weight: 600;">
                    {p_agg:.0%}
                </div>
                <div style="display: flex; gap: 2px; align-items: flex-end;">
                    <div style="width: 16px; background: #93c5fd; height: {q_h}px;
                                border-radius: 3px 3px 0 0; transition: height 0.3s;"
                         title="q_t: {q_t:.3f}"></div>
                    <div style="width: 16px; background: {p_color}; height: {p_h}px;
                                border-radius: 3px 3px 0 0; transition: height 0.3s;"
                         title="p_agg: {p_agg:.3f}"></div>
                </div>
                <div style="font-size: 12px; color: #374151; font-weight: 600;">
                    T{r['turn_index']}
                </div>
            </div>
        """

    chart_html += """
        </div>
        <div style="display: flex; align-items: center; gap: 16px; margin-top: 8px; padding-left: 12px;">
            <div style="display: flex; align-items: center; gap: 4px;">
                <div style="width: 12px; height: 12px; background: #93c5fd;
                            border-radius: 2px;"></div>
                <span style="font-size: 12px; color: #6b7280;">q_t (evidence)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 4px;">
                <div style="width: 12px; height: 12px; background: #f97316;
                            border-radius: 2px;"></div>
                <span style="font-size: 12px; color: #6b7280;">p_agg (cumulative)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 4px;">
                <div style="width: 40px; height: 1px; background: #ef4444;
                            border-top: 2px dashed #ef4444;"></div>
                <span style="font-size: 12px; color: #6b7280;">threshold (0.5)</span>
            </div>
        </div>
    </div>
    """
    return chart_html


# ============================================================
# Gradio App
# ============================================================
def create_app(engine: EarlyExitInferenceEngine):
    """Tạo Gradio app cho Early-Exit model với Noisy-OR."""

    # ── Batch analysis ──
    def analyze_batch(text_input, speaker_pattern):
        """Phân tích cả hội thoại (paste text)."""
        if not text_input.strip():
            return "<p>Vui lòng nhập hội thoại.</p>", ""

        lines = [l.strip() for l in text_input.strip().split("\n") if l.strip()]
        if not lines:
            return "<p>Không tìm thấy turn nào.</p>", ""

        # Parse speaker pattern
        if speaker_pattern.strip():
            pattern = [s.strip() for s in speaker_pattern.split(",")]
        else:
            pattern = ["người gọi", "người nghe"]

        messages = []
        for i, line in enumerate(lines):
            sp = pattern[i % len(pattern)]
            messages.append({"role": sp, "text": line})

        results = engine.predict_conversation(messages, "batch_analysis")
        html = build_result_html(results, messages)
        chart = build_prob_chart(results)
        return html, chart

    # ── Preset examples ──
    def run_example(example_name):
        """Chạy preset example."""
        all_examples = SCAM_EXAMPLES + LEGIT_EXAMPLES
        example = next((e for e in all_examples if e["name"] == example_name), None)
        if example is None:
            return "<p>Không tìm thấy example.</p>", "", ""

        messages = example["messages"]
        results = engine.predict_conversation(messages, f"example_{example_name}")
        html = build_result_html(results, messages)
        chart = build_prob_chart(results)

        text = "\n".join(m["text"] for m in messages)
        return html, chart, text

    # ── Streaming chat ──
    def chat_step(text, speaker, chat_history, dialogue_state):
        """Thêm 1 turn mới vào hội thoại."""
        if not text.strip():
            return chat_history, dialogue_state, "", ""

        if dialogue_state is None:
            dialogue_state = {"messages": [], "results": []}
            engine.reset("chat_session")

        result = engine.predict_turn("chat_session", text, speaker=speaker)
        msg = {"role": speaker, "text": text}

        dialogue_state["messages"].append(msg)
        dialogue_state["results"].append(result)

        q_t = result["q_t"]
        p_agg = result["p_agg"]
        icon = "🔴" if result["is_scam"] else "🟢"
        status = f" [{icon} q={q_t:.2f} p={p_agg:.2f}]"

        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": text + status})

        html = build_result_html(dialogue_state["results"], dialogue_state["messages"])
        chart = build_prob_chart(dialogue_state["results"])

        return chat_history, dialogue_state, html, chart

    def reset_chat():
        engine.reset("chat_session")
        return [], None, "", ""

    # ── Build UI ──
    with gr.Blocks(
        title="Early-Exit Scam Detection (Noisy-OR)",
        theme=gr.themes.Soft(primary_hue="red", secondary_hue="green"),
        css="""
        .gradio-container {
            max-width: 1050px !important;
            margin: 0 auto !important;
        }
        .result-panel { min-height: 200px; }
        """,
    ) as app:
        gr.Markdown("""
        # 🛡️ Early-Exit Scam Detection (Noisy-OR)
        **PhoBERT + Cross-Turn Attention + Noisy-OR Aggregation**

        Model phân tích từng turn hội thoại và đưa ra xác suất scam tích lũy:
        - **q_t**: evidence probability — bằng chứng scam tại turn t
        - **p_agg**: cumulative probability — xác suất tích lũy Noisy-OR
        """)

        with gr.Tabs():
            # ── Tab 1: Chat Mode ──
            with gr.Tab("💬 Chat Mode"):
                gr.Markdown("Nhập từng turn để mô phỏng hội thoại streaming real-time.")

                with gr.Row():
                    with gr.Column(scale=1):
                        chatbot = gr.Chatbot(
                            label="Hội thoại",
                            height=380,
                            type="messages",
                        )
                        with gr.Row():
                            chat_input = gr.Textbox(
                                label="Nội dung turn",
                                placeholder="Nhập nội dung turn...",
                                scale=4,
                            )
                            chat_speaker = gr.Dropdown(
                                choices=["người gọi", "người nghe", "unknown"],
                                value="người gọi",
                                label="Speaker",
                                scale=1,
                            )
                        with gr.Row():
                            send_btn = gr.Button("📤 Gửi turn", variant="primary")
                            reset_btn = gr.Button("🔄 Reset", variant="secondary")

                    with gr.Column(scale=1):
                        chat_detail = gr.HTML(
                            label="Chi tiết prediction",
                            elem_classes="result-panel",
                        )
                        chat_chart = gr.HTML(label="Timeline")

                chat_state = gr.State(None)

                send_btn.click(
                    chat_step,
                    [chat_input, chat_speaker, chatbot, chat_state],
                    [chatbot, chat_state, chat_detail, chat_chart],
                ).then(lambda: "", None, chat_input)

                chat_input.submit(
                    chat_step,
                    [chat_input, chat_speaker, chatbot, chat_state],
                    [chatbot, chat_state, chat_detail, chat_chart],
                ).then(lambda: "", None, chat_input)

                reset_btn.click(
                    reset_chat, None,
                    [chatbot, chat_state, chat_detail, chat_chart],
                )

            # ── Tab 2: Batch Mode ──
            with gr.Tab("📋 Batch Mode"):
                gr.Markdown("Paste cả hội thoại (mỗi dòng = 1 turn), chọn speaker pattern.")

                with gr.Row():
                    with gr.Column(scale=1):
                        batch_input = gr.Textbox(
                            label="Hội thoại (mỗi dòng = 1 turn)",
                            placeholder="Alo ai đấy?\nTôi là công an...\nCái gì ạ?\nChuyển tiền ngay!",
                            lines=8,
                        )
                        batch_speakers = gr.Textbox(
                            label="Speaker pattern (lặp lại theo vòng)",
                            value="người gọi, người nghe",
                            placeholder="người gọi, người nghe",
                        )
                        analyze_btn = gr.Button("🔍 Phân tích", variant="primary")

                    with gr.Column(scale=1):
                        batch_result = gr.HTML(
                            label="Kết quả",
                            elem_classes="result-panel",
                        )
                        batch_chart = gr.HTML(label="Timeline")

                analyze_btn.click(
                    analyze_batch,
                    [batch_input, batch_speakers],
                    [batch_result, batch_chart],
                )

            # ── Tab 3: Preset Examples ──
            with gr.Tab("📚 Preset Examples"):
                gr.Markdown("Chọn ví dụ có sẵn để test nhanh.")

                example_names = [e["name"] for e in SCAM_EXAMPLES + LEGIT_EXAMPLES]
                with gr.Row():
                    example_dropdown = gr.Dropdown(
                        choices=example_names,
                        label="Chọn ví dụ",
                        value=example_names[0],
                        scale=3,
                    )
                    run_example_btn = gr.Button("▶️ Chạy", variant="primary", scale=1)

                example_result = gr.HTML(label="Kết quả")
                example_chart = gr.HTML(label="Timeline")
                example_text = gr.Textbox(
                    label="Raw text",
                    lines=4,
                    interactive=False,
                )

                run_example_btn.click(
                    run_example,
                    [example_dropdown],
                    [example_result, example_chart, example_text],
                )

    return app


# ============================================================
# Launch helpers
# ============================================================
def launch_app(model_path=None, vncorenlp_dir=None, share=True):
    """
    Khởi động Gradio app.

    Parameters
    ----------
    model_path : str
        Đường dẫn đến thư mục best_model (chứa model.pt, config.json, tokenizer)
    vncorenlp_dir : str
        Đường dẫn đến thư mục vncorenlp (optional)
    share : bool
        Tạo public link (cho Colab)
    """
    if model_path is None:
        cfg = EarlyExitConfig()
        model_path = os.path.join(cfg.output_dir, "best_model")

    print(f"Loading model from: {model_path}")

    engine = EarlyExitInferenceEngine(
        model_path=model_path,
        vncorenlp_dir=vncorenlp_dir,
        threshold=0.5,
    )

    app = create_app(engine)
    app.launch(share=share, debug=False)


if __name__ == "__main__":
    launch_app(share=True)
