"""
Gradio Visualization cho Baseline2: Early-Exit with Weighted Loss.

Giao diện trực quan để demo model:
  - Chat Mode: nhập từng turn, xem kết quả real-time (streaming)
  - Batch Mode: paste cả hội thoại, xem timeline predictions
  - Preset Examples để test nhanh

Hiển thị multi-class predictions (LEGIT/SCAM/AMBIGUOUS) với:
  - Probability bars per class per turn
  - Timeline chart theo turn
  - Cross-attention context visualization

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
            {"speaker_role": "normal", "text": "Alo ai đấy ạ?"},
            {"speaker_role": "scammer", "text": "Tôi là Đại úy Nguyễn Văn Hùng, Công an thành phố. Bạn đang bị điều tra vì liên quan đến đường dây rửa tiền."},
            {"speaker_role": "normal", "text": "Cái gì ạ? Tôi không biết gì cả."},
            {"speaker_role": "scammer", "text": "Chuyển tiền vào tài khoản an toàn ngay trong vòng 30 phút, nếu không sẽ bị bắt."},
        ],
    },
    {
        "name": "Lừa đảo ngân hàng",
        "messages": [
            {"speaker_role": "normal", "text": "Alo xin nghe?"},
            {"speaker_role": "scammer", "text": "Xin chào, tôi gọi từ ngân hàng BIDV. Thẻ ATM của bạn sắp hết hạn và cần gia hạn trực tuyến ngay hôm nay."},
            {"speaker_role": "normal", "text": "Thẻ tôi vẫn dùng được bình thường mà?"},
            {"speaker_role": "scammer", "text": "Hệ thống mới cập nhật ạ. Bạn cung cấp số thẻ, ngày hết hạn và mã CVV để em xử lý nhé."},
        ],
    },
    {
        "name": "Lừa đảo đầu tư crypto",
        "messages": [
            {"speaker_role": "scammer", "text": "Chào bạn, bạn có muốn đầu tư Bitcoin với lãi suất 500% trong 30 ngày không?"},
            {"speaker_role": "normal", "text": "Nghe hấp dẫn nhỉ, nhưng làm sao được lãi cao vậy?"},
            {"speaker_role": "scammer", "text": "Chúng tôi có đội ngũ chuyên gia giao dịch AI tự động. Bạn chỉ cần nạp tối thiểu 5 triệu vào ví điện tử."},
            {"speaker_role": "normal", "text": "Có an toàn không?"},
            {"speaker_role": "scammer", "text": "Hoàn toàn an toàn, đã có hàng nghìn người tham gia thành công. Nhưng chương trình chỉ mở thêm 24 giờ nữa thôi."},
        ],
    },
]

LEGIT_EXAMPLES = [
    {
        "name": "Đặt pizza",
        "messages": [
            {"speaker_role": "normal", "text": "Alo Pizza Hut phải không ạ?"},
            {"speaker_role": "normal", "text": "Dạ đúng rồi ạ. Anh chị muốn đặt gì ạ?"},
            {"speaker_role": "normal", "text": "Cho tôi 1 pizza hải sản cỡ lớn và 2 lon Pepsi nhé."},
            {"speaker_role": "normal", "text": "Dạ tổng 285 nghìn ạ. Anh cho em địa chỉ giao hàng."},
        ],
    },
    {
        "name": "Hẹn đi cà phê",
        "messages": [
            {"speaker_role": "normal", "text": "Alo bạn ơi, chiều nay đi cà phê không?"},
            {"speaker_role": "normal", "text": "Oke, mấy giờ?"},
            {"speaker_role": "normal", "text": "3 giờ nhé, quán trên Nguyễn Huệ."},
            {"speaker_role": "normal", "text": "Được rồi, hẹn gặp!"},
        ],
    },
]


# ============================================================
# Color & styling helpers
# ============================================================
CLASS_COLORS = {
    "LEGIT": "#22c55e",     # green
    "SCAM": "#ef4444",      # red
    "AMBIGUOUS": "#f59e0b",  # amber
}

CLASS_ICONS = {
    "LEGIT": "🟢",
    "SCAM": "🔴",
    "AMBIGUOUS": "🟡",
}

CLASS_BG = {
    "LEGIT": "#f0fdf4",
    "SCAM": "#fef2f2",
    "AMBIGUOUS": "#fffbeb",
}


def prediction_to_color(pred_name):
    """Màu theo predicted class."""
    return CLASS_COLORS.get(pred_name, "#6b7280")


def scam_prob_to_color(scam_prob):
    """Màu theo xác suất SCAM."""
    if scam_prob < 0.3:
        return "#22c55e"
    elif scam_prob < 0.5:
        return "#eab308"
    elif scam_prob < 0.7:
        return "#f97316"
    else:
        return "#ef4444"


# ============================================================
# Build HTML result
# ============================================================
def build_result_html(results, messages):
    """Tạo HTML trực quan cho kết quả multi-class."""
    html_parts = []

    html_parts.append("""
    <div style="font-family: 'Segoe UI', sans-serif; max-width: 750px; margin: 0 auto;">
    """)

    for r, msg in zip(results, messages):
        pred = r["prediction"]
        probs = r["probabilities"]
        scam_prob = probs.get("SCAM", 0.0)
        pred_color = prediction_to_color(pred)
        pred_icon = CLASS_ICONS.get(pred, "⚪")
        speaker = msg.get("speaker_role", "unknown")
        speaker_icon = "👤" if speaker == "normal" else "⚠️"
        speaker_color = "#6b7280" if speaker == "normal" else "#dc2626"
        bg_color = CLASS_BG.get(pred, "#f9fafb")

        # Probability bars per class
        prob_bars_html = ""
        for cls_name in ["LEGIT", "SCAM", "AMBIGUOUS"]:
            p = probs.get(cls_name, 0.0)
            cls_color = CLASS_COLORS.get(cls_name, "#9ca3af")
            bar_w = int(p * 100)
            prob_bars_html += f"""
                <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 2px;">
                    <span style="font-size: 11px; color: #6b7280; width: 72px; text-align: right;">{cls_name}</span>
                    <div style="flex: 1; background: #e5e7eb; border-radius: 3px; height: 8px; overflow: hidden;">
                        <div style="background: {cls_color}; width: {bar_w}%; height: 100%; border-radius: 3px;
                                    transition: width 0.3s;"></div>
                    </div>
                    <span style="font-size: 11px; color: #374151; width: 40px;">{p:.1%}</span>
                </div>
            """

        html_parts.append(f"""
        <div style="margin-bottom: 14px; padding: 14px 18px; border-radius: 12px;
                    border-left: 5px solid {pred_color};
                    background: {bg_color}; box-shadow: 0 1px 3px rgba(0,0,0,0.06);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; color: {speaker_color}; font-size: 14px;">
                    {speaker_icon} Turn {r['turn_index']} ({speaker})
                </span>
                <span style="font-weight: 700; color: {pred_color}; font-size: 15px;
                             padding: 2px 10px; border-radius: 6px;
                             background: {pred_color}18;">
                    {pred_icon} {pred}
                </span>
            </div>
            <div style="color: #374151; font-size: 14px; margin-bottom: 10px; line-height: 1.5;">
                {msg['text']}
            </div>
            <div style="padding: 6px 0;">
                {prob_bars_html}
            </div>
        </div>
        """)

    # Summary
    final = results[-1]
    final_pred = final["prediction"]
    final_probs = final["probabilities"]
    scam_prob = final_probs.get("SCAM", 0.0)

    # Check if any turn predicted SCAM
    scam_turns = [r for r in results if r["prediction"] == "SCAM"]
    first_scam_turn = scam_turns[0]["turn_index"] if scam_turns else None

    if final_pred == "SCAM":
        summary_color = "#dc2626"
        summary_bg = "#fef2f2"
        summary_icon = "🚨"
        summary_text = f"CẢNH BÁO: Phát hiện lừa đảo! (từ Turn {first_scam_turn})"
    elif final_pred == "AMBIGUOUS":
        summary_color = "#d97706"
        summary_bg = "#fffbeb"
        summary_icon = "⚠️"
        summary_text = "NGHI NGỜ: Hội thoại có dấu hiệu đáng ngờ"
    else:
        summary_color = "#16a34a"
        summary_bg = "#f0fdf4"
        summary_icon = "✅"
        summary_text = "AN TOÀN: Không phát hiện dấu hiệu lừa đảo"

    total = len(results)
    pred_counts = {}
    for r in results:
        p = r["prediction"]
        pred_counts[p] = pred_counts.get(p, 0) + 1
    counts_str = " | ".join(f"{k}: {v}/{total}" for k, v in sorted(pred_counts.items()))

    html_parts.append(f"""
    <div style="margin-top: 18px; padding: 16px 20px; border-radius: 12px;
                background: {summary_bg}; border: 2px solid {summary_color};
                box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
        <div style="font-weight: 700; color: {summary_color}; font-size: 17px;">
            {summary_icon} {summary_text}
        </div>
        <div style="color: #6b7280; font-size: 13px; margin-top: 6px;">
            Final prediction: <b>{final_pred}</b>
            (SCAM: {scam_prob:.1%}) &nbsp;|&nbsp; {counts_str}
        </div>
    </div>
    """)

    html_parts.append("</div>")
    return "\n".join(html_parts)


# ============================================================
# Build probability chart (HTML/CSS)
# ============================================================
def build_prob_chart(results):
    """Tạo chart xác suất multi-class theo turn bằng HTML/CSS."""
    chart_html = """
    <div style="font-family: 'Segoe UI', sans-serif; max-width: 750px; margin: 18px auto 0;">
        <div style="font-weight: 600; color: #374151; margin-bottom: 12px; font-size: 15px;">
            📊 Prediction Timeline
        </div>
        <div style="display: flex; align-items: flex-end; gap: 6px; height: 200px;
                    padding: 12px; background: #f9fafb; border-radius: 12px;
                    border: 1px solid #e5e7eb;">
    """

    for r in results:
        probs = r["probabilities"]
        pred = r["prediction"]

        # Stacked bar cho 3 classes
        legit_p = probs.get("LEGIT", 0)
        scam_p = probs.get("SCAM", 0)
        ambig_p = probs.get("AMBIGUOUS", 0)

        bar_height = 160
        legit_h = int(legit_p * bar_height)
        scam_h = int(scam_p * bar_height)
        ambig_h = int(ambig_p * bar_height)

        # Border color = predicted class
        border_color = CLASS_COLORS.get(pred, "#9ca3af")

        chart_html += f"""
            <div style="flex: 1; display: flex; flex-direction: column; align-items: center;
                        justify-content: flex-end;">
                <div style="font-size: 10px; color: #6b7280; margin-bottom: 3px;
                            font-weight: 600;">{pred[:3]}</div>
                <div style="width: 100%; max-width: 48px; display: flex; flex-direction: column;
                            border-radius: 6px 6px 0 0; overflow: hidden;
                            border: 2px solid {border_color}; border-bottom: none;">
                    <div style="background: {CLASS_COLORS['SCAM']}; height: {scam_h}px;
                                transition: height 0.3s;" title="SCAM: {scam_p:.2f}"></div>
                    <div style="background: {CLASS_COLORS['AMBIGUOUS']}; height: {ambig_h}px;
                                transition: height 0.3s;" title="AMBIGUOUS: {ambig_p:.2f}"></div>
                    <div style="background: {CLASS_COLORS['LEGIT']}; height: {legit_h}px;
                                transition: height 0.3s;" title="LEGIT: {legit_p:.2f}"></div>
                </div>
                <div style="font-size: 12px; color: #374151; margin-top: 5px;
                            font-weight: 600;">T{r['turn_index']}</div>
            </div>
        """

    chart_html += """
        </div>
        <div style="display: flex; align-items: center; gap: 16px; margin-top: 8px; padding-left: 12px;">
    """

    for cls_name, cls_color in CLASS_COLORS.items():
        chart_html += f"""
            <div style="display: flex; align-items: center; gap: 4px;">
                <div style="width: 12px; height: 12px; background: {cls_color};
                            border-radius: 2px;"></div>
                <span style="font-size: 12px; color: #6b7280;">{cls_name}</span>
            </div>
        """

    chart_html += """
        </div>
    </div>
    """
    return chart_html


# ============================================================
# Gradio App
# ============================================================
def create_app(engine: EarlyExitInferenceEngine):
    """Tạo Gradio app cho Early-Exit model."""

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
            pattern = [s.strip().lower() for s in speaker_pattern.split(",")]
        else:
            pattern = ["normal", "scammer"]

        messages = []
        for i, line in enumerate(lines):
            sp = pattern[i % len(pattern)]
            messages.append({"speaker_role": sp, "text": line})

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
        msg = {"speaker_role": speaker, "text": text}

        dialogue_state["messages"].append(msg)
        dialogue_state["results"].append(result)

        pred = result["prediction"]
        scam_p = result["probabilities"].get("SCAM", 0)
        icon = CLASS_ICONS.get(pred, "⚪")
        status = f" [{icon} {pred} | SCAM: {scam_p:.0%}]"

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
        title="Early-Exit Scam Detection",
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
        # 🛡️ Early-Exit Scam Detection
        **PhoBERT + Cross-Turn Attention + Weighted Loss**

        Model phân tích từng turn hội thoại và đưa ra dự đoán multi-class
        (LEGIT / SCAM / AMBIGUOUS) với cross-attention trên các turn trước.
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
                                choices=["normal", "scammer", "unknown"],
                                value="normal",
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
                            value="normal, scammer",
                            placeholder="normal, scammer",
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
