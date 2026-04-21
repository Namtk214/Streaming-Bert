"""
Gradio Visualization cho Streaming Scam Detection.

Giao dien truc quan de demo model:
  - Chat-style: nhap tung turn, xem ket qua real-time
  - Batch mode: paste ca hoi thoai, xem timeline probability
  - Preset examples de test nhanh

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

# Import streaming modules
_streaming_dir = os.path.dirname(os.path.abspath(__file__))
if _streaming_dir not in sys.path:
    sys.path.insert(0, _streaming_dir)

from config import StreamingConfig
from infer_stream import StreamingInferenceEngine

# ============================================================
# Preset examples
# ============================================================
SCAM_EXAMPLES = [
    {
        "name": "Gia mao cong an",
        "messages": [
            {"speaker_role": "normal", "text": "Alo ai day a?"},
            {"speaker_role": "scammer", "text": "Toi la Dai uy Nguyen Van Hung, Cong an thanh pho. Ban dang bi dieu tra vi lien quan den duong day rua tien."},
            {"speaker_role": "normal", "text": "Cai gi a? Toi khong biet gi ca."},
            {"speaker_role": "scammer", "text": "Chuyen tien vao tai khoan an toan ngay trong vong 30 phut, neu khong se bi bat."},
        ],
    },
    {
        "name": "Lua dao ngan hang",
        "messages": [
            {"speaker_role": "normal", "text": "Alo xin nghe?"},
            {"speaker_role": "scammer", "text": "Xin chao, toi goi tu ngan hang BIDV. The ATM cua ban sap het han va can gia han truc tuyen ngay hom nay."},
            {"speaker_role": "normal", "text": "The toi van dung duoc binh thuong ma?"},
            {"speaker_role": "scammer", "text": "He thong moi cap nhat a. Ban cung cap so the, ngay het han va ma CVV de em xu ly nhe."},
        ],
    },
    {
        "name": "Lua dao dau tu crypto",
        "messages": [
            {"speaker_role": "scammer", "text": "Chao ban, ban co muon dau tu Bitcoin voi lai suat 500% trong 30 ngay khong?"},
            {"speaker_role": "normal", "text": "Nghe hap dan nhi, nhung lam sao duoc lai cao vay?"},
            {"speaker_role": "scammer", "text": "Chung toi co doi ngu chuyen gia giao dich AI tu dong. Ban chi can nap toi thieu 5 trieu vao vi dien tu."},
            {"speaker_role": "normal", "text": "Co an toan khong?"},
            {"speaker_role": "scammer", "text": "Hoan toan an toan, da co hang nghin nguoi tham gia thanh cong. Nhung chuong trinh chi mo them 24 gio nua thoi."},
        ],
    },
]

LEGIT_EXAMPLES = [
    {
        "name": "Dat pizza",
        "messages": [
            {"speaker_role": "normal", "text": "Alo Pizza Hut phai khong a?"},
            {"speaker_role": "normal", "text": "Da dung roi a. Anh chi muon dat gi a?"},
            {"speaker_role": "normal", "text": "Cho toi 1 pizza hai san co lon va 2 lon Pepsi nhe."},
            {"speaker_role": "normal", "text": "Da tong 285 nghin a. Anh cho em dia chi giao hang."},
        ],
    },
    {
        "name": "Hen di ca phe",
        "messages": [
            {"speaker_role": "normal", "text": "Alo ban oi, chieu nay di ca phe khong?"},
            {"speaker_role": "normal", "text": "Oke, may gio?"},
            {"speaker_role": "normal", "text": "3 gio nhe, quan tren Nguyen Hue."},
            {"speaker_role": "normal", "text": "Duoc roi, hen gap!"},
        ],
    },
]


# ============================================================
# Color helpers
# ============================================================
def prob_to_color(prob):
    """Chuyen probability thanh mau (xanh → do)."""
    if prob < 0.3:
        return "#22c55e"  # green
    elif prob < 0.5:
        return "#eab308"  # yellow
    elif prob < 0.7:
        return "#f97316"  # orange
    else:
        return "#ef4444"  # red


def prob_to_label(prob, threshold=0.5):
    if prob >= threshold:
        return "SCAM"
    return "OK"


# ============================================================
# Build HTML result
# ============================================================
def build_result_html(results, messages):
    """Tao HTML truc quan cho ket qua."""
    html_parts = []

    # Header
    html_parts.append("""
    <div style="font-family: 'Segoe UI', sans-serif; max-width: 700px; margin: 0 auto;">
    """)

    # Turn-by-turn
    for r, msg in zip(results, messages):
        prob = r["probability"]
        color = prob_to_color(prob)
        label = prob_to_label(prob)
        speaker = msg.get("speaker_role", "unknown")
        speaker_icon = "&#128100;" if speaker == "normal" else "&#9888;&#65039;"
        speaker_color = "#6b7280" if speaker == "normal" else "#dc2626"

        # Progress bar
        bar_width = int(prob * 100)

        html_parts.append(f"""
        <div style="margin-bottom: 12px; padding: 12px 16px; border-radius: 10px;
                    border-left: 4px solid {color};
                    background: {'#fef2f2' if label == 'SCAM' else '#f0fdf4'};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-weight: 600; color: {speaker_color};">
                    {speaker_icon} Turn {r['turn_index']} ({speaker})
                </span>
                <span style="font-weight: 700; color: {color}; font-size: 14px;">
                    {label} ({prob:.1%})
                </span>
            </div>
            <div style="color: #374151; font-size: 14px; margin-bottom: 8px;">
                {msg['text']}
            </div>
            <div style="background: #e5e7eb; border-radius: 4px; height: 6px; overflow: hidden;">
                <div style="background: {color}; width: {bar_width}%; height: 100%; border-radius: 4px;
                            transition: width 0.3s;"></div>
            </div>
        </div>
        """)

    # Summary
    max_prob = max(r["probability"] for r in results)
    scam_turns = sum(1 for r in results if r["is_scam"])
    total_turns = len(results)

    if scam_turns > 0:
        first_alert = next(r["turn_index"] for r in results if r["is_scam"])
        summary_color = "#dc2626"
        summary_bg = "#fef2f2"
        summary_text = f"CANH BAO: Phat hien dau hieu lua dao tu Turn {first_alert} ({scam_turns}/{total_turns} turns)"
    else:
        first_alert = None
        summary_color = "#16a34a"
        summary_bg = "#f0fdf4"
        summary_text = f"AN TOAN: Khong phat hien dau hieu lua dao ({total_turns} turns)"

    html_parts.append(f"""
    <div style="margin-top: 16px; padding: 14px 18px; border-radius: 10px;
                background: {summary_bg}; border: 2px solid {summary_color};">
        <div style="font-weight: 700; color: {summary_color}; font-size: 16px;">
            {summary_text}
        </div>
        <div style="color: #6b7280; font-size: 13px; margin-top: 4px;">
            Max probability: {max_prob:.1%}
        </div>
    </div>
    """)

    html_parts.append("</div>")
    return "\n".join(html_parts)


# ============================================================
# Build probability chart (text-based, no matplotlib needed)
# ============================================================
def build_prob_chart(results):
    """Tao chart probability bang HTML/CSS."""
    chart_html = """
    <div style="font-family: 'Segoe UI', sans-serif; max-width: 700px; margin: 16px auto 0;">
        <div style="font-weight: 600; color: #374151; margin-bottom: 10px; font-size: 15px;">
            Scam Probability Timeline
        </div>
        <div style="display: flex; align-items: flex-end; gap: 8px; height: 160px;
                    padding: 10px; background: #f9fafb; border-radius: 10px; border: 1px solid #e5e7eb;">
    """

    # Threshold line position
    threshold_bottom = int(0.5 * 140)

    for r in results:
        prob = r["probability"]
        bar_h = max(int(prob * 140), 4)
        color = prob_to_color(prob)

        chart_html += f"""
            <div style="flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end;">
                <div style="font-size: 11px; color: #6b7280; margin-bottom: 4px;">{prob:.2f}</div>
                <div style="width: 100%; max-width: 50px; height: {bar_h}px; background: {color};
                            border-radius: 4px 4px 0 0; transition: height 0.3s;"></div>
                <div style="font-size: 12px; color: #374151; margin-top: 4px; font-weight: 500;">T{r['turn_index']}</div>
            </div>
        """

    chart_html += f"""
        </div>
        <div style="display: flex; align-items: center; gap: 6px; margin-top: 6px; padding-left: 10px;">
            <div style="width: 20px; height: 2px; background: #ef4444; border-top: 2px dashed #ef4444;"></div>
            <span style="font-size: 11px; color: #6b7280;">Threshold (0.5)</span>
            <div style="margin-left: 16px; display: flex; gap: 12px;">
                <span style="font-size: 11px;"><span style="color: #22c55e;">&#9632;</span> Safe</span>
                <span style="font-size: 11px;"><span style="color: #eab308;">&#9632;</span> Suspicious</span>
                <span style="font-size: 11px;"><span style="color: #ef4444;">&#9632;</span> Scam</span>
            </div>
        </div>
    </div>
    """
    return chart_html


# ============================================================
# Gradio App
# ============================================================
def create_app(engine: StreamingInferenceEngine):
    """Tao Gradio app."""

    # ── Batch analysis ──
    def analyze_batch(text_input, speaker_pattern):
        """Phan tich ca hoi thoai (paste text)."""
        if not text_input.strip():
            return "<p>Vui long nhap hoi thoai.</p>", ""

        lines = [l.strip() for l in text_input.strip().split("\n") if l.strip()]
        if not lines:
            return "<p>Khong tim thay turn nao.</p>", ""

        # Parse speaker pattern
        speakers = []
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
        """Chay preset example."""
        all_examples = SCAM_EXAMPLES + LEGIT_EXAMPLES
        example = next((e for e in all_examples if e["name"] == example_name), None)
        if example is None:
            return "<p>Khong tim thay example.</p>", "", ""

        messages = example["messages"]
        results = engine.predict_conversation(messages, f"example_{example_name}")
        html = build_result_html(results, messages)
        chart = build_prob_chart(results)

        # Fill text box
        text = "\n".join(m["text"] for m in messages)
        speakers = ",".join(m["speaker_role"] for m in messages)

        return html, chart, text

    # ── Streaming chat ──
    def chat_step(text, speaker, chat_history, dialogue_state):
        """Them 1 turn moi vao hoi thoai."""
        if not text.strip():
            return chat_history, dialogue_state, "", ""

        if dialogue_state is None:
            dialogue_state = {"messages": [], "results": [], "dlg_id": "chat_session"}
            engine.reset("chat_session")

        # Predict
        result = engine.predict_turn("chat_session", text, speaker)
        msg = {"speaker_role": speaker, "text": text}

        dialogue_state["messages"].append(msg)
        dialogue_state["results"].append(result)

        # Build chat display
        prob = result["probability"]
        label = prob_to_label(prob)
        color = prob_to_color(prob)
        status = f" [{label} {prob:.1%}]"

        if speaker == "scammer":
            chat_history = chat_history or []
            chat_history.append([None, text + status])
        else:
            chat_history = chat_history or []
            chat_history.append([text + status, None])

        # Build result + chart
        html = build_result_html(dialogue_state["results"], dialogue_state["messages"])
        chart = build_prob_chart(dialogue_state["results"])

        return chat_history, dialogue_state, html, chart

    def reset_chat():
        """Reset hoi thoai."""
        engine.reset("chat_session")
        return [], None, "", ""

    # ── Build UI ──
    with gr.Blocks(
        title="Streaming Scam Detection",
        theme=gr.themes.Soft(
            primary_hue="red",
            secondary_hue="green",
        ),
        css="""
        .gradio-container {
            max-width: 960px !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        .main { display: flex; justify-content: center; }
        .result-panel { min-height: 200px; }
        """
    ) as app:
        gr.Markdown("""
        # Streaming Scam Detection
        **PhoBERT + GRU** - Phat hien lua dao theo thoi gian thuc

        Model phan tich tung turn hoi thoai va dua ra xac suat scam.
        """)

        with gr.Tabs():
            # ── Tab 1: Chat Mode ──
            with gr.Tab("Chat Mode"):
                gr.Markdown("Nhap tung turn de mo phong hoi thoai streaming.")

                with gr.Row():
                    with gr.Column(scale=1):
                        chatbot = gr.Chatbot(label="Hoi thoai", height=350)
                        with gr.Row():
                            chat_input = gr.Textbox(
                                label="Nhap noi dung turn",
                                placeholder="Nhap text...",
                                scale=3,
                            )
                            chat_speaker = gr.Radio(
                                ["normal", "scammer"],
                                value="normal",
                                label="Speaker",
                                scale=1,
                            )
                        with gr.Row():
                            send_btn = gr.Button("Gui turn", variant="primary")
                            reset_btn = gr.Button("Reset", variant="secondary")

                    with gr.Column(scale=1):
                        chat_detail = gr.HTML(label="Chi tiet", elem_classes="result-panel")
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

                reset_btn.click(reset_chat, None, [chatbot, chat_state, chat_detail, chat_chart])

            # ── Tab 2: Batch Mode ──
            with gr.Tab("Batch Mode"):
                gr.Markdown("Paste ca hoi thoai (moi dong = 1 turn), chon speaker pattern.")

                with gr.Row():
                    with gr.Column(scale=1):
                        batch_input = gr.Textbox(
                            label="Hoi thoai (moi dong = 1 turn)",
                            placeholder="Alo ai day?\nToi la cong an...\nCai gi a?\nChuyen tien ngay!",
                            lines=8,
                        )
                        batch_speakers = gr.Textbox(
                            label="Speaker pattern (lap lai theo vong)",
                            value="normal, scammer",
                            placeholder="normal, scammer",
                        )
                        analyze_btn = gr.Button("Phan tich", variant="primary")

                    with gr.Column(scale=1):
                        batch_result = gr.HTML(label="Ket qua", elem_classes="result-panel")
                        batch_chart = gr.HTML(label="Timeline")

                analyze_btn.click(
                    analyze_batch,
                    [batch_input, batch_speakers],
                    [batch_result, batch_chart],
                )

            # ── Tab 3: Examples ──
            with gr.Tab("Preset Examples"):
                gr.Markdown("Chon vi du co san de test nhanh.")

                example_names = [e["name"] for e in SCAM_EXAMPLES + LEGIT_EXAMPLES]
                example_dropdown = gr.Dropdown(
                    choices=example_names,
                    label="Chon vi du",
                    value=example_names[0],
                )
                run_example_btn = gr.Button("Chay", variant="primary")

                example_result = gr.HTML(label="Ket qua")
                example_chart = gr.HTML(label="Timeline")
                example_text = gr.Textbox(label="Raw text", lines=4, interactive=False)

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
    Khoi dong Gradio app.

    Parameters
    ----------
    model_path : str
        Duong dan den thu muc best_model (chua model.pt, config.json, tokenizer)
    vncorenlp_dir : str
        Duong dan den thu muc vncorenlp (optional)
    share : bool
        Tao public link (cho Colab)
    """
    if model_path is None:
        cfg = StreamingConfig()
        model_path = os.path.join(cfg.output_dir, "best_model")

    print(f"Loading model from: {model_path}")

    engine = StreamingInferenceEngine(
        model_path=model_path,
        vncorenlp_dir=vncorenlp_dir,
        threshold=0.5,
    )

    app = create_app(engine)
    app.launch(share=share, debug=False)


if __name__ == "__main__":
    launch_app(share=False)
