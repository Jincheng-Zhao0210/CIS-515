from __future__ import annotations

from pathlib import Path
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import gradio as gr

from pipeline.detector import detect_persons
from pipeline.scorer import compute_scores, PULSE_COLOUR
from pipeline.visualizer import annotate_frame, build_signal_chart, build_gauge


# ---------------------------------------------------------------------------
# Sample image
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLE_PATH = _DATA_DIR / "sample_classroom.jpg"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_rgb() -> np.ndarray:
    img = np.full((400, 600, 3), 26, dtype=np.uint8)
    cv2.putText(
        img,
        "No image",
        (185, 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (120, 120, 120),
        2,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _empty_fig() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    fig.patch.set_facecolor("#0f1923")
    ax.set_facecolor("#0f1923")
    ax.text(
        0.5,
        0.5,
        "—",
        ha="center",
        va="center",
        fontsize=24,
        color="#2d3748",
        transform=ax.transAxes,
    )
    ax.axis("off")
    fig.tight_layout()
    return fig


def _save_fig(fig: plt.Figure, name: str) -> str:
    path = Path(tempfile.gettempdir()) / name
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return str(path)


def _small_card(icon: str, label: str, value: str, border: str) -> str:
    return (
        f"<div style='display:inline-flex; flex-direction:column; align-items:center;"
        f"background:#fff; border-radius:12px; padding:11px 14px; min-width:90px;"
        f"text-align:center; box-shadow:0 2px 10px rgba(0,0,0,0.09);"
        f"border:1.5px solid {border}; margin:4px;'>"
        f"<span style='font-size:1.1rem;'>{icon}</span>"
        f"<span style='font-size:1.65rem; font-weight:800; color:#222; line-height:1.2;'>{value}</span>"
        f"<span style='font-size:0.7rem; color:#888; margin-top:2px;'>{label}</span>"
        f"</div>"
    )


_PULSE_EMOJI = {"High": "🟢", "Moderate": "🟡", "Low": "🔴"}


def _metrics_html(s: dict, is_demo: bool = False) -> str:
    pulse_colour = PULSE_COLOUR.get(s["pulse_label"], "#555")
    att_rate = s["attendance_rate"]
    eng_ratio = s["engaged_count"] / max(s["detected"], 1)

    banners = ""
    if is_demo:
        banners += (
            "<div style='background:#fff8e1; border:1px solid #ffc107; "
            "border-radius:9px; padding:8px 14px; margin-bottom:10px; "
            "font-size:0.83rem; color:#795548; font-family:sans-serif;'>"
            "🎭 <strong>Demo mode</strong> — bundled classroom image."
            "</div>"
        )

    if s["low_attendance"]:
        banners += (
            "<div style='background:#fdecea; border:1px solid #e74c3c; "
            "border-radius:9px; padding:8px 14px; margin-bottom:10px; "
            "font-size:0.83rem; color:#b71c1c; font-family:sans-serif;'>"
            f"⚠️ <strong>Low attendance:</strong> "
            f"{s['detected']} of {s['expected']} students detected "
            f"({att_rate:.0%}). Class score penalised."
            "</div>"
        )

    emoji = _PULSE_EMOJI.get(s["pulse_label"], "")
    badge = (
        f"<div style='margin-bottom:12px;'>"
        f"<span style='background:{pulse_colour}22; color:{pulse_colour}; "
        f"border:1.5px solid {pulse_colour}; border-radius:22px; "
        f"padding:5px 18px; font-size:0.94rem; font-weight:700; "
        f"font-family:sans-serif;'>"
        f"{emoji} {s['pulse_label']} Engagement"
        f"</span>"
        f"</div>"
    )

    pulse_card = (
        f"<div style='display:inline-flex; flex-direction:column; align-items:center; "
        f"background:#fff; border-radius:14px; padding:16px 26px; text-align:center; "
        f"box-shadow:0 4px 20px rgba(0,0,0,0.13); border:2.5px solid {pulse_colour}; "
        f"margin:4px 8px 4px 0;'>"
        f"<span style='font-size:1rem;'>📊</span>"
        f"<span style='font-size:3rem; font-weight:900; color:{pulse_colour}; "
        f"line-height:1.05; margin-top:2px;'>{s['class_score_pct']:.0f}%</span>"
        f"<span style='font-size:0.7rem; color:#888; margin-top:4px;'>Class Pulse</span>"
        f"</div>"
    )

    att_border = "#27ae60" if att_rate >= 0.65 else ("#f39c12" if att_rate >= 0.40 else "#e74c3c")
    eng_border = "#27ae60" if eng_ratio >= 0.5 else "#f39c12"
    neu_border = "#ddd"
    dis_border = "#e74c3c" if s["disengaged_count"] > s["engaged_count"] else "#ddd"
    det_border = "#2980b9"

    cards = (
        pulse_card
        + _small_card("👥", "Detected", str(s["detected"]), det_border)
        + _small_card("📋", "Attendance", f"{att_rate:.0%}", att_border)
        + _small_card("✅", "Engaged", str(s["engaged_count"]), eng_border)
        + _small_card("➡️", "Neutral", str(s["neutral_count"]), neu_border)
        + _small_card("❌", "Disengaged", str(s["disengaged_count"]), dis_border)
    )

    footer = (
        "<div style='background:linear-gradient(135deg,#0f3460,#0e4d8a); "
        "border-radius:10px; padding:10px 16px; margin-top:8px;'>"
        "<span style='color:rgba(255,255,255,0.75); font-size:0.78rem; font-family:sans-serif;'>"
        "🔒 Faces blurred · In-memory only · Aggregate class trends · No individual identified"
        "</span>"
        "</div>"
    )

    return (
        "<div style='font-family:sans-serif;'>"
        + banners
        + badge
        + f"<div style='display:flex; flex-wrap:wrap; align-items:stretch; gap:4px;'>{cards}</div>"
        + footer
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(image, expected_size, demo_mode):
    try:
        if demo_mode or image is None:
            if not SAMPLE_PATH.exists():
                return (
                    _empty_rgb(),
                    "<p style='color:red;'>Missing data/sample_classroom.jpg</p>",
                    None,
                    None,
                )
            pil_image = Image.open(SAMPLE_PATH).convert("RGB")
        else:
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image.astype("uint8")).convert("RGB")
            else:
                pil_image = image.convert("RGB")

        bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        persons = detect_persons(bgr)
        scores = compute_scores(persons, int(expected_size))
        annotated = annotate_frame(bgr, persons)

        signal_chart = build_signal_chart(scores)
        gauge_fig = build_gauge(scores["class_score"], scores["pulse_label"])

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        html = _metrics_html(scores, is_demo=demo_mode)

        signal_chart_path = _save_fig(signal_chart, "signal_chart.png")
        gauge_path = _save_fig(gauge_fig, "gauge_chart.png")

        return annotated_rgb, html, gauge_path, signal_chart_path

    except Exception as exc:
        return (
            _empty_rgb(),
            f"<p style='color:red; font-family:sans-serif;'>Error: {exc}</p>",
            None,
            None,
        )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
# 🎓 Classroom Engagement Detector

Behavioral proxy analysis for classroom engagement.

### What it does
- Detects students in a classroom image
- Estimates class engagement signals
- Shows annotated image, KPI summary, and charts

### Privacy
- No face recognition
- Faces blurred
- Aggregate class-level results only
"""

HOW_IT_WORKS = """
### Detection pipeline

1. CLAHE equalisation  
2. Haar face cascade  
3. HOG pedestrian detector  
4. Non-max suppression  
5. Engagement scoring signals:
   - head pose
   - posture
   - hand raise
   - phone use
   - talking
"""

with gr.Blocks(theme=gr.themes.Soft(), title="Classroom Engagement Detector") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload classroom image")
            expected_size = gr.Number(value=30, precision=0, label="Expected class size")
            demo_mode = gr.Checkbox(value=True, label="Demo mode")
            analyze_btn = gr.Button("Analyze", variant="primary")
            gr.Markdown(HOW_IT_WORKS)

        with gr.Column(scale=2):
            image_output = gr.Image(label="Annotated classroom view")
            metrics_output = gr.HTML(label="Class summary")

            with gr.Row():
                gauge_output = gr.Image(label="Class pulse gauge")
                signal_output = gr.Image(label="Signal chart")

    analyze_btn.click(
        fn=run_analysis,
        inputs=[image_input, expected_size, demo_mode],
        outputs=[image_output, metrics_output, gauge_output, signal_output],
    )

if __name__ == "__main__":
    demo.launch()
