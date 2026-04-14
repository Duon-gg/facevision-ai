"""
app.py — Dự đoán Thuộc tính Khuôn mặt
Giao diện Light Professional · Clean Design · Streamlit
"""

import io
import os
import csv
import json
import time
import base64
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.inference.analyzer import FaceAnalyzer
from src.utils.constants import GENDER_LABELS, RACE_LABELS


# Nhãn tiếng Việt
GENDER_VI = {0: "Nam", 1: "Nữ"}
RACE_VI = {0: "Châu Âu-Mỹ", 1: "Châu Phi", 2: "Châu Á", 3: "Khác"}  # Sắc tộc


# ────────────────────────────────────────────────────────────
# Cấu hình trang
# ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceVision AI · Dự đoán Thuộc tính Khuôn mặt",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ────────────────────────────────────────────────────────────
# CSS — Light Professional / Clean Design
# ────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ══════ CSS Variables ══════ */
    :root {
        --bg-deep: #F8FAFC;
        --bg-primary: #FFFFFF;
        --bg-card: #F1F5F9;
        --bg-glass: rgba(255, 255, 255, 0.85);
        --border-subtle: #E2E8F0;
        --border-glow: rgba(79, 70, 229, 0.3);
        --accent-primary: #4F46E5;
        --accent-secondary: #6366F1;
        --accent-cyan: #0891B2;
        --accent-emerald: #059669;
        --accent-rose: #DB2777;
        --accent-amber: #D97706;
        --text-primary: #1E293B;
        --text-secondary: #475569;
        --text-muted: #94A3B8;
        --glow-purple: 0 1px 3px rgba(0, 0, 0, 0.08);
        --glow-cyan: 0 1px 2px rgba(0, 0, 0, 0.04);
        --radius-lg: 16px;
        --radius-md: 12px;
        --radius-sm: 8px;
    }

    /* ══════ Globals ══════ */
    .stApp {
        font-family: 'Inter', sans-serif !important;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }

    /* ══════ Scrollbar ══════ */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #F1F5F9; }
    ::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 6px; }

    /* ══════ Sidebar ══════ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'Inter', sans-serif !important;
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-cyan));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* ══════ Hero Header ══════ */
    .hero {
        text-align: center;
        padding: 28px 20px 18px;
        margin-bottom: 20px;
        background: linear-gradient(135deg, rgba(79,70,229,0.04) 0%, rgba(8,145,178,0.03) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(79,70,229,0.03) 0%, transparent 50%),
                    radial-gradient(circle at 70% 50%, rgba(8,145,178,0.02) 0%, transparent 50%);
        animation: hero-pulse 8s ease-in-out infinite;
    }
    @keyframes hero-pulse {
        0%, 100% { opacity: 0.5; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.05); }
    }
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1E293B 0%, var(--accent-primary) 50%, var(--accent-cyan) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        position: relative;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-top: 6px;
        position: relative;
        letter-spacing: 0.04em;
    }
    .hero-badges {
        display: flex;
        justify-content: center;
        gap: 8px;
        margin-top: 14px;
        flex-wrap: wrap;
        position: relative;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        border: 1px solid var(--border-subtle);
        background: rgba(241, 245, 249, 0.9);
        color: var(--text-secondary);
        backdrop-filter: blur(8px);
    }

    /* ══════ Column Glass Effect ══════ */
    [data-testid="stColumn"] > div {
        background: var(--bg-glass);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 22px;
        min-height: 65vh;
        box-shadow: var(--glow-purple);
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    [data-testid="stColumn"] > div:hover {
        border-color: var(--border-glow);
        box-shadow: var(--glow-purple), var(--glow-cyan);
    }
    .panel-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 18px;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--border-subtle);
    }
    .panel-icon {
        width: 32px;
        height: 32px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }
    .panel-icon--input { background: linear-gradient(135deg, #6C63FF, #818CF8); }
    .panel-icon--output { background: linear-gradient(135deg, #22D3EE, #34D399); }
    .panel-label {
        font-size: 0.8rem;
        font-weight: 700;
        color: var(--text-primary);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-family: 'Inter', sans-serif;
    }

    /* ══════ Image Display ══════ */
    [data-testid="stImage"] {
        border-radius: var(--radius-md);
        overflow: hidden;
        border: 1px solid var(--border-subtle);
    }
    [data-testid="stImage"] img {
        max-height: 400px;
        object-fit: contain;
        width: 100%;
    }

    /* ══════ Probability Bars ══════ */
    .prob-group { margin: 6px 0 14px; }
    .prob-row {
        display: flex;
        align-items: center;
        margin: 5px 0;
        gap: 8px;
    }
    .prob-label {
        width: 88px;
        font-size: 0.76rem;
        color: var(--text-secondary);
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        text-align: right;
    }
    .prob-track {
        flex: 1;
        height: 8px;
        background: rgba(226, 232, 240, 0.8);
        border-radius: 4px;
        overflow: hidden;
        position: relative;
    }
    .prob-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        position: relative;
    }
    .prob-fill::after {
        content: '';
        position: absolute;
        right: 0;
        top: 0;
        width: 6px;
        height: 100%;
        background: rgba(255,255,255,0.3);
        border-radius: 0 4px 4px 0;
    }
    .prob-pct {
        width: 48px;
        font-size: 0.76rem;
        font-weight: 600;
        color: var(--text-primary);
        text-align: right;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ══════ Face Result Card ══════ */
    .face-result {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.98) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 16px 18px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .face-result:hover {
        border-color: var(--border-glow);
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.1);
    }
    .face-result::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 3px;
        height: 100%;
        border-radius: 3px 0 0 3px;
    }
    .face-result--male::before {
        background: linear-gradient(180deg, #6C63FF, #3B82F6);
    }
    .face-result--female::before {
        background: linear-gradient(180deg, #F472B6, #EC4899);
    }
    .face-head {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 12px;
    }
    .face-avatar {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
    }
    .face-avatar--male { background: linear-gradient(135deg, rgba(108,99,255,0.2), rgba(59,130,246,0.2)); }
    .face-avatar--female { background: linear-gradient(135deg, rgba(244,114,182,0.2), rgba(236,72,153,0.2)); }
    .face-name {
        font-weight: 700;
        font-size: 0.88rem;
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    .face-summary {
        font-size: 0.74rem;
        color: var(--text-muted);
        font-family: 'JetBrains Mono', monospace;
    }
    .section-label {
        font-size: 0.68rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 10px 0 4px;
    }

    /* ══════ Age Badge ══════ */
    .age-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--accent-cyan);
        background: rgba(8, 145, 178, 0.08);
        border: 1px solid rgba(8, 145, 178, 0.2);
        font-family: 'JetBrains Mono', monospace;
    }

    /* ══════ Metrics Row ══════ */
    .metrics-strip {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin: 14px 0;
    }
    .metric-tile {
        background: rgba(241, 245, 249, 0.9);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        padding: 12px 14px;
        text-align: center;
        transition: border-color 0.3s ease;
    }
    .metric-tile:hover {
        border-color: var(--border-glow);
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, var(--text-primary), var(--accent-secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.68rem;
        color: var(--text-muted);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 2px;
    }

    /* ══════ Empty State ══════ */
    .empty-state {
        text-align: center;
        padding: 60px 24px;
        color: var(--text-muted);
    }
    .empty-icon {
        font-size: 2.8rem;
        margin-bottom: 12px;
        opacity: 0.5;
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
    }
    .empty-text {
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .empty-hint {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-top: 8px;
        opacity: 0.7;
    }

    /* ══════ Status Pill ══════ */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .status--ok {
        background: rgba(52, 211, 153, 0.1);
        color: var(--accent-emerald);
        border: 1px solid rgba(52, 211, 153, 0.2);
    }
    .status--err {
        background: rgba(244, 114, 182, 0.1);
        color: var(--accent-rose);
        border: 1px solid rgba(244, 114, 182, 0.2);
    }

    /* ══════ Divider ══════ */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-subtle), transparent);
        margin: 16px 0;
    }

    /* ══════ Button Override ══════ */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent-primary), #6366F1) !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.2) !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.4) !important;
        transform: translateY(-1px);
    }

    /* ══════ Download Button ══════ */
    .stDownloadButton > button {
        background: rgba(241, 245, 249, 0.9) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-secondary) !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    .stDownloadButton > button:hover {
        border-color: var(--border-glow) !important;
        color: var(--text-primary) !important;
    }

    /* ══════ Pulse Dot ══════ */
    .pulse-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--accent-emerald);
        animation: pulse-glow 2s ease-in-out infinite;
    }
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 0 0 rgba(52,211,153,0.4); }
        50% { box-shadow: 0 0 0 6px rgba(52,211,153,0); }
    }
    </style>""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
# Hàm hỗ trợ hiển thị
# ────────────────────────────────────────────────────────────
GRADIENTS = {
    'male': ['#6C63FF', '#3B82F6'],
    'female': ['#F472B6', '#EC4899'],
    'race': ['#818CF8', '#FBBF24', '#F87171', '#A78BFA'],
}


def _prob_bars(probs, labels, colors):
    """Tạo HTML thanh xác suất kiểu neon."""
    html = '<div class="prob-group">'
    for i, (lbl, p) in enumerate(zip(labels, probs)):
        c = colors[i % len(colors)]
        v = p * 100
        html += (
            f'<div class="prob-row">'
            f'<span class="prob-label">{lbl}</span>'
            f'<div class="prob-track">'
            f'<div class="prob-fill" style="width:{v:.1f}%;background:linear-gradient(90deg,{c},{c}88)"></div>'
            f'</div>'
            f'<span class="prob-pct">{v:.1f}%</span>'
            f'</div>'
        )
    html += '</div>'
    return html


def _face_card_html(r, idx):
    """Tạo HTML cho thẻ kết quả một khuôn mặt — dark futuristic."""
    g = r["gender"]
    gender_key = "female" if g == "Female" else "male"
    cls = f"face-result face-result--{gender_key}"
    avatar_cls = f"face-avatar face-avatar--{gender_key}"
    icon = "👩" if g == "Female" else "👨"
    gender_vi = GENDER_VI.get(r.get("gender_idx", 0), g)
    race_vi = RACE_VI.get(r.get("race_idx", 0), r["race"])

    # Nhãn tiếng Việt cho thanh xác suất
    gender_labels_vi = [GENDER_VI.get(i, GENDER_LABELS[i]) for i in range(len(r["gender_prob"]))]
    race_labels_vi = [RACE_VI.get(i, RACE_LABELS[i]) for i in range(len(r["race_prob"]))]

    g_colors = GRADIENTS['male'] if gender_key == 'male' else GRADIENTS['female']
    g_html = _prob_bars(r["gender_prob"], gender_labels_vi, g_colors)
    r_html = _prob_bars(r["race_prob"], race_labels_vi, GRADIENTS['race'])

    return (
        f'<div class="{cls}">'
        f'<div class="face-head">'
        f'<div class="{avatar_cls}">{icon}</div>'
        f'<div>'
        f'<div class="face-name">Khuôn mặt {idx + 1}</div>'
        f'<div class="face-summary">{gender_vi} · {r["age"]} tuổi · {race_vi}</div>'
        f'</div>'
        f'</div>'
        f'<div class="section-label">Giới tính</div>{g_html}'
        f'<div class="section-label">Sắc tộc</div>{r_html}'
        f'<div class="section-label">Tuổi</div>'
        f'<span class="age-chip">🎂 {r["age"]} tuổi · {r.get("age_group", "")}</span>'
        f'</div>'
    )


def _to_csv(results):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["khuon_mat", "x", "y", "rong", "cao", "gioi_tinh", "do_tin_cay_gt",
                "tuoi", "sac_toc", "do_tin_cay_ct"])
    for i, r in enumerate(results):
        x, y, bw, bh = r["bbox"]
        gender_vi = GENDER_VI.get(r.get("gender_idx", 0), r["gender"])
        race_vi = RACE_VI.get(r.get("race_idx", 0), r["race"])
        w.writerow([
            i + 1, x, y, bw, bh,
            gender_vi, f"{max(r['gender_prob']):.3f}",
            r["age"], race_vi, f"{max(r['race_prob']):.3f}",
        ])
    return buf.getvalue()


def _empty(icon, msg, hint=""):
    hint_html = f'<div class="empty-hint">{hint}</div>' if hint else ''
    st.markdown(
        f'<div class="empty-state">'
        f'<div class="empty-icon">{icon}</div>'
        f'<div class="empty-text">{msg}</div>'
        f'{hint_html}</div>',
        unsafe_allow_html=True,
    )


@st.cache_resource
def _load_analyzer(path, mode):
    return FaceAnalyzer(model_path=path, age_mode=mode)


# ────────────────────────────────────────────────────────────
# Giao diện
# ────────────────────────────────────────────────────────────

def sidebar_settings():
    """Thanh bên: cài đặt mô hình, tiền xử lý, trạng thái."""
    with st.sidebar:
        st.markdown("### 🧠 FaceVision AI")
        st.caption("Hệ thống phân tích thuộc tính khuôn mặt")
        st.divider()

        # Chọn chế độ
        view_mode = st.radio(
            "Chế độ xem",
            ["🖼️ Ảnh tĩnh", "🎥 Camera trực tiếp", "📊 Biểu đồ đánh giá"],
            label_visibility="collapsed",
        )
        st.divider()

        # Mô hình
        st.markdown("**⚡ Mô hình**")
        model_name = st.selectbox(
            "Kiến trúc",
            ["ResNet50 + SE Attention"],
            help="Backbone ResNet50 với Squeeze-and-Excitation attention",
        )
        model_path = st.text_input("Tệp mô hình", value="checkpoints/best_model.pth")
        age_mode = "regression"

        st.divider()

        # Phát hiện khuôn mặt
        st.markdown("**🔍 Phát hiện khuôn mặt**")
        confidence = st.slider("Ngưỡng tin cậy", 1, 10, 5)

        st.divider()

        # Tiền xử lý
        st.markdown("**🎨 Tiền xử lý ảnh**")
        use_clahe = st.checkbox("Cân bằng histogram (CLAHE)", value=True)
        use_blur = st.checkbox("Làm mờ Gaussian", value=False)

        st.divider()

        # Trạng thái mô hình
        if os.path.exists(model_path):
            sz = os.path.getsize(model_path) / (1024 * 1024)
            st.markdown(
                f'<div class="status-pill status--ok">'
                f'<span class="pulse-dot"></span> Mô hình đã tải · {sz:.0f} MB</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-pill status--err">⚠ Không tìm thấy mô hình</div>',
                unsafe_allow_html=True,
            )

        device = 'CUDA (GPU)' if __import__('torch').cuda.is_available() else 'CPU'
        st.caption(f"⚙ Thiết bị: {device}")

    return view_mode, model_path, age_mode, confidence, use_blur


def header():
    """Hero header với gradient text và badges."""
    st.markdown("""
    <div class="hero">
        <div class="hero-title">🧠 FaceVision AI</div>
        <div class="hero-sub">
            Hệ thống dự đoán thuộc tính khuôn mặt bằng Deep Learning
        </div>
        <div class="hero-badges">
            <span class="hero-badge">🏗 ResNet50 + SE</span>
            <span class="hero-badge">📊 Multi-Task CNN</span>
            <span class="hero-badge">🎯 TTA Inference</span>
            <span class="hero-badge">📸 23K UTKFace</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def input_panel():
    """Panel đầu vào: chế độ Ảnh (tải lên / webcam / clipboard).
    Trả về PIL Image hoặc None."""

    st.markdown(
        '<div class="panel-header">'
        '<div class="panel-icon panel-icon--input">📥</div>'
        '<span class="panel-label">Đầu vào</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Chọn nguồn — 3 tùy chọn trên 1 hàng
    src = st.radio(
        "Nguồn",
        ["📁 Tải ảnh lên", "📷 Chụp webcam", "📋 Dán clipboard"],
        horizontal=True,
        label_visibility="collapsed",
    )

    img = None

    if src == "📁 Tải ảnh lên":
        f = st.file_uploader(
            "Chọn ảnh",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )
        if f:
            img = Image.open(f).convert("RGB")

    elif src == "📷 Chụp webcam":
        cam = st.camera_input("Chụp ảnh", label_visibility="collapsed")
        if cam:
            img = Image.open(cam).convert("RGB")

    else:  # Clipboard — dán ảnh từ bộ nhớ tạm (Ctrl+C)
        st.info("💡 **Cách dùng:** Copy ảnh từ internet (chuột phải → Sao chép ảnh), rồi bấm nút bên dưới.")
        if st.button("📋 Dán ảnh từ bộ nhớ tạm", type="primary", use_container_width=True):
            try:
                from PIL import ImageGrab
                clipboard_img = ImageGrab.grabclipboard()
                if clipboard_img is not None and isinstance(clipboard_img, Image.Image):
                    img = clipboard_img.convert("RGB")
                    st.session_state["input_img"] = img
                    st.success("✅ Đã dán ảnh từ clipboard!")
                elif isinstance(clipboard_img, list) and len(clipboard_img) > 0:
                    # Clipboard chứa đường dẫn file ảnh
                    img = Image.open(clipboard_img[0]).convert("RGB")
                    st.session_state["input_img"] = img
                    st.success("✅ Đã dán ảnh từ clipboard!")
                else:
                    st.warning("⚠️ Không tìm thấy ảnh trong bộ nhớ tạm. Hãy copy ảnh trước (chuột phải → Sao chép ảnh).")
            except Exception as e:
                st.error(f"❌ Không thể đọc clipboard: {e}")

    # Giữ ảnh qua các lần rerun
    if img is not None:
        st.session_state["input_img"] = img
    elif "input_img" in st.session_state:
        img = st.session_state["input_img"]

    # Xem trước
    if img:
        st.image(img, caption="Ảnh xem trước", use_container_width=True)
    else:
        _empty(
            "📸",
            "Tải ảnh lên hoặc chụp ảnh để bắt đầu",
            "Hỗ trợ JPG, PNG, BMP, WebP · Tối đa 10 MB"
        )

    return img


def output_panel(analyzer, img, confidence, blur):
    """Panel kết quả: chạy suy luận và hiển thị kết quả."""

    st.markdown(
        '<div class="panel-header">'
        '<div class="panel-icon panel-icon--output">📊</div>'
        '<span class="panel-label">Kết quả phân tích</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    if img is None:
        _empty(
            "🔬",
            "Đang chờ ảnh đầu vào…",
            "Kết quả sẽ hiển thị sau khi phân tích"
        )
        return

    # Nút phân tích
    if st.button("⚡ Phân tích khuôn mặt", type="primary", use_container_width=True):
        with st.spinner("🧠 Đang phát hiện khuôn mặt và dự đoán thuộc tính…"):
            arr = np.array(img)
            results, res_img = analyzer.analyze_image(
                arr, min_neighbors=confidence, apply_blur=blur, use_tta=True
            )
            st.session_state["results"] = results
            st.session_state["res_img"] = res_img
            st.rerun()

    results = st.session_state.get("results", [])
    res_img = st.session_state.get("res_img", None)

    if not results or res_img is None:
        _empty(
            "⚡",
            'Bấm "Phân tích khuôn mặt" để bắt đầu',
            "Sử dụng Test-Time Augmentation cho kết quả chính xác hơn"
        )
        return

    # ── Ảnh đã chú thích ──
    annotated = analyzer.draw_results(res_img, results)
    st.image(annotated, caption=f"Phát hiện {len(results)} khuôn mặt", use_container_width=True)

    # ── Thống kê nhanh ──
    n = len(results)
    males = sum(1 for r in results if r["gender"] == "Male")
    females = n - males
    avg_age = sum(r["age"] for r in results) / n if n else 0

    st.markdown(f"""
    <div class="metrics-strip">
        <div class="metric-tile">
            <div class="metric-value">{n}</div>
            <div class="metric-label">Khuôn mặt</div>
        </div>
        <div class="metric-tile">
            <div class="metric-value">{males} / {females}</div>
            <div class="metric-label">Nam / Nữ</div>
        </div>
        <div class="metric-tile">
            <div class="metric-value">{avg_age:.0f}</div>
            <div class="metric-label">Tuổi trung bình</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Thẻ từng khuôn mặt ──
    cards_html = ""
    for i, r in enumerate(results):
        cards_html += _face_card_html(r, i)
    st.markdown(cards_html, unsafe_allow_html=True)

    # ── Xuất dữ liệu ──
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.download_button(
        "📥 Xuất kết quả CSV",
        _to_csv(results),
        file_name="ket_qua_khuon_mat.csv",
        mime="text/csv",
        use_container_width=True,
    )


def live_camera_view(col_in, col_out, analyzer, confidence, blur):
    """Camera trực tiếp: tối ưu FPS tối đa với Haar + batch predict + JPEG compression."""

    # ── Cấu hình tối ưu FPS ──
    DETECT_INTERVAL = 5       # Detect mỗi N frame (frames giữa reuse bbox)
    PROCESS_WIDTH = 320       # Resize nhỏ cho processing (detection + CNN)
    JPEG_QUALITY = 70         # Chất lượng JPEG cho st.image (giảm bandwidth)

    with col_in:
        st.markdown(
            '<div class="panel-header">'
            '<div class="panel-icon panel-icon--input">🎥</div>'
            '<span class="panel-label">Camera trực tiếp</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        run = st.toggle("Bật camera", value=False)
        cam_placeholder = st.empty()
        fps_placeholder = st.empty()

    with col_out:
        st.markdown(
            '<div class="panel-header">'
            '<div class="panel-icon panel-icon--output">📊</div>'
            '<span class="panel-label">Kết quả trực tiếp</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        result_placeholder = st.empty()

    if not run:
        with col_in:
            _empty("🎥", "Bật công tắc để khởi động camera")
        with col_out:
            _empty("📊", "Kết quả sẽ hiện khi camera hoạt động")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        with col_in:
            st.error("❌ Không thể mở webcam")
        return

    # Giảm resolution camera + buffer tối thiểu
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_idx = 0
    fps_count = 0
    t_fps = time.time()
    cached_bboxes = None
    cached_results = []

    while run:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = rgb.shape[:2]

        # ── Resize nhỏ cho processing ──
        if w_orig > PROCESS_WIDTH:
            scale = PROCESS_WIDTH / w_orig
            small = cv2.resize(rgb, (PROCESS_WIDTH, int(h_orig * scale)),
                               interpolation=cv2.INTER_LINEAR)
        else:
            scale = 1.0
            small = rgb

        # ── Detection + Prediction ──
        if frame_idx % DETECT_INTERVAL == 0:
            # Frame detection: Haar + batch predict
            res, new_bboxes = analyzer.fast_analyze(
                small, bboxes=None, min_neighbors=confidence
            )
            if res:
                cached_results = res
                cached_bboxes = new_bboxes
        else:
            # Frames trung gian: reuse bbox, chỉ chạy CNN
            if cached_bboxes:
                res, _ = analyzer.fast_analyze(
                    small, bboxes=cached_bboxes, min_neighbors=confidence
                )
                if res:
                    cached_results = res

        # ── Scale bboxes về resolution gốc ──
        display_results = []
        for r in cached_results:
            r_scaled = r.copy()
            if scale != 1.0:
                bx, by, bw, bh = r_scaled['bbox']
                r_scaled['bbox'] = [
                    int(bx / scale), int(by / scale),
                    int(bw / scale), int(bh / scale)
                ]
            display_results.append(r_scaled)

        # ── Vẽ annotation lên ảnh gốc ──
        annotated = analyzer.draw_results(rgb, display_results)

        # ── Encode JPEG trước khi gửi (giảm bandwidth ~5x) ──
        _, jpg_buf = cv2.imencode('.jpg', cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        cam_placeholder.image(jpg_buf.tobytes(), use_container_width=True)

        # ── FPS counter ──
        fps_count += 1
        frame_idx += 1
        elapsed = time.time() - t_fps
        if elapsed > 0.5:
            fps_placeholder.caption(f"⚡ {fps_count / elapsed:.1f} FPS")
            fps_count = 0
            t_fps = time.time()

        # ── Update result cards (mỗi detect frame) ──
        if frame_idx % DETECT_INTERVAL == 1:
            if cached_results:
                html = ""
                for i, r in enumerate(cached_results):
                    html += _face_card_html(r, i)
                result_placeholder.markdown(html, unsafe_allow_html=True)
            else:
                result_placeholder.info("Không phát hiện khuôn mặt nào")

    cap.release()



# ────────────────────────────────────────────────────────────
# Trang Biểu đồ Đánh giá Mô hình
# ────────────────────────────────────────────────────────────

# Plotly layout mặc định cho light theme
_PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(255,255,255,0)',
    plot_bgcolor='rgba(248,250,252,0.8)',
    font=dict(family='Inter, sans-serif', color='#1E293B'),
    margin=dict(l=50, r=30, t=50, b=50),
)

_COLORS = {
    'primary': '#4F46E5',
    'cyan': '#0891B2',
    'emerald': '#059669',
    'rose': '#DB2777',
    'amber': '#D97706',
    'secondary': '#6366F1',
    'blue': '#2563EB',
    'orange': '#EA580C',
    'palette': ['#4F46E5', '#0891B2', '#059669', '#DB2777', '#D97706', '#6366F1', '#2563EB', '#EA580C'],
}


def _load_metrics():
    """Tải metrics từ logs/metrics.json."""
    paths = ['logs/metrics.json', 'logs/metrics_v3.json', 'logs/metrics_v4.json']
    data = {}
    for p in paths:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                data[p] = json.load(f)
    return data


def _chart_training_curves(history, title_suffix=''):
    """Biểu đồ Loss và Accuracy qua các epoch."""
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train']['loss'] for h in history]
    val_loss = [h['val']['loss'] for h in history]
    train_gender = [h['train']['gender_acc'] for h in history]
    val_gender = [h['val']['gender_acc'] for h in history]
    train_race = [h['train']['race_acc'] for h in history]
    val_race = [h['val']['race_acc'] for h in history]
    train_age = [h['train']['age_mae'] for h in history]
    val_age = [h['val']['age_mae'] for h in history]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'📉 Loss {title_suffix}',
            f'🎂 Age MAE {title_suffix}',
            f'👤 Gender Accuracy {title_suffix}',
            f'🌍 Race Accuracy {title_suffix}',
        ),
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )

    # Loss
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, name='Train Loss',
                             line=dict(color=_COLORS['primary'], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, name='Val Loss',
                             line=dict(color=_COLORS['cyan'], width=2, dash='dash')), row=1, col=1)

    # Age MAE
    fig.add_trace(go.Scatter(x=epochs, y=train_age, name='Train MAE',
                             line=dict(color=_COLORS['amber'], width=2), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=val_age, name='Val MAE',
                             line=dict(color=_COLORS['emerald'], width=2, dash='dash'), showlegend=False), row=1, col=2)

    # Gender Acc
    fig.add_trace(go.Scatter(x=epochs, y=train_gender, name='Train Gender',
                             line=dict(color=_COLORS['rose'], width=2), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=val_gender, name='Val Gender',
                             line=dict(color=_COLORS['secondary'], width=2, dash='dash'), showlegend=False), row=2, col=1)

    # Race Acc
    fig.add_trace(go.Scatter(x=epochs, y=train_race, name='Train Race',
                             line=dict(color=_COLORS['orange'], width=2), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=val_race, name='Val Race',
                             line=dict(color=_COLORS['blue'], width=2, dash='dash'), showlegend=False), row=2, col=2)

    fig.update_layout(**_PLOTLY_LAYOUT, height=600, showlegend=True,
                      legend=dict(orientation='h', yanchor='bottom', y=1.08, xanchor='center', x=0.5,
                                  bgcolor='rgba(255,255,255,0.9)', bordercolor='#E2E8F0'))
    fig.update_xaxes(title_text='Epoch', gridcolor='rgba(148,163,184,0.2)')
    fig.update_yaxes(gridcolor='rgba(148,163,184,0.2)')
    return fig


def _chart_confusion_matrix(cm, labels, title):
    """Heatmap confusion matrix."""
    # Tỉ lệ %
    cm_np = np.array(cm)
    cm_pct = cm_np / cm_np.sum(axis=1, keepdims=True) * 100

    text = [[f'{cm_np[i][j]}<br>({cm_pct[i][j]:.1f}%)' for j in range(len(labels))] for i in range(len(labels))]

    fig = go.Figure(data=go.Heatmap(
        z=cm_pct,
        x=[f'Pred: {l}' for l in labels],
        y=[f'True: {l}' for l in labels],
        textfont=dict(size=13, color='white'),
        colorscale=[[0, '#EEF2FF'], [0.5, '#6366F1'], [1, '#4F46E5']],
        showscale=True, colorbar=dict(title='%'),
        hovertemplate='True: %{y}<br>Pred: %{x}<br>Count: %{text}<extra></extra>',
    ))
    fig.update_layout(**_PLOTLY_LAYOUT, title=dict(text=title, font=dict(size=16)),
                      height=420, xaxis_title='Dự đoán', yaxis_title='Thực tế',
                      yaxis=dict(autorange='reversed'))
    return fig


def _chart_classification_report(report, labels, title):
    """Bar chart Precision / Recall / F1-Score."""
    prec = [report[l]['precision'] * 100 for l in labels]
    rec = [report[l]['recall'] * 100 for l in labels]
    f1 = [report[l]['f1-score'] * 100 for l in labels]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Precision', x=labels, y=prec,
                         marker_color=_COLORS['primary'], text=[f'{v:.1f}%' for v in prec], textposition='outside'))
    fig.add_trace(go.Bar(name='Recall', x=labels, y=rec,
                         marker_color=_COLORS['cyan'], text=[f'{v:.1f}%' for v in rec], textposition='outside'))
    fig.add_trace(go.Bar(name='F1-Score', x=labels, y=f1,
                         marker_color=_COLORS['emerald'], text=[f'{v:.1f}%' for v in f1], textposition='outside'))

    fig.update_layout(**_PLOTLY_LAYOUT, title=dict(text=title, font=dict(size=16)),
                      barmode='group', height=400, yaxis_title='%',
                      yaxis=dict(range=[60, 105], gridcolor='rgba(148,163,184,0.2)'))
    return fig


def _chart_age_error_by_group():
    """Bar chart sai số tuổi theo nhóm (hardcoded từ eval results)."""
    groups = ['0-12\nChild', '13-19\nTeen', '20-35\nYoung Adult', '36-55\nAdult', '56+\nSenior']
    mae = [1.67, 3.64, 3.77, 6.68, 8.44]
    median = [0.79, 2.77, 2.86, 5.56, 6.92]
    counts = [507, 186, 1558, 785, 520]

    fig = make_subplots(specs=[[{'secondary_y': True}]])

    fig.add_trace(go.Bar(name='MAE (năm)', x=groups, y=mae,
                         marker=dict(color=mae, colorscale=[[0, '#059669'], [0.5, '#D97706'], [1, '#DB2777']]),
                         text=[f'{v:.2f}' for v in mae], textposition='outside'), secondary_y=False)
    fig.add_trace(go.Bar(name='Median Error', x=groups, y=median,
                         marker_color='rgba(79,70,229,0.3)',
                         text=[f'{v:.2f}' for v in median], textposition='outside'), secondary_y=False)
    fig.add_trace(go.Scatter(name='Số mẫu', x=groups, y=counts, mode='lines+markers+text',
                             text=[str(c) for c in counts], textposition='top center',
                             line=dict(color=_COLORS['cyan'], width=2),
                             marker=dict(size=8, color=_COLORS['cyan'])), secondary_y=True)

    fig.update_layout(**_PLOTLY_LAYOUT, title=dict(text='🎂 Sai số Tuổi theo Nhóm tuổi', font=dict(size=16)),
                      barmode='group', height=420)
    fig.update_yaxes(title_text='MAE (năm)', gridcolor='rgba(148,163,184,0.2)', secondary_y=False)
    fig.update_yaxes(title_text='Số mẫu test', gridcolor='rgba(148,163,184,0.1)', secondary_y=True)
    return fig


def _chart_radar(test_metrics):
    """Radar chart tổng quan hiệu năng."""
    age_score = max(0, 100 - test_metrics['age_mae'] * 5)  # MAE 0=100%, MAE 20=0%
    gender_score = test_metrics['gender_accuracy']
    race_score = test_metrics['race_accuracy']

    # Lấy thêm chi tiết
    gr = test_metrics.get('gender_classification_report', {})
    rr = test_metrics.get('race_classification_report', {})
    gender_f1 = gr.get('macro avg', {}).get('f1-score', 0) * 100
    race_f1 = rr.get('macro avg', {}).get('f1-score', 0) * 100

    categories = ['Age Score', 'Gender Acc', 'Race Acc', 'Gender F1', 'Race F1']
    values = [age_score, gender_score, race_score, gender_f1, race_f1]
    values.append(values[0])  # close the radar
    categories.append(categories[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories,
        fill='toself', fillcolor='rgba(79,70,229,0.1)',
        line=dict(color=_COLORS['primary'], width=2),
        marker=dict(size=6, color=_COLORS['cyan']),
        text=[f'{v:.1f}%' for v in values],
        hovertemplate='%{theta}: %{r:.1f}%<extra></extra>',
    ))
    # Target line
    target = [80, 94, 88, 93, 85]
    target.append(target[0])
    fig.add_trace(go.Scatterpolar(
        r=target, theta=categories,
        line=dict(color=_COLORS['amber'], width=1.5, dash='dash'),
        name='Mục tiêu', opacity=0.7,
    ))

    fig.update_layout(
        **_PLOTLY_LAYOUT, height=420,
        title=dict(text='🎯 Radar Hiệu năng Tổng quan', font=dict(size=16)),
        polar=dict(
            bgcolor='rgba(248,250,252,0.6)',
            radialaxis=dict(visible=True, range=[50, 100], gridcolor='rgba(148,163,184,0.2)'),
            angularaxis=dict(gridcolor='rgba(148,163,184,0.2)'),
        ),
        showlegend=True,
    )
    return fig


def _chart_dataset_distribution():
    """Pie charts phân bố dataset."""
    fig = make_subplots(rows=1, cols=3,
                        specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]],
                        subplot_titles=('Giới tính', 'Sắc tộc', 'Nhóm tuổi'))

    # Gender
    fig.add_trace(go.Pie(labels=['Nam', 'Nữ'], values=[1859, 1697],
                         marker=dict(colors=[_COLORS['primary'], _COLORS['rose']]),
                         textinfo='label+percent', hole=0.45), row=1, col=1)
    # Race
    fig.add_trace(go.Pie(labels=['White', 'Black', 'Asian', 'Others'],
                         values=[1494, 686, 505, 871],
                         marker=dict(colors=[_COLORS['secondary'], _COLORS['amber'],
                                             _COLORS['cyan'], _COLORS['orange']]),
                         textinfo='label+percent', hole=0.45), row=1, col=2)
    # Age groups
    fig.add_trace(go.Pie(labels=['0-12', '13-19', '20-35', '36-55', '56+'],
                         values=[507, 186, 1558, 785, 520],
                         marker=dict(colors=_COLORS['palette'][:5]),
                         textinfo='label+percent', hole=0.45), row=1, col=3)

    fig.update_layout(**_PLOTLY_LAYOUT, height=380,
                      title=dict(text='📦 Phân bố Tập Test (3,556 mẫu)', font=dict(size=16)),
                      showlegend=False)
    return fig


def _chart_accuracy_comparison():
    """So sánh accuracy sai lệch ≤5 năm, ≤10 năm, gender, race."""
    metrics = ['Sai ≤5 năm', 'Sai ≤10 năm', 'Gender Acc', 'Race Acc']
    values = [65.7, 87.4, 93.53, 85.69]
    colors = [_COLORS['amber'], _COLORS['emerald'], _COLORS['rose'], _COLORS['cyan']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metrics, y=values,
        marker_color=colors,
        text=[f'{v:.1f}%' for v in values],
        textposition='outside',
        textfont=dict(size=14, color='#1E293B'),
        width=0.5,
    ))
    # Target lines
    fig.add_hline(y=90, line_dash='dash', line_color=_COLORS['amber'],
                  annotation_text='Target 90%', annotation_position='top right')

    fig.update_layout(**_PLOTLY_LAYOUT, height=400,
                      title=dict(text='📊 Tổng quan Độ chính xác', font=dict(size=16)),
                      yaxis=dict(range=[0, 105], title='%', gridcolor='rgba(148,163,184,0.2)'),
                      showlegend=False)
    return fig


def chart_panel():
    """Trang biểu đồ đánh giá mô hình đầy đủ."""
    st.markdown(
        '<div class="panel-header">'
        '<div class="panel-icon panel-icon--output">📊</div>'
        '<span class="panel-label">Đánh Giá Mô Hình — Biểu Đồ Chi Tiết</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Load metrics
    all_metrics = _load_metrics()
    if not all_metrics:
        st.warning('⚠️ Không tìm thấy file metrics trong thư mục logs/')
        return

    # Chọn version nếu có nhiều
    versions = list(all_metrics.keys())
    version_labels = {
        'logs/metrics.json': 'v2 (ResNet50 + SE)',
        'logs/metrics_v3.json': 'v3 (Nâng cao)',
        'logs/metrics_v4.json': 'v4 (Fine-tune)',
    }

    if len(versions) > 1:
        selected = st.selectbox(
            '📂 Chọn phiên bản mô hình',
            versions,
            format_func=lambda x: version_labels.get(x, x),
        )
    else:
        selected = versions[0]

    data = all_metrics[selected]
    test_m = data.get('test_metrics', {})
    history = data.get('history', data.get('epoch_history', []))

    # ── Hero Stats ──
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric('🎂 Age MAE', f"{test_m.get('age_mae', 0):.2f} năm")
    with c2:
        st.metric('👤 Gender Acc', f"{test_m.get('gender_accuracy', 0):.1f}%")
    with c3:
        st.metric('🌍 Race Acc', f"{test_m.get('race_accuracy', 0):.1f}%")
    with c4:
        score = (100 - test_m.get('age_mae', 5) * 2) * 0.3 + \
                test_m.get('gender_accuracy', 90) * 0.35 + \
                test_m.get('race_accuracy', 85) * 0.35
        grade = 'A' if score >= 90 else 'B' if score >= 80 else 'C'
        st.metric('🏆 Tổng điểm', f"{score:.1f}/100 ({grade})")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Tab Charts ──
    tabs = st.tabs([
        '📈 Quá trình Training',
        '🎯 Tổng quan Accuracy',
        '🔥 Confusion Matrix',
        '📊 Precision/Recall/F1',
        '🎂 Sai số theo Tuổi',
        '📦 Phân bố Dữ liệu',
    ])

    # Tab 1: Training curves
    with tabs[0]:
        if history:
            st.plotly_chart(_chart_training_curves(history, version_labels.get(selected, '')),
                           use_container_width=True)
            st.caption(f'Tổng cộng {len(history)} epochs · Best epoch: {data.get("best_epoch", "?")} · '
                       f'Best val loss: {data.get("best_val_loss", 0):.4f}')
        else:
            st.info('Không có dữ liệu training history.')

    # Tab 2: Overall Accuracy + Radar
    with tabs[1]:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(_chart_accuracy_comparison(), use_container_width=True)
        with col_b:
            if test_m:
                st.plotly_chart(_chart_radar(test_m), use_container_width=True)

    # Tab 3: Confusion Matrices
    with tabs[2]:
        col_g, col_r = st.columns(2)
        with col_g:
            gcm = test_m.get('gender_confusion_matrix')
            if gcm:
                st.plotly_chart(
                    _chart_confusion_matrix(gcm, ['Male', 'Female'], '👤 Gender Confusion Matrix'),
                    use_container_width=True)
        with col_r:
            rcm = test_m.get('race_confusion_matrix')
            if rcm:
                st.plotly_chart(
                    _chart_confusion_matrix(rcm, ['White', 'Black', 'Asian', 'Others'], '🌍 Race Confusion Matrix'),
                    use_container_width=True)

    # Tab 4: Classification Report Bars
    with tabs[3]:
        col_g2, col_r2 = st.columns(2)
        with col_g2:
            gr = test_m.get('gender_classification_report', {})
            if gr:
                st.plotly_chart(
                    _chart_classification_report(gr, ['Male', 'Female'], '👤 Gender — Precision/Recall/F1'),
                    use_container_width=True)
        with col_r2:
            rr = test_m.get('race_classification_report', {})
            if rr:
                st.plotly_chart(
                    _chart_classification_report(rr, ['White', 'Black', 'Asian', 'Others'],
                                                '🌍 Race — Precision/Recall/F1'),
                    use_container_width=True)

    # Tab 5: Age Error
    with tabs[4]:
        st.plotly_chart(_chart_age_error_by_group(), use_container_width=True)
        st.markdown("""
        <div class="face-result">
            <div class="section-label">📝 Nhận xét</div>
            <ul style="color: var(--text-secondary); font-size: 0.82rem; line-height: 1.7;">
                <li><strong>Nhóm 0-12 (Child):</strong> MAE = 1.67 — Xuất sắc, trẻ em có đặc trưng khuôn mặt rõ ràng</li>
                <li><strong>Nhóm 13-35:</strong> MAE ≈ 3.7 — Tốt, lứa tuổi phổ biến nhất trong dataset</li>
                <li><strong>Nhóm 36-55 (Adult):</strong> MAE = 6.68 — Trung bình, sự lão hóa đa dạng giữa các sắc tộc</li>
                <li><strong>Nhóm 56+ (Senior):</strong> MAE = 8.44 — Yếu nhất, ít mẫu + biến thiên lớn</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Tab 6: Dataset Distribution
    with tabs[5]:
        st.plotly_chart(_chart_dataset_distribution(), use_container_width=True)
        st.markdown("""
        <div class="face-result">
            <div class="section-label">📝 Phân tích Mất cân bằng</div>
            <ul style="color: var(--text-secondary); font-size: 0.82rem; line-height: 1.7;">
                <li><strong>Giới tính:</strong> Tương đối cân bằng (52.3% Nam / 47.7% Nữ)</li>
                <li><strong>Sắc tộc:</strong> White chiếm 42% — Others là nhóm khó nhất (24.5%)</li>
                <li><strong>Nhóm tuổi:</strong> Young Adult 20-35 chiếm đa số (43.8%) — Teen ít nhất (5.2%)</li>
                <li>→ Sử dụng <strong>WeightedRandomSampler</strong> để cân bằng Race + Age group khi training</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
# Chạy chính
# ────────────────────────────────────────────────────────────
def main():
    inject_css()
    view_mode, model_path, age_mode, confidence, blur = sidebar_settings()
    analyzer = _load_analyzer(model_path, age_mode)
    header()

    if view_mode == "📊 Biểu đồ đánh giá":
        chart_panel()
    elif view_mode == "🖼️ Ảnh tĩnh":
        # Bố cục hai cột
        col_left, col_right = st.columns(2, gap="medium")
        with col_left:
            img = input_panel()
        with col_right:
            output_panel(analyzer, img, confidence, blur)
    else:
        col_left, col_right = st.columns(2, gap="medium")
        live_camera_view(col_left, col_right, analyzer, confidence, blur)


if __name__ == "__main__":
    main()
