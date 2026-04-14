"""
export_charts.py — Xuất TẤT CẢ biểu đồ đánh giá mô hình ra folder chart/
Tạo ảnh PNG chất lượng cao (300 DPI) cho báo cáo/thesis
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
CHART_DIR = os.path.join(PROJECT_ROOT, 'chart')
os.makedirs(CHART_DIR, exist_ok=True)

# Light theme colors
C = {
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

# Plotly light layout
LAYOUT = dict(
    paper_bgcolor='#FFFFFF',
    plot_bgcolor='#F8FAFC',
    font=dict(family='Inter, Segoe UI, sans-serif', color='#0F172A', size=14),
    margin=dict(l=60, r=40, t=70, b=60),
)

WIDTH = 1200
HEIGHT = 700
SCALE = 2  # 2x resolution


def save_fig(fig, name):
    """Lưu figure ra PNG."""
    path = os.path.join(CHART_DIR, f'{name}.png')
    fig.write_image(path, width=WIDTH, height=HEIGHT, scale=SCALE)
    print(f'  ✅ {path}')


# ────────────────────────────────────────────────────────────
# Load metrics
# ────────────────────────────────────────────────────────────
def load_metrics():
    paths = [
        ('v2', os.path.join(PROJECT_ROOT, 'logs', 'metrics.json')),
        ('v3', os.path.join(PROJECT_ROOT, 'logs', 'metrics_v3.json')),
        ('v4', os.path.join(PROJECT_ROOT, 'logs', 'metrics_v4.json')),
    ]
    data = {}
    for label, p in paths:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                data[label] = json.load(f)
            print(f'  📂 Loaded {label}: {p}')
    return data


# ────────────────────────────────────────────────────────────
# 1. Training Curves (Loss + Age MAE + Gender Acc + Race Acc)
# ────────────────────────────────────────────────────────────
def chart_training_curves(history, version=''):
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
            '📉 Loss', '🎂 Age MAE (năm)',
            '👤 Gender Accuracy (%)', '🌍 Race Accuracy (%)',
        ),
        vertical_spacing=0.14, horizontal_spacing=0.10,
    )

    # Loss
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, name='Train Loss',
                             line=dict(color=C['primary'], width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, name='Val Loss',
                             line=dict(color=C['cyan'], width=2.5, dash='dash')), row=1, col=1)

    # Age MAE
    fig.add_trace(go.Scatter(x=epochs, y=train_age, name='Train MAE',
                             line=dict(color=C['amber'], width=2.5), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=val_age, name='Val MAE',
                             line=dict(color=C['emerald'], width=2.5, dash='dash'), showlegend=False), row=1, col=2)

    # Gender Acc
    fig.add_trace(go.Scatter(x=epochs, y=train_gender, name='Train Gender',
                             line=dict(color=C['rose'], width=2.5), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=val_gender, name='Val Gender',
                             line=dict(color=C['secondary'], width=2.5, dash='dash'), showlegend=False), row=2, col=1)

    # Race Acc
    fig.add_trace(go.Scatter(x=epochs, y=train_race, name='Train Race',
                             line=dict(color=C['orange'], width=2.5), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=val_race, name='Val Race',
                             line=dict(color=C['blue'], width=2.5, dash='dash'), showlegend=False), row=2, col=2)

    fig.update_layout(
        **LAYOUT, height=HEIGHT,
        title=dict(text=f'📈 Quá trình Training {version}', font=dict(size=20)),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.06, xanchor='center', x=0.5,
                    bgcolor='rgba(255,255,255,0.9)', bordercolor='#E2E8F0', borderwidth=1),
    )
    fig.update_xaxes(title_text='Epoch', gridcolor='rgba(148,163,184,0.2)')
    fig.update_yaxes(gridcolor='rgba(148,163,184,0.2)')
    return fig


# ────────────────────────────────────────────────────────────
# 2. Confusion Matrix (Gender / Race)
# ────────────────────────────────────────────────────────────
def chart_confusion_matrix(cm, labels, title):
    cm_np = np.array(cm)
    cm_pct = cm_np / cm_np.sum(axis=1, keepdims=True) * 100

    text = [[f'<b>{cm_np[i][j]}</b><br>({cm_pct[i][j]:.1f}%)'
             for j in range(len(labels))] for i in range(len(labels))]

    fig = go.Figure(data=go.Heatmap(
        z=cm_pct,
        x=[f'Pred: {l}' for l in labels],
        y=[f'True: {l}' for l in labels],
        text=text, texttemplate='%{text}',
        textfont=dict(size=16, color='#0F172A'),
        colorscale=[[0, '#EEF2FF'], [0.35, '#A5B4FC'], [0.65, '#6366F1'], [1, '#3730A3']],
        showscale=True, colorbar=dict(title='%'),
    ))
    fig.update_layout(
        **LAYOUT, height=550,
        title=dict(text=title, font=dict(size=18)),
        xaxis_title='Dự đoán (Predicted)',
        yaxis_title='Thực tế (Actual)',
        yaxis=dict(autorange='reversed'),
    )
    return fig


# ────────────────────────────────────────────────────────────
# 3. Classification Report (Precision / Recall / F1)
# ────────────────────────────────────────────────────────────
def chart_classification_report(report, labels, title):
    prec = [report[l]['precision'] * 100 for l in labels]
    rec = [report[l]['recall'] * 100 for l in labels]
    f1 = [report[l]['f1-score'] * 100 for l in labels]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Precision', x=labels, y=prec,
                         marker_color=C['primary'],
                         text=[f'<b>{v:.1f}</b>' for v in prec], textposition='outside',
                         textfont=dict(size=14, color='#0F172A')))
    fig.add_trace(go.Bar(name='Recall', x=labels, y=rec,
                         marker_color=C['cyan'],
                         text=[f'<b>{v:.1f}</b>' for v in rec], textposition='outside',
                         textfont=dict(size=14, color='#0F172A')))
    fig.add_trace(go.Bar(name='F1-Score', x=labels, y=f1,
                         marker_color=C['emerald'],
                         text=[f'<b>{v:.1f}</b>' for v in f1], textposition='outside',
                         textfont=dict(size=14, color='#0F172A')))

    fig.update_layout(
        **LAYOUT, height=500,
        title=dict(text=title, font=dict(size=18)),
        barmode='group', yaxis_title='%',
        yaxis=dict(range=[55, 105], gridcolor='rgba(148,163,184,0.2)'),
    )
    return fig


# ────────────────────────────────────────────────────────────
# 4. Age Error by Group
# ────────────────────────────────────────────────────────────
def chart_age_error_by_group():
    groups = ['0-12\nChild', '13-19\nTeen', '20-35\nYoung Adult', '36-55\nAdult', '56+\nSenior']
    mae = [1.67, 3.64, 3.77, 6.68, 8.44]
    median = [0.79, 2.77, 2.86, 5.56, 6.92]
    counts = [507, 186, 1558, 785, 520]

    fig = make_subplots(specs=[[{'secondary_y': True}]])

    fig.add_trace(go.Bar(
        name='MAE (năm)', x=groups, y=mae,
        marker=dict(color=mae, colorscale=[[0, '#059669'], [0.5, '#D97706'], [1, '#DB2777']]),
        text=[f'<b>{v:.2f}</b>' for v in mae], textposition='outside',
        textfont=dict(size=14, color='#0F172A'),
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        name='Median Error', x=groups, y=median,
        marker_color='rgba(79,70,229,0.3)',
        text=[f'{v:.2f}' for v in median], textposition='outside',
        textfont=dict(size=13, color='#334155'),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        name='Số mẫu test', x=groups, y=counts, mode='lines+markers+text',
        text=[str(c) for c in counts], textposition='top center',
        line=dict(color=C['cyan'], width=2.5),
        marker=dict(size=10, color=C['cyan']),
        textfont=dict(size=13, color='#0F172A'),
    ), secondary_y=True)

    fig.update_layout(
        **LAYOUT, height=550,
        title=dict(text='🎂 Sai số Tuổi theo Nhóm tuổi', font=dict(size=18)),
        barmode='group',
    )
    fig.update_yaxes(title_text='MAE (năm)', gridcolor='rgba(148,163,184,0.2)', secondary_y=False)
    fig.update_yaxes(title_text='Số mẫu test', gridcolor='rgba(148,163,184,0.1)', secondary_y=True)
    return fig


# ────────────────────────────────────────────────────────────
# 5. Radar Chart (Overall Performance)
# ────────────────────────────────────────────────────────────
def chart_radar(test_metrics):
    age_score = max(0, 100 - test_metrics['age_mae'] * 5)
    gender_score = test_metrics['gender_accuracy']
    race_score = test_metrics['race_accuracy']

    gr = test_metrics.get('gender_classification_report', {})
    rr = test_metrics.get('race_classification_report', {})
    gender_f1 = gr.get('macro avg', {}).get('f1-score', 0) * 100
    race_f1 = rr.get('macro avg', {}).get('f1-score', 0) * 100

    categories = ['Age Score', 'Gender Acc', 'Race Acc', 'Gender F1', 'Race F1']
    values = [age_score, gender_score, race_score, gender_f1, race_f1]
    values.append(values[0])
    categories.append(categories[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories,
        fill='toself', fillcolor='rgba(79,70,229,0.1)',
        line=dict(color=C['primary'], width=2.5),
        marker=dict(size=8, color=C['primary']),
        name='Thực tế',
    ))

    target = [80, 94, 88, 93, 85]
    target.append(target[0])
    fig.add_trace(go.Scatterpolar(
        r=target, theta=categories,
        line=dict(color=C['amber'], width=2, dash='dash'),
        name='Mục tiêu', opacity=0.8,
    ))

    fig.update_layout(
        **LAYOUT, height=600,
        title=dict(text='🎯 Radar Hiệu năng Tổng quan', font=dict(size=18)),
        polar=dict(
            bgcolor='rgba(248,250,252,0.8)',
            radialaxis=dict(visible=True, range=[50, 100],
                            gridcolor='rgba(148,163,184,0.2)'),
            angularaxis=dict(gridcolor='rgba(148,163,184,0.2)'),
        ),
        showlegend=True,
        legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='#E2E8F0', borderwidth=1),
    )
    return fig


# ────────────────────────────────────────────────────────────
# 6. Dataset Distribution (Pie Charts)
# ────────────────────────────────────────────────────────────
def chart_dataset_distribution():
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]],
        subplot_titles=('👤 Giới tính', '🌍 Sắc tộc', '🎂 Nhóm tuổi'),
    )

    fig.add_trace(go.Pie(
        labels=['Nam', 'Nữ'], values=[1859, 1697],
        marker=dict(colors=[C['primary'], C['rose']]),
        textinfo='label+percent', hole=0.45,
        textfont=dict(size=13),
    ), row=1, col=1)

    fig.add_trace(go.Pie(
        labels=['White', 'Black', 'Asian', 'Others'],
        values=[1494, 686, 505, 871],
        marker=dict(colors=[C['secondary'], C['amber'], C['cyan'], C['orange']]),
        textinfo='label+percent', hole=0.45,
        textfont=dict(size=12),
    ), row=1, col=2)

    fig.add_trace(go.Pie(
        labels=['0-12', '13-19', '20-35', '36-55', '56+'],
        values=[507, 186, 1558, 785, 520],
        marker=dict(colors=C['palette'][:5]),
        textinfo='label+percent', hole=0.45,
        textfont=dict(size=12),
    ), row=1, col=3)

    fig.update_layout(
        **LAYOUT, height=450,
        title=dict(text='📦 Phân bố Tập Test (3,556 mẫu)', font=dict(size=18)),
        showlegend=False,
    )
    return fig


# ────────────────────────────────────────────────────────────
# 7. Overall Accuracy Comparison
# ────────────────────────────────────────────────────────────
def chart_accuracy_comparison():
    metrics = ['Sai ≤5 năm', 'Sai ≤10 năm', 'Gender Acc', 'Race Acc']
    values = [65.7, 87.4, 93.53, 85.69]
    colors = [C['amber'], C['emerald'], C['rose'], C['cyan']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metrics, y=values,
        marker_color=colors,
        text=[f'{v:.1f}%' for v in values],
        textposition='outside',
        textfont=dict(size=16, color='#0F172A'),
        width=0.5,
    ))
    fig.add_hline(y=90, line_dash='dash', line_color=C['amber'],
                  annotation_text='Target 90%', annotation_position='top right',
                  annotation_font=dict(color=C['amber'], size=12))

    fig.update_layout(
        **LAYOUT, height=500,
        title=dict(text='📊 Tổng quan Độ chính xác', font=dict(size=18)),
        yaxis=dict(range=[0, 105], title='%', gridcolor='rgba(148,163,184,0.2)'),
        showlegend=False,
    )
    return fig


# ────────────────────────────────────────────────────────────
# 8. Loss Curves Only (chi tiết hơn)
# ────────────────────────────────────────────────────────────
def chart_loss_detail(history, version=''):
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train']['loss'] for h in history]
    val_loss = [h['val']['loss'] for h in history]

    best_epoch = epochs[np.argmin(val_loss)]
    best_val = min(val_loss)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=train_loss, name='Train Loss',
        line=dict(color=C['primary'], width=2.5),
        fill='tozeroy', fillcolor='rgba(79,70,229,0.06)',
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=val_loss, name='Val Loss',
        line=dict(color=C['cyan'], width=2.5),
        fill='tozeroy', fillcolor='rgba(8,145,178,0.04)',
    ))
    # Best point
    fig.add_trace(go.Scatter(
        x=[best_epoch], y=[best_val], name=f'Best (ep{best_epoch})',
        mode='markers+text', text=[f'{best_val:.3f}'], textposition='top center',
        marker=dict(size=12, color=C['emerald'], symbol='star'),
        textfont=dict(size=14, color='#0F172A'),
    ))

    fig.update_layout(
        **LAYOUT, height=500,
        title=dict(text=f'📉 Loss Curve {version}', font=dict(size=18)),
        xaxis_title='Epoch', yaxis_title='Loss',
        xaxis=dict(gridcolor='rgba(148,163,184,0.2)'),
        yaxis=dict(gridcolor='rgba(148,163,184,0.2)'),
        legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='#E2E8F0', borderwidth=1),
    )
    return fig


# ────────────────────────────────────────────────────────────
# 9. Age MAE Curve (chi tiết)
# ────────────────────────────────────────────────────────────
def chart_age_mae_detail(history, version=''):
    epochs = [h['epoch'] for h in history]
    train_age = [h['train']['age_mae'] for h in history]
    val_age = [h['val']['age_mae'] for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=train_age, name='Train MAE',
        line=dict(color=C['amber'], width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=val_age, name='Val MAE',
        line=dict(color=C['emerald'], width=2.5, dash='dash'),
    ))
    fig.add_hline(y=4.2, line_dash='dot', line_color=C['rose'],
                  annotation_text='Target ≤ 4.2', annotation_position='top right',
                  annotation_font=dict(color=C['rose'], size=12))

    fig.update_layout(
        **LAYOUT, height=500,
        title=dict(text=f'🎂 Age MAE qua các Epoch {version}', font=dict(size=18)),
        xaxis_title='Epoch', yaxis_title='MAE (năm)',
        xaxis=dict(gridcolor='rgba(148,163,184,0.2)'),
        yaxis=dict(gridcolor='rgba(148,163,184,0.2)'),
        legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='#E2E8F0', borderwidth=1),
    )
    return fig


# ────────────────────────────────────────────────────────────
# 10. Gender + Race Accuracy Curves
# ────────────────────────────────────────────────────────────
def chart_accuracy_curves(history, version=''):
    epochs = [h['epoch'] for h in history]
    train_gender = [h['train']['gender_acc'] for h in history]
    val_gender = [h['val']['gender_acc'] for h in history]
    train_race = [h['train']['race_acc'] for h in history]
    val_race = [h['val']['race_acc'] for h in history]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('👤 Gender Accuracy', '🌍 Race Accuracy'),
                        horizontal_spacing=0.10)

    fig.add_trace(go.Scatter(x=epochs, y=train_gender, name='Train',
                             line=dict(color=C['rose'], width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=val_gender, name='Val',
                             line=dict(color=C['secondary'], width=2.5, dash='dash')), row=1, col=1)

    fig.add_trace(go.Scatter(x=epochs, y=train_race, name='Train',
                             line=dict(color=C['orange'], width=2.5), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=val_race, name='Val',
                             line=dict(color=C['blue'], width=2.5, dash='dash'), showlegend=False), row=1, col=2)

    fig.update_layout(
        **LAYOUT, height=500,
        title=dict(text=f'📊 Accuracy Curves {version}', font=dict(size=18)),
        legend=dict(orientation='h', yanchor='bottom', y=1.06, xanchor='center', x=0.5,
                    bgcolor='rgba(255,255,255,0.9)', bordercolor='#E2E8F0', borderwidth=1),
    )
    fig.update_xaxes(title_text='Epoch', gridcolor='rgba(148,163,184,0.2)')
    fig.update_yaxes(gridcolor='rgba(148,163,184,0.2)')
    return fig


# ────────────────────────────────────────────────────────────
# 11. Summary Score Card
# ────────────────────────────────────────────────────────────
def chart_score_card(test_m):
    age_mae = test_m.get('age_mae', 0)
    gender_acc = test_m.get('gender_accuracy', 0)
    race_acc = test_m.get('race_accuracy', 0)
    score = (100 - age_mae * 2) * 0.3 + gender_acc * 0.35 + race_acc * 0.35
    grade = 'A' if score >= 90 else 'B' if score >= 80 else 'C'

    categories = ['Age MAE ↓', 'Gender Acc ↑', 'Race Acc ↑']
    # Normalize: MAE → 100 - MAE*5 (lower is better)
    age_norm = max(0, 100 - age_mae * 5)
    values = [age_norm, gender_acc, race_acc]
    targets = [79, 94, 88]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Thực tế', x=categories, y=values,
        marker_color=[C['amber'], C['rose'], C['cyan']],
        text=[f'{age_mae:.2f}y', f'{gender_acc:.1f}%', f'{race_acc:.1f}%'],
        textposition='outside', textfont=dict(size=16, color='#0F172A'),
        width=0.35,
    ))
    fig.add_trace(go.Bar(
        name='Mục tiêu', x=categories, y=targets,
        marker_color='rgba(226,232,240,0.5)',
        marker_line=dict(color=C['amber'], width=2),
        text=[f'≤4.2y', f'≥94%', f'≥88%'],
        textposition='outside', textfont=dict(size=13, color='#92400E'),
        width=0.35,
    ))

    fig.update_layout(
        **LAYOUT, height=500, barmode='group',
        title=dict(text=f'🏆 Tổng điểm: {score:.1f}/100 — Xếp hạng {grade}', font=dict(size=20)),
        yaxis=dict(range=[0, 110], title='Score', gridcolor='rgba(148,163,184,0.2)'),
        legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='#E2E8F0', borderwidth=1),
    )
    return fig


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    print('=' * 60)
    print('  📊 XUẤT BIỂU ĐỒ ĐÁNH GIÁ MÔ HÌNH')
    print(f'  📁 Output: {os.path.abspath(CHART_DIR)}')
    print('=' * 60)

    all_data = load_metrics()
    if not all_data:
        print('❌ Không tìm thấy file metrics nào trong logs/')
        return

    # Chọn version chính (ưu tiên v2 → v4 → v3)
    for key in ['v2', 'v4', 'v3']:
        if key in all_data:
            primary = key
            break

    data = all_data[primary]
    test_m = data.get('test_metrics', {})
    history = data.get('history', data.get('epoch_history', []))
    version_tag = f'({primary})'

    print(f'\n  🔵 Primary version: {primary}')
    print(f'     Age MAE: {test_m.get("age_mae", 0):.2f} | '
          f'Gender: {test_m.get("gender_accuracy", 0):.1f}% | '
          f'Race: {test_m.get("race_accuracy", 0):.1f}%\n')

    chart_count = 0

    # ── 1. Training Curves (4-in-1) ──
    if history:
        print('  [1/11] Training Curves...')
        save_fig(chart_training_curves(history, version_tag), '01_training_curves')
        chart_count += 1

    # ── 2. Loss Detail ──
    if history:
        print('  [2/11] Loss Curve...')
        save_fig(chart_loss_detail(history, version_tag), '02_loss_curve')
        chart_count += 1

    # ── 3. Age MAE Detail ──
    if history:
        print('  [3/11] Age MAE Curve...')
        save_fig(chart_age_mae_detail(history, version_tag), '03_age_mae_curve')
        chart_count += 1

    # ── 4. Accuracy Curves ──
    if history:
        print('  [4/11] Accuracy Curves...')
        save_fig(chart_accuracy_curves(history, version_tag), '04_accuracy_curves')
        chart_count += 1

    # ── 5. Gender Confusion Matrix ──
    gcm = test_m.get('gender_confusion_matrix')
    if gcm:
        print('  [5/11] Gender Confusion Matrix...')
        save_fig(chart_confusion_matrix(gcm, ['Male', 'Female'],
                                        '👤 Gender Confusion Matrix'),
                 '05_gender_confusion_matrix')
        chart_count += 1

    # ── 6. Race Confusion Matrix ──
    rcm = test_m.get('race_confusion_matrix')
    if rcm:
        print('  [6/11] Race Confusion Matrix...')
        save_fig(chart_confusion_matrix(rcm, ['White', 'Black', 'Asian', 'Others'],
                                        '🌍 Race Confusion Matrix'),
                 '06_race_confusion_matrix')
        chart_count += 1

    # ── 7. Gender Precision/Recall/F1 ──
    g_report = test_m.get('gender_classification_report')
    if g_report:
        print('  [7/11] Gender Classification Report...')
        save_fig(chart_classification_report(g_report, ['Male', 'Female'],
                                             '👤 Gender — Precision / Recall / F1'),
                 '07_gender_precision_recall_f1')
        chart_count += 1

    # ── 8. Race Precision/Recall/F1 ──
    r_report = test_m.get('race_classification_report')
    if r_report:
        print('  [8/11] Race Classification Report...')
        save_fig(chart_classification_report(r_report, ['White', 'Black', 'Asian', 'Others'],
                                             '🌍 Race — Precision / Recall / F1'),
                 '08_race_precision_recall_f1')
        chart_count += 1

    # ── 9. Age Error by Group ──
    print('  [9/11] Age Error by Group...')
    save_fig(chart_age_error_by_group(), '09_age_error_by_group')
    chart_count += 1

    # ── 10. Radar + Accuracy Overview ──
    if test_m:
        print('  [10/11] Radar + Overview...')
        save_fig(chart_radar(test_m), '10_radar_performance')
        save_fig(chart_accuracy_comparison(), '11_accuracy_overview')
        chart_count += 2

    # ── 11. Dataset Distribution ──
    print('  [11/11] Dataset Distribution...')
    save_fig(chart_dataset_distribution(), '12_dataset_distribution')
    chart_count += 1

    # ── 12. Score Card ──
    if test_m:
        print('  [Bonus] Score Card...')
        save_fig(chart_score_card(test_m), '13_score_card')
        chart_count += 1

    # ── Per-version training curves (nếu có nhiều version) ──
    for ver, d in all_data.items():
        h = d.get('history', d.get('epoch_history', []))
        if ver != primary and h:
            print(f'  [Extra] Training Curves ({ver})...')
            save_fig(chart_training_curves(h, f'({ver})'),
                     f'01_training_curves_{ver}')
            chart_count += 1

    print(f'\n{"=" * 60}')
    print(f'  ✅ Hoàn tất! Đã xuất {chart_count} biểu đồ')
    print(f'  📁 {os.path.abspath(CHART_DIR)}')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
