"""
draw_transformer_architecture.py
=================================
Generates a tidy three-panel Transformer Encoder Architecture diagram.
Panels: Encoder Stack  |  Single Encoder Layer  |  Multi-Head Self-Attention

Run:
    python draw_transformer_architecture.py
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Canvas ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 11))
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis('off')
BG = '#F8F9FA'
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# ── Colour palette ─────────────────────────────────────────────────────────────
TEAL_B, TEAL_E = '#E0F7FA', '#00838F'   # input embedding
PINK_B, PINK_E = '#FCE4EC', '#AD1457'   # positional encoding
BLUE_B, BLUE_E = '#E3F2FD', '#1565C0'   # encoder layers / linear
ORNG_B, ORNG_E = '#FFF3E0', '#E65100'   # layer norm
GRNE_B, GRNE_E = '#E8F5E9', '#2E7D32'   # FFN / FC output
PURP_B, PURP_E = '#EDE7F6', '#6A1B9A'   # attention / Q K V
YELO_B, YELO_E = '#FFFDE7', '#F9A825'   # attention equations box
GREY_B, GREY_E = '#ECEFF1', '#546E7A'   # input placeholder boxes
SKIP_C         = '#C62828'              # residual skip (red)
ARR_C          = '#37474F'              # default arrow

# ── Helpers ────────────────────────────────────────────────────────────────────
def rbox(x, y, w, h, fc, ec, lw=1.8, zo=4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
                 linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zo))

def txt(x, y, s, fs=9, c='#212121', bold=False, italic=False, ha='center', zo=5):
    ax.text(x, y, s, ha=ha, va='center', fontsize=fs, color=c, zorder=zo,
            fontweight='bold' if bold else 'normal',
            fontstyle='italic' if italic else 'normal')

def arr(x1, y1, x2, y2, c=ARR_C, lw=1.8, zo=6):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw, mutation_scale=14))

def seg(xs, ys, c=ARR_C, lw=1.5, ls='-'):
    ax.plot(xs, ys, color=c, lw=lw, linestyle=ls, zorder=5)

def hdiv(x0, x1, y, c='#DDDDDD'):
    seg([x0, x1], [y, y], c=c, lw=1.0)

def plus_circle(cx, cy, r=0.22):
    ax.add_patch(plt.Circle((cx, cy), r, fc='white', ec=SKIP_C, lw=2.0, zorder=7))
    txt(cx, cy, '+', fs=11, c=SKIP_C, bold=True, zo=8)

# ── Title ──────────────────────────────────────────────────────────────────────
txt(9, 10.75, 'Transformer Encoder Architecture', fs=16, bold=True)
txt(9, 10.42,
    'Self-attention captures global dependencies across the full sequence in O(1) layers',
    fs=9.5, c='#546E7A', italic=True)

# Panel geometry
PW = 5.3
P1X, P1CX = 0.2,  0.2  + PW/2
P2X, P2CX = 6.25, 6.25 + PW/2
P3X, P3CX = 12.3, 12.3 + PW/2
PH, PY    = 9.9, 0.2

for px, title, sub in [
    (P1X, 'Encoder Stack',             'prediction'),
    (P2X, 'Single Encoder Layer',      'output'),
    (P3X, 'Multi-Head Self-Attention', 'attention output'),
]:
    cx = px + PW/2
    rbox(px, PY, PW, PH, '#F5F5F5', '#B0BEC5', lw=1.5, zo=2)
    txt(cx, PY + PH - 0.26, sub,   fs=8,   c='#888888', italic=True)
    txt(cx, PY + PH - 0.55, title, fs=11,  bold=True)

# ══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL — ENCODER STACK
# ══════════════════════════════════════════════════════════════════════════════
BW1, BX1 = 4.5, P1CX - 2.25

# Input arrow
txt(P1CX, 0.25, 'x_t   (input sequence)', fs=8.5, c='#555555')
arr(P1CX, 0.38, P1CX, 0.65)

# Boxes: (center_y, fc, ec, title, subtitle)
stack = [
    (1.05, TEAL_B, TEAL_E, 'Input Embedding',  '→  d_model'),
    (2.15, PINK_B, PINK_E, 'Positional Enc.',  '+ positional info'),
    (3.35, BLUE_B, BLUE_E, 'Encoder Layer 1',  None),
    (4.45, BLUE_B, BLUE_E, 'Encoder Layer 2',  None),
    (5.95, BLUE_B, BLUE_E, 'Encoder Layer N',  'x N stacked'),
    (7.25, GRNE_B, GRNE_E, 'FC Output',        'Global Avg Pool + Linear'),
]
prev_top = 0.65
for cy, fb, ec, title, sub in stack:
    H = 0.72
    rbox(BX1, cy - H/2, BW1, H, fb, ec, zo=4)
    if sub:
        txt(P1CX, cy + 0.13, title, fs=9.5, bold=True)
        txt(P1CX, cy - 0.14, sub,   fs=8,   c='#666666')
    else:
        txt(P1CX, cy, title, fs=9.5, bold=True)
    arr(P1CX, prev_top, P1CX, cy - H/2)
    prev_top = cy + H/2

# Dots between layer 2 and layer N
txt(P1CX, 4.95, '⋮', fs=18, c='#999999')

arr(P1CX, prev_top, P1CX, 8.65)
txt(P1CX, 8.78, 'prediction', fs=8.5, c='#888888', italic=True)

# ══════════════════════════════════════════════════════════════════════════════
# MIDDLE PANEL — SINGLE ENCODER LAYER  (Pre-LN)
# ══════════════════════════════════════════════════════════════════════════════
BW2, BX2 = 4.2, P2CX - 2.1
RX = BX2 + BW2 + 0.22   # x of right-side skip lines

# Input box
rbox(BX2, 0.35, BW2, 0.72, GREY_B, GREY_E, zo=4)
txt(P2CX, 0.73, 'Input',          fs=9.5, bold=True)
txt(P2CX, 0.50, 'from prev layer', fs=8,  c='#666666')
arr(P2CX, 1.07, P2CX, 1.45)

# LayerNorm 1  (Pre-LN: norm BEFORE attention)
rbox(BX2, 1.45, BW2, 0.65, ORNG_B, ORNG_E, zo=4)
txt(P2CX, 1.775, 'LayerNorm', fs=9.5, bold=True, c='#E65100')
arr(P2CX, 2.10, P2CX, 2.45)

# Multi-Head Self-Attn
rbox(BX2, 2.45, BW2, 0.82, BLUE_B, BLUE_E, lw=1.8, zo=4)
txt(P2CX, 2.93, 'Multi-Head Self-Attn', fs=9.5, bold=True, c='#1565C0')
txt(P2CX, 2.65, 'Attention(Q=x, K=x, V=x)', fs=8.5, c='#555555')
arr(P2CX, 3.27, P2CX, 3.65, SKIP_C, 1.8)

# Skip 1: right-side line from Input top → + circle 1  (bypasses LN1 + MHSA)
seg([RX, RX], [1.07, 3.85], SKIP_C, 1.8)
arr(RX, 3.85, P2CX + 0.25, 3.85, SKIP_C, 1.8)
txt(RX + 0.15, 2.45, 'skip 1', fs=8, c=SKIP_C, italic=True, ha='left')

# + circle 1  (residual add: x_in + MHSA output)
plus_circle(P2CX, 3.85)
arr(P2CX, 4.07, P2CX, 4.45)

# LayerNorm 2  (Pre-LN: norm BEFORE FFN)
rbox(BX2, 4.45, BW2, 0.65, ORNG_B, ORNG_E, zo=4)
txt(P2CX, 4.775, 'LayerNorm', fs=9.5, bold=True, c='#E65100')
arr(P2CX, 5.10, P2CX, 5.45)

# Feed-Forward
rbox(BX2, 5.45, BW2, 0.82, GRNE_B, GRNE_E, lw=1.8, zo=4)
txt(P2CX, 5.93, 'Feed-Forward',      fs=9.5, bold=True, c='#2E7D32')
txt(P2CX, 5.65, 'Linear → ReLU → Linear', fs=8.5, c='#555555')
arr(P2CX, 6.27, P2CX, 6.65, SKIP_C, 1.8)

# Skip 2: right-side line from after + circle 1 → + circle 2  (bypasses LN2 + FFN)
seg([RX, RX], [4.07, 6.85], SKIP_C, 1.8)
arr(RX, 6.85, P2CX + 0.25, 6.85, SKIP_C, 1.8)
txt(RX + 0.15, 5.50, 'skip 2', fs=8, c=SKIP_C, italic=True, ha='left')

# + circle 2  (residual add: x_mid + FFN output)
plus_circle(P2CX, 6.85)
arr(P2CX, 7.07, P2CX, 7.45)
txt(P2CX, 7.62, 'output', fs=8.5, c='#888888', italic=True)

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — MULTI-HEAD SELF-ATTENTION
# ══════════════════════════════════════════════════════════════════════════════
BW3, BX3 = 4.6, P3CX - 2.3

# Input x
rbox(BX3, 0.35, BW3, 0.72, GREY_B, GREY_E, zo=4)
txt(P3CX, 0.73, 'Input  x',          fs=9.5, bold=True)
txt(P3CX, 0.50, 'same x for Q, K, V', fs=8,  c='#666666')
arr(P3CX, 1.07, P3CX, 1.45)

# Q / K / V Projections
rbox(BX3, 1.45, BW3, 0.82, PURP_B, PURP_E, lw=1.8, zo=4)
txt(P3CX, 1.93, 'Q / K / V  Projections', fs=9.5, bold=True, c='#6A1B9A')
txt(P3CX, 1.65, 'W_q x,   W_k x,   W_v x',  fs=8.5, c='#555555')
arr(P3CX, 2.27, P3CX, 2.65)

# Scaled Dot-Product Attention
rbox(BX3, 2.65, BW3, 0.82, PURP_B, PURP_E, lw=1.8, zo=4)
txt(P3CX, 3.13, 'Scaled Dot-Product Attn', fs=9.5, bold=True, c='#6A1B9A')
txt(P3CX, 2.85, 'x H heads in parallel',   fs=8.5, c='#555555')
arr(P3CX, 3.47, P3CX, 3.85)

# Concat + Linear W_o
rbox(BX3, 3.85, BW3, 0.82, BLUE_B, BLUE_E, lw=1.8, zo=4)
txt(P3CX, 4.33, 'Concat + Linear  W_o', fs=9.5, bold=True, c='#1565C0')
txt(P3CX, 4.05, 'merge all heads',       fs=8.5, c='#555555')
arr(P3CX, 4.67, P3CX, 5.1)

# Attention(Q,K,V) equations box  (height trimmed to leave room for output label)
rbox(BX3, 5.1, BW3, 3.4, YELO_B, YELO_E, lw=2.0, zo=4)
txt(P3CX, 8.35, 'Attention( Q, K, V )',             fs=11,  bold=True, c='#7B6000')
txt(P3CX, 7.90, '=  softmax( QKᵀ / √d_k )  ·  V',  fs=10,  c='#333333')
hdiv(BX3 + 0.3, BX3 + BW3 - 0.3, 7.55)
txt(P3CX, 7.22, 'd_k  =  d_model / num_heads',       fs=9,   c='#555555')
hdiv(BX3 + 0.3, BX3 + BW3 - 0.3, 6.88)
txt(P3CX, 6.52, 'Each head learns',                  fs=9,   c='#7B6000', italic=True)
txt(P3CX, 6.20, 'different attention patterns',       fs=9,   c='#7B6000', italic=True)
hdiv(BX3 + 0.3, BX3 + BW3 - 0.3, 5.88)
txt(P3CX, 5.55, 'Complexity:  O( T² · d )',           fs=8.5, c='#888888', italic=True)
txt(P3CX, 5.22, 'T = sequence length,  d = d_model',  fs=8.5, c='#888888', italic=True)

arr(P3CX, 8.50, P3CX, 9.05)
txt(P3CX, 9.22, 'attention output', fs=8.5, c='#888888', italic=True)

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_items = [
    (TEAL_B, TEAL_E, 'Embedding + PosEnc'),
    (BLUE_B, BLUE_E, 'Attention / Linear'),
    (PURP_B, PURP_E, 'Q / K / V per head'),
    (GRNE_B, GRNE_E, 'Feed-Forward MLP'),
    (ORNG_B, ORNG_E, 'LayerNorm'),
    ('#FFFFFF', SKIP_C, 'Residual skip (+)'),
]
lx = 0.35
for fb, ec, name in legend_items:
    ax.add_patch(FancyBboxPatch((lx, 0.04), 0.45, 0.15, boxstyle='round,pad=0.03',
                 linewidth=1.2, edgecolor=ec, facecolor=fb, zorder=6))
    txt(lx + 0.55, 0.12, name, fs=7.5, c='#444444', ha='left', zo=6)
    lx += 2.9

# ── Save ───────────────────────────────────────────────────────────────────────
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'transformer_diagram.png')
plt.tight_layout(pad=0.3)
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
print('Saved: ' + save_path)
