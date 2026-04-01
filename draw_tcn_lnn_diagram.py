"""
TCN-LNN Architecture Diagram (TCNLiquidNetworkModel).

Pipeline:
  Input → TCN Encoder (3 dilated blocks) → Linear Proj → LiquidODECell × T → FC → ŷ

Detail panels (below):
  Left:  TCN Encoder — 3 blocks with dilation / channels / receptive field
  Right: LiquidODECell — Drive → ODE step → Update
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── palette ──────────────────────────────────────────────────────────────
C_INPUT  = '#E3F2FD'
C_TCN1   = '#FFF9C4'   # block 1 — yellow
C_TCN2   = '#FFE0B2'   # block 2 — orange
C_TCN3   = '#FFCCBC'   # block 3 — deep orange
C_TCN_BG = '#FFF8E1'   # TCN panel bg
C_PROJ   = '#FCE4EC'
C_LNN    = '#E0F7FA'
C_DRIVE  = '#B2EBF2'
C_ODE    = '#B2DFDB'
C_UPD    = '#C8E6C9'
C_FC     = '#FBE9E7'
C_OUT    = '#FFFDE7'
BD       = '#37474F'
TEAL     = '#00838F'
ORANGE   = '#E65100'
PURPLE   = '#6A1B9A'

fig, ax = plt.subplots(figsize=(20, 11))
ax.set_xlim(0, 20)
ax.set_ylim(0, 11)
ax.axis('off')
fig.patch.set_facecolor('#F8F9FA')


# ── helpers ──────────────────────────────────────────────────────────────
def box(x, y, w, h, fc, label, fs=9, ec=BD, lw=1.5, r=0.15, color='black'):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0.04,rounding_size={r}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fs, fontweight='bold', color=color,
            zorder=4, multialignment='center')


def arrow(x1, y1, x2, y2, lbl=None, color=BD, lw=1.8, lbl_dy=0.17):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                        mutation_scale=14), zorder=5)
    if lbl:
        ax.text((x1+x2)/2, (y1+y2)/2 + lbl_dy, lbl,
                ha='center', fontsize=7.5, color='#546E7A', zorder=6)


def dim(cx, y, text):
    ax.text(cx, y, text, ha='center', fontsize=7.5,
            color='#78909C', style='italic')


# ══════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════
ax.text(10, 10.65, 'TCN-LNN Architecture  (TCNLiquidNetworkModel)',
        ha='center', va='center', fontsize=15,
        fontweight='bold', color='#1A237E')

# ══════════════════════════════════════════════════════════════════════════
# PIPELINE ROW  (centre y = 8.5, box height = 1.3)
# ══════════════════════════════════════════════════════════════════════════
PY = 8.5
BH = 1.3
Y0 = PY - BH / 2   # 7.85

GAP = 0.5

# — Input —
IX, IW = 0.3, 1.6
box(IX, Y0, IW, BH, C_INPUT, 'Input\nWindow', fs=9)
dim(IX + IW/2, Y0 - 0.3, '(B, T, 1)')
arrow(IX + IW, PY, IX + IW + GAP, PY)

# — TCN Encoder panel —
TX, TW = IX + IW + GAP, 4.8
ax.add_patch(FancyBboxPatch((TX, Y0 - 0.4), TW, BH + 0.8,
    boxstyle="round,pad=0.05,rounding_size=0.15",
    lw=1.5, edgecolor=ORANGE, facecolor=C_TCN_BG, zorder=2))
ax.text(TX + TW/2, Y0 + BH + 0.25, 'TCN Encoder',
        ha='center', fontsize=9.5, fontweight='bold', color=ORANGE)

# 3 mini-blocks inside TCN panel
BLK_W, BLK_H = 1.25, 0.95
BLK_Y = PY - BLK_H/2
blk_colors = [C_TCN1, C_TCN2, C_TCN3]
blk_labels = ['Block 1\nd=1', 'Block 2\nd=2', 'Block 3\nd=4']
blk_xs = [TX + 0.12, TX + 0.12 + BLK_W + 0.35, TX + 0.12 + (BLK_W + 0.35)*2]
for i, (bx, col, lbl) in enumerate(zip(blk_xs, blk_colors, blk_labels)):
    box(bx, BLK_Y, BLK_W, BLK_H, col, lbl, fs=8, ec=ORANGE, r=0.1)
    if i < 2:
        arrow(bx + BLK_W, PY, bx + BLK_W + 0.35, PY, color=ORANGE, lw=1.2)
dim(TX + TW/2, Y0 - 0.7, '(B, 128, T)')
arrow(TX + TW, PY, TX + TW + GAP, PY)

# — Linear Projection —
PX2, PW2 = TX + TW + GAP, 2.0
box(PX2, Y0, PW2, BH, C_PROJ, 'Linear\nProjection', fs=9)
dim(PX2 + PW2/2, Y0 - 0.3, '(B, T, 64)')
arrow(PX2 + PW2, PY, PX2 + PW2 + GAP, PY, lbl='x̃₁…x̃ₜ')

# — LNN reservoir —
LX, LW = PX2 + PW2 + GAP, 2.8
box(LX, Y0, LW, BH, C_LNN, 'Liquid\nODE Cell × T', fs=9, ec=TEAL, lw=1.8)
dim(LX + LW/2, Y0 - 0.3, '(B, 64)')
# recurrent loop arc above
ax.annotate('', xy=(LX + 0.25, Y0 + BH + 0.04),
            xytext=(LX + LW - 0.25, Y0 + BH + 0.04),
            arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.6,
                            connectionstyle='arc3,rad=-0.4',
                            mutation_scale=12), zorder=5)
ax.text(LX + LW/2, Y0 + BH + 0.62, 'hₜ₋₁  (recurrent)',
        ha='center', fontsize=7.5, color=PURPLE, style='italic')
arrow(LX + LW, PY, LX + LW + GAP, PY, lbl='h_T')

# — FC Layer —
FX, FW = LX + LW + GAP, 1.7
box(FX, Y0, FW, BH, C_FC, 'FC\nLayer', fs=9)
dim(FX + FW/2, Y0 - 0.3, '(B, 1)')
arrow(FX + FW, PY, FX + FW + GAP, PY)

# — Output —
OX, OW = FX + FW + GAP, 1.3
box(OX, Y0, OW, BH, C_OUT, 'ŷ', fs=14)
dim(OX + OW/2, Y0 - 0.3, '(B, 1)')

# ══════════════════════════════════════════════════════════════════════════
# DETAIL PANELS  (side by side, centre y = 4.3)
# ══════════════════════════════════════════════════════════════════════════
DY  = 4.3
SPH = 2.2
DY0 = DY - SPH / 2   # 3.2

# ── LEFT PANEL: TCN Encoder detail ───────────────────────────────────────
LTCN_X, LTCN_W = 0.2, 9.3
LTCN_Y = DY0 - 0.55
LTCN_H = SPH + 1.1
ax.add_patch(FancyBboxPatch((LTCN_X, LTCN_Y), LTCN_W, LTCN_H,
    boxstyle="round,pad=0.08,rounding_size=0.3",
    lw=2.0, edgecolor=ORANGE, facecolor=C_TCN_BG, zorder=2))
ax.text(LTCN_X + LTCN_W/2, LTCN_Y + LTCN_H + 0.12,
        'TCN Encoder — expanded view',
        ha='center', va='bottom', fontsize=10,
        fontweight='bold', color=ORANGE)

# 3 TCN blocks in detail
DSPW = 2.3   # detail sub-panel width
DSPG = 0.45  # gap between
blk_data = [
    (C_TCN1, 'Block 1',  '1 → 32 channels',  'kernel=3,  dilation=1',  'RF = 3'),
    (C_TCN2, 'Block 2',  '32 → 64 channels', 'kernel=3,  dilation=2',  'RF = 7'),
    (C_TCN3, 'Block 3',  '64 → 128 channels','kernel=3,  dilation=4',  'RF = 15'),
]
blk_start_x = LTCN_X + 0.35
for i, (col, title, ch, params, rf) in enumerate(blk_data):
    bx = blk_start_x + i * (DSPW + DSPG)
    # block box
    ax.add_patch(FancyBboxPatch((bx, DY0), DSPW, SPH,
        boxstyle="round,pad=0.05,rounding_size=0.12",
        lw=1.5, edgecolor=ORANGE, facecolor=col, zorder=3))
    ax.text(bx + DSPW/2, DY0 + SPH*0.82, title,
            ha='center', fontsize=10.5, fontweight='bold', color=ORANGE, zorder=4)
    ax.text(bx + DSPW/2, DY0 + SPH*0.57, ch,
            ha='center', fontsize=8.5, color=BD, zorder=4)
    ax.text(bx + DSPW/2, DY0 + SPH*0.35, params,
            ha='center', fontsize=8, color='#5D4037', style='italic', zorder=4)
    ax.text(bx + DSPW/2, DY0 + SPH*0.13, rf,
            ha='center', fontsize=9, fontweight='bold', color=ORANGE, zorder=4)
    if i < 2:
        arrow(bx + DSPW, DY, bx + DSPW + DSPG, DY, color=ORANGE, lw=1.5)

# RF growth annotation
ax.text(LTCN_X + LTCN_W/2, LTCN_Y - 0.25,
        'Receptive field grows as  3 → 7 → 15  timesteps  '
        '(RF = 1 + (k−1)·(1 + 2 + 4))',
        ha='center', fontsize=8, color=ORANGE, style='italic')

# Conv + ReLU + Dropout note
ax.text(LTCN_X + LTCN_W/2, DY0 - 0.28,
        'Each block:  Conv1d → ReLU → Dropout(0.2)',
        ha='center', fontsize=8.5, color=BD)

# ── RIGHT PANEL: LNN ODE detail ──────────────────────────────────────────
RLNN_X, RLNN_W = 9.9, 9.9
RLNN_Y = DY0 - 0.55
RLNN_H = SPH + 1.1
ax.add_patch(FancyBboxPatch((RLNN_X, RLNN_Y), RLNN_W, RLNN_H,
    boxstyle="round,pad=0.08,rounding_size=0.3",
    lw=2.0, edgecolor=TEAL, facecolor=C_LNN, zorder=2))
ax.text(RLNN_X + RLNN_W/2, RLNN_Y + RLNN_H + 0.12,
        'LiquidODECell — expanded view  (same cell unrolled T times)',
        ha='center', va='bottom', fontsize=10,
        fontweight='bold', color='#004D40')

# 3 LNN sub-panels
LSPW = 2.6
LSPG = 0.55
lnn_data = [
    (C_DRIVE, '① Drive',    r'$f_t = \tanh(W\tilde{x}_t + Uh_{t-1})$', '#00695C'),
    (C_ODE,   '② ODE step', r'$dh = (-h/\tau + f_t)\cdot\Delta t$',     '#004D40'),
    (C_UPD,   '③ Update',   r'$h_t = \mathrm{clamp}(h_{t-1}+dh,\,-10,\,10)$', '#1B5E20'),
]
lsp_start_x = RLNN_X + 0.75
for i, (col, title, formula, fc) in enumerate(lnn_data):
    lx = lsp_start_x + i * (LSPW + LSPG)
    ax.add_patch(FancyBboxPatch((lx, DY0), LSPW, SPH,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        lw=1.5, edgecolor=TEAL, facecolor=col, zorder=3))
    ax.text(lx + LSPW/2, DY0 + SPH*0.73, title,
            ha='center', fontsize=10.5, fontweight='bold', color=fc, zorder=4)
    ax.text(lx + LSPW/2, DY0 + SPH*0.35, formula,
            ha='center', fontsize=9, color=fc, zorder=4)
    if i < 2:
        lbl = 'f_t' if i == 0 else 'dh'
        arrow(lx + LSPW, DY, lx + LSPW + LSPG, DY, lbl=lbl, color=TEAL, lw=1.5)

# hₜ₋₁ enters
ax.text(RLNN_X + 0.2, DY, 'hₜ₋₁', ha='center', va='center',
        fontsize=9, fontweight='bold', color=PURPLE)
ax.annotate('', xy=(lsp_start_x, DY), xytext=(RLNN_X + 0.45, DY),
    arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.5,
                    mutation_scale=13), zorder=5)

# hₜ exits
lx_last = lsp_start_x + 2 * (LSPW + LSPG)
ax.annotate('', xy=(RLNN_X + RLNN_W - 0.1, DY),
            xytext=(lx_last + LSPW, DY),
    arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.5,
                    mutation_scale=13), zorder=5)
ax.text(RLNN_X + RLNN_W - 0.05, DY, ' hₜ', ha='left', va='center',
        fontsize=9, fontweight='bold', color=TEAL)

# τ note
ax.text(RLNN_X + RLNN_W/2, DY0 - 0.28,
        'τ ∈ ℝᴴ — learnable time constants (one per neuron, init = 1)',
        ha='center', fontsize=8.5, color='#004D40')

# Recurrent feedback arc
ax.annotate('', xy=(RLNN_X + 0.75, RLNN_Y - 0.05),
            xytext=(lx_last + LSPW, RLNN_Y - 0.05),
    arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.6,
                    connectionstyle='arc3,rad=0.12',
                    mutation_scale=13), zorder=5)
ax.text(RLNN_X + RLNN_W/2, RLNN_Y - 0.28,
        'hₜ₋₁  ←  recurrent feedback from previous step',
        ha='center', fontsize=8, color=PURPLE, style='italic')

# ══════════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════════
items = [
    (C_INPUT, 'Input window'),
    (C_TCN1,  'TCN Block 1  (d=1)'),
    (C_TCN2,  'TCN Block 2  (d=2)'),
    (C_TCN3,  'TCN Block 3  (d=4)'),
    (C_PROJ,  'Linear projection'),
    (C_LNN,   'LNN reservoir'),
    (C_DRIVE, '① Drive  f_t'),
    (C_ODE,   '② ODE step  dh'),
    (C_UPD,   '③ State update  hₜ'),
    (C_FC,    'FC decoder'),
    (C_OUT,   'Output  ŷ'),
]
LGX, LGY = 0.4, 1.65
ax.text(LGX, LGY + 0.12, 'Legend', fontsize=9, fontweight='bold', color=BD)
COL_W = 3.6
for i, (col, lbl) in enumerate(items):
    ci, ri = i % 6, i // 6
    rx = LGX + ci * COL_W
    ry = LGY - 0.45 - ri * 0.50
    ax.add_patch(FancyBboxPatch((rx, ry - 0.13), 0.30, 0.26,
        boxstyle="round,pad=0.02",
        facecolor=col, edgecolor=BD, lw=0.9, zorder=3))
    ax.text(rx + 0.42, ry, lbl, fontsize=8, va='center', color=BD)

# Recurrent swatch
rx_fb = LGX
ry_fb = LGY - 0.45 - 2 * 0.50
ax.annotate('', xy=(rx_fb + 0.30, ry_fb), xytext=(rx_fb, ry_fb),
    arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.5), zorder=4)
ax.text(rx_fb + 0.42, ry_fb, 'Recurrent feedback  hₜ₋₁',
        fontsize=8, va='center', color=PURPLE)

plt.tight_layout(pad=0.3)
plt.savefig('tcn_lnn_diagram.png', dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("Saved: tcn_lnn_diagram.png")
plt.close()
