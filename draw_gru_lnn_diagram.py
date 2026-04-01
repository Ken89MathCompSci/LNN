"""
GRU-LNN Architecture Diagram — clean 2-row layout.

Row 1: Overall pipeline   (Input → BiGRU → Proj → LNN → FC → ŷ)
Row 2: LNN detail panel   (Drive → ODE step → Update, with recurrent arc)
Row 3: Legend
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── palette ──────────────────────────────────────────────────────────────
C_INPUT = '#E3F2FD'
C_GRU_F = '#E8F5E9'
C_GRU_B = '#FFF3E0'
C_PROJ  = '#FCE4EC'
C_LNN   = '#E0F7FA'
C_DRIVE = '#B2EBF2'
C_ODE   = '#B2DFDB'
C_UPD   = '#C8E6C9'
C_FC    = '#FBE9E7'
C_OUT   = '#FFFDE7'
BD      = '#37474F'
TEAL    = '#00838F'
PURPLE  = '#6A1B9A'

fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('#F8F9FA')


# ── helpers ──────────────────────────────────────────────────────────────
def box(x, y, w, h, fc, label, fs=9, ec=BD, lw=1.5, r=0.15):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0.04,rounding_size={r}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fs, fontweight='bold', zorder=4,
            multialignment='center')


def arrow(x1, y1, x2, y2, lbl=None, color=BD, lw=1.8, lbl_offset=0.17):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                        mutation_scale=14), zorder=5)
    if lbl:
        ax.text((x1+x2)/2, (y1+y2)/2 + lbl_offset, lbl,
                ha='center', fontsize=7.5, color='#546E7A', zorder=6)


def dim_label(cx, y, text):
    """Dimension annotation below a block."""
    ax.text(cx, y, text, ha='center', fontsize=7.5,
            color='#78909C', style='italic')


# ══════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════
ax.text(9, 9.65, 'GRU-LNN Architecture  (GRULiquidNetworkModel)',
        ha='center', va='center', fontsize=15,
        fontweight='bold', color='#1A237E')

# ══════════════════════════════════════════════════════════════════════════
# PIPELINE ROW  (centre y = 7.5, box height = 1.3)
# ══════════════════════════════════════════════════════════════════════════
PY = 7.5
BH = 1.3
Y0 = PY - BH / 2   # 6.85

GAP = 0.5   # arrow gap between blocks

# — Input —
IX, IW = 0.3, 1.6
box(IX, Y0, IW, BH, C_INPUT, 'Input\nWindow', fs=9)
dim_label(IX + IW/2, Y0 - 0.3, '(B, T, 1)')
arrow(IX + IW, PY, IX + IW + GAP, PY)

# — BiGRU panel —
BX, BW = IX + IW + GAP, 3.2
ax.add_patch(FancyBboxPatch((BX, Y0 - 0.35), BW, BH + 0.7,
    boxstyle="round,pad=0.05,rounding_size=0.15",
    lw=1.2, edgecolor='#90A4AE', facecolor='#ECEFF1', zorder=2))
ax.text(BX + BW/2, Y0 + BH + 0.2, 'BiGRU Encoder',
        ha='center', fontsize=9, fontweight='bold', color=BD)
# forward sub-row
ax.add_patch(FancyBboxPatch((BX+0.1, PY+0.04), BW-0.2, 0.48,
    boxstyle="round,pad=0.03,rounding_size=0.08",
    lw=1.0, edgecolor='#81C784', facecolor=C_GRU_F, zorder=3))
ax.text(BX + BW/2, PY + 0.28, 'GRU  →  forward',
        ha='center', fontsize=7.5, zorder=4)
ax.annotate('', xy=(BX+BW-0.2, PY+0.28), xytext=(BX+0.2, PY+0.28),
    arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.0), zorder=4)
# backward sub-row
ax.add_patch(FancyBboxPatch((BX+0.1, PY-0.52), BW-0.2, 0.48,
    boxstyle="round,pad=0.03,rounding_size=0.08",
    lw=1.0, edgecolor='#FFB74D', facecolor=C_GRU_B, zorder=3))
ax.text(BX + BW/2, PY - 0.28, '←  GRU  backward',
        ha='center', fontsize=7.5, zorder=4)
ax.annotate('', xy=(BX+0.2, PY-0.28), xytext=(BX+BW-0.2, PY-0.28),
    arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.0), zorder=4)
dim_label(BX + BW/2, Y0 - 0.65, '(B, T, 2·H_gru)')
arrow(BX + BW, PY, BX + BW + GAP, PY)

# — Linear Projection —
PX2, PW2 = BX + BW + GAP, 2.0
box(PX2, Y0, PW2, BH, C_PROJ, 'Linear\nProjection', fs=9)
dim_label(PX2 + PW2/2, Y0 - 0.3, '(B, T, H)')
arrow(PX2 + PW2, PY, PX2 + PW2 + GAP, PY, lbl='x̃₁…x̃ₜ')

# — LNN reservoir (pipeline) —
LX, LW = PX2 + PW2 + GAP, 2.8
box(LX, Y0, LW, BH, C_LNN, 'Liquid\nODE Cell × T', fs=9, ec=TEAL, lw=1.8)
dim_label(LX + LW/2, Y0 - 0.3, '(B, H)')
# recurrent loop arc above the box
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
dim_label(FX + FW/2, Y0 - 0.3, '(B, 1)')
arrow(FX + FW, PY, FX + FW + GAP, PY)

# — Output —
OX, OW = FX + FW + GAP, 1.3
box(OX, Y0, OW, BH, C_OUT, 'ŷ', fs=14)
dim_label(OX + OW/2, Y0 - 0.3, '(B, 1)')

# ══════════════════════════════════════════════════════════════════════════
# LNN DETAIL PANEL  (centre y = 4.1)
# ══════════════════════════════════════════════════════════════════════════
DY  = 4.1    # centre y of sub-panels
SPH = 2.0    # sub-panel height
DY0 = DY - SPH / 2   # 3.1  (sub-panel bottom)

# Outer teal frame
OFX, OFW = 1.8, 14.5
OFY = DY0 - 0.5
OFH = SPH + 1.0
ax.add_patch(FancyBboxPatch((OFX, OFY), OFW, OFH,
    boxstyle="round,pad=0.08,rounding_size=0.3",
    lw=2.0, edgecolor=TEAL, facecolor=C_LNN, zorder=2))
ax.text(OFX + OFW/2, OFY + OFH + 0.12,
        'LiquidODECell — expanded view  (same cell unrolled T times)',
        ha='center', va='bottom', fontsize=10,
        fontweight='bold', color='#004D40')

# Sub-panel geometry
SPW  = 3.5    # sub-panel width
SPGP = 0.75   # gap between sub-panels
SP1X = OFX + 0.85
SP2X = SP1X + SPW + SPGP
SP3X = SP2X + SPW + SPGP


def sp_box(x, fc, title, formula, title_color='#004D40', form_color='#004D40'):
    """Draw one sub-panel with a title and one formula line."""
    ax.add_patch(FancyBboxPatch((x, DY0), SPW, SPH,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        lw=1.5, edgecolor=TEAL, facecolor=fc, zorder=3))
    ax.text(x + SPW/2, DY0 + SPH*0.73, title,
            ha='center', fontsize=10.5, fontweight='bold',
            color=title_color, zorder=4)
    ax.text(x + SPW/2, DY0 + SPH*0.38, formula,
            ha='center', fontsize=9.5, color=form_color, zorder=4)


sp_box(SP1X, C_DRIVE, '① Drive',
       r'$f_t = \tanh(W\tilde{x}_t + Uh_{t-1})$',
       title_color='#00695C', form_color='#00695C')

sp_box(SP2X, C_ODE,   '② ODE step',
       r'$dh = (-h/\tau + f_t)\cdot\Delta t$',
       title_color='#004D40', form_color='#004D40')

sp_box(SP3X, C_UPD,   '③ Update',
       r'$h_t = \mathrm{clamp}(h_{t-1}+dh,\,-10,\,10)$',
       title_color='#1B5E20', form_color='#1B5E20')

# Arrows between sub-panels
arrow(SP1X + SPW, DY, SP2X,       DY, lbl='f_t')
arrow(SP2X + SPW, DY, SP3X,       DY, lbl='dh')

# hₜ₋₁ enters from left
ax.text(OFX + 0.1, DY, 'hₜ₋₁', ha='center', va='center',
        fontsize=9, fontweight='bold', color=PURPLE)
ax.annotate('', xy=(SP1X, DY), xytext=(OFX + 0.4, DY),
    arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.5,
                    mutation_scale=13), zorder=5)

# hₜ exits to right
ax.annotate('', xy=(OFX + OFW - 0.1, DY), xytext=(SP3X + SPW, DY),
    arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.5,
                    mutation_scale=13), zorder=5)
ax.text(OFX + OFW - 0.05, DY, ' hₜ', ha='left', va='center',
        fontsize=9, fontweight='bold', color=TEAL)

# Recurrent feedback arc — BELOW the outer frame
FB_Y = OFY - 0.22
ax.annotate('', xy=(OFX + 0.45, OFY - 0.05),
            xytext=(SP3X + SPW, OFY - 0.05),
            arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.6,
                            connectionstyle='arc3,rad=0.12',
                            mutation_scale=13), zorder=5)
ax.text(OFX + OFW/2, FB_Y,
        'hₜ₋₁  ←  recurrent feedback from previous step',
        ha='center', fontsize=8, color=PURPLE, style='italic')

# ══════════════════════════════════════════════════════════════════════════
# LEGEND  (y ≈ 1.6)
# ══════════════════════════════════════════════════════════════════════════
items = [
    (C_INPUT, 'Input window'),
    (C_GRU_F, 'GRU (forward)'),
    (C_GRU_B, 'GRU (backward)'),
    (C_PROJ,  'Linear projection'),
    (C_LNN,   'LNN reservoir'),
    (C_DRIVE, '① Drive  f_t'),
    (C_ODE,   '② ODE step  dh'),
    (C_UPD,   '③ State update  hₜ'),
    (C_FC,    'FC decoder'),
    (C_OUT,   'Output  ŷ'),
]
LGX, LGY = 0.5, 1.65
ax.text(LGX, LGY + 0.12, 'Legend',
        fontsize=9, fontweight='bold', color=BD)
COL_W = 3.5
for i, (col, lbl) in enumerate(items):
    ci, ri = i % 5, i // 5
    rx = LGX + ci * COL_W
    ry = LGY - 0.45 - ri * 0.50
    ax.add_patch(FancyBboxPatch((rx, ry - 0.14), 0.32, 0.28,
        boxstyle="round,pad=0.02",
        facecolor=col, edgecolor=BD, lw=0.9, zorder=3))
    ax.text(rx + 0.44, ry, lbl, fontsize=8, va='center', color=BD)

# Recurrent feedback swatch in legend
rx_fb = LGX
ry_fb = LGY - 0.45 - 2 * 0.50
ax.annotate('', xy=(rx_fb + 0.32, ry_fb), xytext=(rx_fb, ry_fb),
    arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.5), zorder=4)
ax.text(rx_fb + 0.44, ry_fb, 'Recurrent feedback  hₜ₋₁',
        fontsize=8, va='center', color=PURPLE)

plt.tight_layout(pad=0.3)
plt.savefig('gru_lnn_diagram.png', dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("Saved: gru_lnn_diagram.png")
plt.close()
