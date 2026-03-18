"""
Draw a publication-quality diagram of the Standard LNN Cell (LiquidTimeLayer)
matching the visual style of the LSTM cell diagram.

ODE:  dh/dt = -h/τ + tanh(W·x_t + U·h_{t-1})
Step: h_t   = clamp(h_{t-1} + dh·Δt, -10, 10)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 9))
ax.set_xlim(0, 18)
ax.set_ylim(0, 9)
ax.axis('off')
fig.patch.set_facecolor('#FFFEF5')
ax.set_facecolor('#FFFEF5')

# ── Panel background ──────────────────────────────────────────────────────────
panel = FancyBboxPatch((0.25, 0.75), 17.2, 7.55,
    boxstyle="round,pad=0.12",
    facecolor='#EEEEF8', edgecolor='#CCCCCC', linewidth=1.5, zorder=1)
ax.add_patch(panel)

# ── Colour palette ─────────────────────────────────────────────────────────────
RED_B,    RED_E    = '#F4AAAA', '#B83030'   # decay / -h/τ
BLUE_B,   BLUE_E   = '#AACCEE', '#2266AA'   # input projection W
GREEN_B,  GREEN_E  = '#AADDAA', '#228844'   # recurrent proj U  / tanh
ORANGE_B, ORANGE_E = '#FFD08C', '#CC8800'   # τ parameter
GRAY_B,   GRAY_E   = '#CCCCCC', '#777777'   # clamp
PURPLE_B, PURPLE_E = '#C8B8E8', '#5535A8'   # state highway h

# ── Helpers ───────────────────────────────────────────────────────────────────
def draw_box(cx, cy, w, h, title, subtitle, bg, edge, fs=10.5, zorder=5):
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.1",
        facecolor=bg, edgecolor=edge, linewidth=2, zorder=zorder)
    ax.add_patch(p)
    yoff = 0.13 if subtitle else 0
    ax.text(cx, cy + yoff, title, ha='center', va='center',
            fontsize=fs, fontweight='bold', zorder=zorder+1)
    if subtitle:
        ax.text(cx, cy - 0.22, subtitle, ha='center', va='center',
                fontsize=8, color='#444444', zorder=zorder+1)

def draw_circle(cx, cy, r, sym, bg='white', edge='#4488CC', zorder=5):
    c = plt.Circle((cx, cy), r, facecolor=bg, edgecolor=edge,
                   linewidth=2.2, zorder=zorder)
    ax.add_patch(c)
    ax.text(cx, cy, sym, ha='center', va='center',
            fontsize=13, fontweight='bold', zorder=zorder+1)

def arr(x0, y0, x1, y1, col='#444444', lw=1.8, rad=0.0, zorder=4, style='->'):
    cs = f'arc3,rad={rad}'
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle=style, color=col, lw=lw,
                        connectionstyle=cs), zorder=zorder)

def label(x, y, txt, **kw):
    ax.text(x, y, txt, ha='center', va='center', fontsize=9.5, **kw)

# ── Y levels ──────────────────────────────────────────────────────────────────
HW_Y   = 6.3    # state highway
TANH_Y = 4.3    # tanh box
PROJ_Y = 2.4    # projection boxes
ADD_Y  = 3.35   # Wx + Uh adder circle

# ── X positions (left → right) ─────────────────────────────────────────────────
X_IN   = 0.9    # h_{t-1} entry
X_DEC  = 3.3    # × (-1/τ) decay circle
X_P1   = 7.0    # + circle: (-h/τ + f_t)
X_DT   = 9.7    # × Δt circle
X_P2   = 12.4   # + circle: h_{t-1} + dh
X_CLP  = 15.1   # clamp box
X_OUT  = 17.2   # h_t exit

X_WPRJ = 4.7    # W·x_t projection box
X_UPRJ = 8.8    # U·h_{t-1} projection box
X_PADD = 6.7    # + circle for Wx + Uh

# ── Title & equation ──────────────────────────────────────────────────────────
ax.text(9, 8.72, 'Standard LNN Cell (LiquidTimeLayer)',
        ha='center', va='center', fontsize=16, fontweight='bold', zorder=10)
ax.text(9, 8.22,
    r'$h_t = \mathrm{clamp}\!\left(h_{t-1}+'
    r'\!\left(\dfrac{-h_{t-1}}{\tau}+\tanh(Wx_t+Uh_{t-1})\right)'
    r'\!\cdot\!\Delta t,\;-10,\;10\right)$',
    ha='center', va='center', fontsize=11, zorder=10)

# ── State highway line ─────────────────────────────────────────────────────────
# Entry arrow
arr(0.28, HW_Y, X_IN, HW_Y, col=PURPLE_E, lw=3)
ax.text(0.62, HW_Y + 0.34, r'$h_{t-1}$',
        ha='center', fontsize=11.5, fontweight='bold', color=PURPLE_E)

# Highway segments (drawn as plain lines, arrows added between components)
for (xa, xb) in [(X_IN, X_DEC - 0.34),
                 (X_DEC + 0.34, X_P1 - 0.34),
                 (X_P1 + 0.34, X_DT - 0.34),
                 (X_DT + 0.34, X_P2 - 0.34),
                 (X_P2 + 0.34, X_CLP - 0.97),
                 (X_CLP + 0.97, X_OUT)]:
    ax.annotate('', xy=(xb, HW_Y), xytext=(xa, HW_Y),
        arrowprops=dict(arrowstyle='->', color=PURPLE_E, lw=2.4), zorder=3)

# Exit label
ax.text(X_OUT + 0.18, HW_Y + 0.34, r'$h_t$',
        ha='left', fontsize=11.5, fontweight='bold', color=PURPLE_E)

# h_{t-1} recurrent feedback (dashed, routed below projections above legend)
FEED_Y = 1.65
ax.plot([X_OUT, X_OUT], [HW_Y, FEED_Y], color=PURPLE_E, lw=1.6,
        linestyle='--', zorder=3)
ax.plot([X_OUT, X_UPRJ], [FEED_Y, FEED_Y], color=PURPLE_E, lw=1.6,
        linestyle='--', zorder=3)
arr(X_UPRJ, FEED_Y, X_UPRJ, PROJ_Y - 0.39, col=PURPLE_E, lw=1.6, zorder=3)
ax.text((X_OUT + X_UPRJ) / 2, FEED_Y + 0.18, r'$h_{t-1}$',
        ha='center', fontsize=9, color=PURPLE_E, style='italic')

# ── × (−1/τ) DECAY circle ──────────────────────────────────────────────────────
draw_circle(X_DEC, HW_Y, 0.34, r'$\times$', bg=RED_B, edge=RED_E)

# τ parameter box (above decay)
draw_box(X_DEC, HW_Y + 1.55, 1.55, 0.72, r'$\tau$', 'Time constant',
         ORANGE_B, ORANGE_E, fs=13)
arr(X_DEC, HW_Y + 1.19, X_DEC, HW_Y + 0.36, col=ORANGE_E, lw=1.8)
ax.text(X_DEC + 0.28, HW_Y + 0.82, r'$-1/\tau$',
        fontsize=9, color=ORANGE_E)

# label between decay and +1
ax.text((X_DEC + X_P1) / 2, HW_Y + 0.3, r'$-h_{t-1}/\tau$',
        ha='center', fontsize=9, color=RED_E)

# ── + circle: −h/τ + f_t ──────────────────────────────────────────────────────
draw_circle(X_P1, HW_Y, 0.34, '+', bg='#E0EEFF', edge='#3355AA')

# ── × Δt circle ───────────────────────────────────────────────────────────────
draw_circle(X_DT, HW_Y, 0.34, r'$\times$', bg='#E4F6E4', edge='#2A7A2A')
ax.text(X_DT, HW_Y - 0.6, r'$\Delta t$', ha='center',
        fontsize=9.5, color='#2A7A2A')

# dh label between ×Δt and +2
ax.text((X_DT + X_P2) / 2, HW_Y + 0.3, r'$\mathrm{d}h$',
        ha='center', fontsize=9, color='#2A7A2A')

# ── + circle: h_{t-1} + dh ────────────────────────────────────────────────────
draw_circle(X_P2, HW_Y, 0.34, '+', bg=PURPLE_B, edge=PURPLE_E)

# Residual skip: h_{t-1} branches down around and into +2
ax.annotate('', xy=(X_P2, HW_Y - 0.34), xytext=(X_IN, HW_Y),
    arrowprops=dict(arrowstyle='->', color=PURPLE_E, lw=1.7,
                    connectionstyle='arc3,rad=-0.22'), zorder=4)
ax.text(10.5, 5.28, r'$h_{t-1}$ (residual)', fontsize=8.5,
        color=PURPLE_E, ha='center', style='italic')

# ── Clamp box ─────────────────────────────────────────────────────────────────
draw_box(X_CLP, HW_Y, 1.92, 0.76, 'clamp', '(-10, 10)',
         GRAY_B, GRAY_E, fs=10.5)

# ── tanh box ──────────────────────────────────────────────────────────────────
draw_box(X_P1, TANH_Y, 1.55, 0.76, 'tanh', None,
         GREEN_B, GREEN_E, fs=12)

# tanh → +1 (upward)
arr(X_P1, TANH_Y + 0.38, X_P1, HW_Y - 0.34, col=GREEN_E, lw=2)
ax.text(X_P1 + 0.32, (TANH_Y + HW_Y) / 2, r'$f_t$',
        fontsize=10.5, color=GREEN_E, fontweight='bold')

# ── W·x_t input projection box ───────────────────────────────────────────────
draw_box(X_WPRJ, PROJ_Y, 2.4, 0.78, r'$W \cdot x_t$', 'Input Projection',
         BLUE_B, BLUE_E, fs=11)

# ── U·h_{t-1} recurrent projection box ───────────────────────────────────────
draw_box(X_UPRJ, PROJ_Y, 2.6, 0.78, r'$U \cdot h_{t-1}$', 'Recurrent Projection',
         GREEN_B, GREEN_E, fs=11)

# ── + circle: Wx_t + Uh_{t-1} ─────────────────────────────────────────────────
draw_circle(X_PADD, ADD_Y, 0.3, '+', bg='#FFFBE0', edge='#AA8800')

# W proj → +add
arr(X_WPRJ, PROJ_Y + 0.39, X_PADD - 0.15, ADD_Y - 0.22, col=BLUE_E, lw=1.8)
# U proj → +add
arr(X_UPRJ, PROJ_Y + 0.39, X_PADD + 0.20, ADD_Y - 0.22,
    col=GREEN_E, lw=1.8, rad=-0.12)

# +add → tanh
arr(X_PADD, ADD_Y + 0.30, X_P1, TANH_Y - 0.38, col='#AA8800', lw=1.8)

# ── x_t input (bottom, below W box) ──────────────────────────────────────────
arr(X_WPRJ, 1.3, X_WPRJ, PROJ_Y - 0.39, col=BLUE_E, lw=2.2)
ax.text(X_WPRJ, 1.08, r'$x_t$',
        ha='center', fontsize=13, fontweight='bold', color=BLUE_E)

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_data = [
    (RED_B,    RED_E,    r'Decay gate  $-h_{t-1}/\tau$'),
    (BLUE_B,   BLUE_E,   r'Input projection  $W \cdot x_t$'),
    (GREEN_B,  GREEN_E,  r'Recurrent projection  $U \cdot h_{t-1}$  /  tanh'),
    (ORANGE_B, ORANGE_E, r'Time constant  $\tau$'),
    (GRAY_B,   GRAY_E,   r'Clamp  $(-10,\;10)$'),
    (PURPLE_B, PURPLE_E, r'Cell state  $h_t$'),
]
xleg = 0.6
for (bg, eg, lbl) in legend_data:
    r = FancyBboxPatch((xleg, 0.82), 0.52, 0.38,
        boxstyle="round,pad=0.04",
        facecolor=bg, edgecolor=eg, linewidth=1.5, zorder=8)
    ax.add_patch(r)
    ax.text(xleg + 0.7, 1.01, lbl,
            va='center', fontsize=8.5, zorder=9)
    xleg += 2.9

plt.tight_layout()
plt.savefig('lnn_cell_diagram.png', dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("Saved: lnn_cell_diagram.png")
