import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.axis('off')
fig.patch.set_facecolor('#FFFBF0')
ax.set_facecolor('#FFFBF0')

# ── helpers ───────────────────────────────────────────────────────────────

def rbox(ax, x, y, w, h, fc, ec, lw=1.8, zorder=4):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder)
    ax.add_patch(p)

def label(ax, x, y, text, fs=9, color='black', bold=False, italic=False, zorder=5, ha='center'):
    w = 'bold' if bold else 'normal'
    s = 'italic' if italic else 'normal'
    ax.text(x, y, text, ha=ha, va='center', fontsize=fs, color=color,
            fontweight=w, fontstyle=s, zorder=zorder)

def arr(ax, x1, y1, x2, y2, color='#333333', lw=1.8, zorder=6):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=14))

# ── Title ─────────────────────────────────────────────────────────────────
label(ax, 8, 10.65, 'TRANSFORMER ENCODER + LNN', fs=17, bold=True)
label(ax, 8, 10.3, 'Architecture for NILM', fs=10, color='#555555', italic=True)

# outer border
rbox(ax, 0.2, 0.25, 15.6, 9.75, fc='none', ec='#8B4513', lw=2.5, zorder=1)

# ══════════════════════════════════════════════════════════════════════════
#  LEFT  –  TRANSFORMER ENCODER BLOCK
# ══════════════════════════════════════════════════════════════════════════
rbox(ax, 0.35, 0.35, 8.1, 9.4, fc='#EBF5FF', ec='#1565C0', lw=2.2, zorder=2)
label(ax, 4.4, 9.52, 'TRANSFORMER ENCODER BLOCK', fs=11, bold=True, color='#1565C0')
label(ax, 4.4, 9.22, 'Global Feature Extraction', fs=9, color='#1565C0', italic=True)

# ── Input x ───────────────────────────────────────────────────────────────
rbox(ax, 0.6, 4.5, 1.6, 0.75, fc='#E8F5E9', ec='#388E3C', zorder=4)
label(ax, 1.4, 4.875, 'Input  x', fs=10, bold=True)
label(ax, 1.4, 4.62, '(B, 100, 1)', fs=8, color='#555555')

arr(ax, 2.2, 4.875, 2.65, 4.875)

# ── Input Projection ──────────────────────────────────────────────────────
rbox(ax, 2.65, 4.25, 2.1, 1.25, fc='#E8F5E9', ec='#388E3C', zorder=4)
label(ax, 3.7, 5.05, 'Input Projection', fs=9, bold=True)
label(ax, 3.7, 4.73, 'Linear( 1 → H )', fs=8.5, color='#333333')
label(ax, 3.7, 4.46, '(B, 100, H)', fs=8, color='#666666')

arr(ax, 4.75, 4.875, 5.1, 4.875)

# ── Transformer Encoder Layer (repeated x2) ───────────────────────────────
rbox(ax, 5.1, 0.55, 3.1, 8.45, fc='#DDEEFF', ec='#1565C0', lw=1.8, zorder=3)
label(ax, 6.65, 8.78, 'Transformer Encoder Layer  ×2', fs=9.5, bold=True, color='#1565C0')

# LayerNorm 1
rbox(ax, 5.35, 7.75, 2.6, 0.65, fc='#FFF9C4', ec='#F9A825', zorder=5)
label(ax, 6.65, 8.08, 'Layer Norm 1', fs=9.5, bold=True, color='#7B6000')

arr(ax, 6.65, 7.75, 6.65, 7.3)

# Multi-Head Self-Attention block
rbox(ax, 5.35, 5.45, 2.6, 1.8, fc='#EDE7F6', ec='#6A1B9A', lw=1.8, zorder=5)
label(ax, 6.65, 7.02, 'Multi-Head Self-Attention', fs=9.5, bold=True, color='#4A148C')
label(ax, 6.65, 6.75, '4 heads', fs=8.5, color='#6A1B9A')

# Q K V boxes
for i, lbl in enumerate(['Q', 'K', 'V']):
    bx = 5.55 + i * 0.88
    rbox(ax, bx, 5.6, 0.72, 0.6, fc='#D1C4E9', ec='#6A1B9A', lw=1.2, zorder=6)
    label(ax, bx + 0.36, 5.9, lbl, fs=10, bold=True, color='#4A148C', zorder=7)

label(ax, 6.65, 5.52, 'softmax( QKᵀ / √d_k ) · V', fs=8, color='#555555', zorder=6)

arr(ax, 6.65, 5.45, 6.65, 5.05)

# Residual Add 1
rbox(ax, 5.35, 4.55, 2.6, 0.5, fc='#FFF9C4', ec='#F9A825', zorder=5)
label(ax, 6.65, 4.8, '⊕  Add & Norm  (Residual)', fs=8.5, color='#7B6000')

arr(ax, 6.65, 4.55, 6.65, 4.1)

# LayerNorm 2
rbox(ax, 5.35, 3.6, 2.6, 0.5, fc='#FFF9C4', ec='#F9A825', zorder=5)
label(ax, 6.65, 3.85, 'Layer Norm 2', fs=9.5, bold=True, color='#7B6000')

arr(ax, 6.65, 3.6, 6.65, 3.15)

# FFN block
rbox(ax, 5.35, 1.75, 2.6, 1.35, fc='#E0F2F1', ec='#00695C', lw=1.8, zorder=5)
label(ax, 6.65, 2.82, 'Feed-Forward Network', fs=9.5, bold=True, color='#004D40')
label(ax, 6.65, 2.52, 'Linear(H → 4H) → ReLU', fs=8.5, color='#333333')
label(ax, 6.65, 2.22, 'Linear(4H → H)', fs=8.5, color='#333333')
label(ax, 6.65, 1.92, 'Dropout', fs=8, color='#666666', italic=True)

arr(ax, 6.65, 1.75, 6.65, 1.35)

# Residual Add 2
rbox(ax, 5.35, 0.85, 2.6, 0.5, fc='#FFF9C4', ec='#F9A825', zorder=5)
label(ax, 6.65, 1.1, '⊕  Add & Norm  (Residual)', fs=8.5, color='#7B6000')

# residual bypass line (left side of encoder layer)
ax.plot([5.2, 5.2], [7.75+0.65, 4.55+0.25], color='#F9A825', lw=1.5,
        linestyle='--', zorder=4)
ax.annotate('', xy=(5.35, 4.8), xytext=(5.2, 4.8),
            arrowprops=dict(arrowstyle='->', color='#F9A825', lw=1.5))

ax.plot([5.2, 5.2], [3.6+0.5, 0.85+0.25], color='#F9A825', lw=1.5,
        linestyle='--', zorder=4)
ax.annotate('', xy=(5.35, 1.1), xytext=(5.2, 1.1),
            arrowprops=dict(arrowstyle='->', color='#F9A825', lw=1.5))

# ── E(x) output arrow ─────────────────────────────────────────────────────
arr(ax, 8.2, 1.1, 8.75, 1.1, color='#1565C0', lw=2.2)
label(ax, 8.47, 1.42, 'E(x)', fs=10, bold=True, color='#1565C0')

# ══════════════════════════════════════════════════════════════════════════
#  RIGHT  –  LNN CORE BLOCK
# ══════════════════════════════════════════════════════════════════════════
rbox(ax, 8.7, 0.35, 6.9, 9.4, fc='#FFF3E0', ec='#E65100', lw=2.2, zorder=2)
label(ax, 12.15, 9.52, 'LNN CORE BLOCK', fs=11, bold=True, color='#E65100')
label(ax, 12.15, 9.22, 'Continuous-Time Temporal Dynamics', fs=9, color='#E65100', italic=True)

# ── LNN ODE box ───────────────────────────────────────────────────────────
rbox(ax, 8.95, 3.5, 4.1, 4.1, fc='#FFE0B2', ec='#E65100', lw=2, zorder=4)
label(ax, 11.0, 7.3, 'LNN', fs=15, bold=True, color='#BF360C')
label(ax, 11.0, 6.9, 'Continuous-Time Dynamics', fs=9.5, color='#333333')

label(ax, 11.0, 6.45, 'dh/dt  =  − h/τ  +  f( E(x) + Uh )', fs=10,
      color='#000000', bold=True)

label(ax, 11.0, 5.95, '─────────────────────────────', fs=8, color='#BBBBBB')

label(ax, 11.0, 5.6, 'Euler discretisation:', fs=8.5, color='#555555', italic=True)
label(ax, 11.0, 5.25, 'h[t] = h[t−1] + Δt · f( h[t−1], x[t] )', fs=9,
      color='#333333')
label(ax, 11.0, 4.9, 'Sequential update   t = 0 → 99', fs=8.5, color='#666666',
      italic=True)

label(ax, 11.0, 4.4, '─────────────────────────────', fs=8, color='#BBBBBB')

rbox(ax, 9.25, 3.65, 1.5, 0.55, fc='#FFCCBC', ec='#E65100', lw=1.2, zorder=5)
label(ax, 10.0, 3.93, 'τ  (Time Constant)', fs=8.5, color='#BF360C')
ax.annotate('', xy=(9.25+0.75, 3.65+0.55), xytext=(9.25+0.75, 3.65+0.55+0.3),
            arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.3))

# ── Hidden state feedback loop ────────────────────────────────────────────
# New hidden state box
rbox(ax, 8.95, 8.05, 2.2, 0.8, fc='#E8F5E9', ec='#2E7D32', zorder=5)
label(ax, 10.05, 8.55, 'New Hidden State', fs=9, bold=True)
label(ax, 10.05, 8.2, 'h(t + Δt)', fs=9, color='#1B5E20')

# Arrow: LNN top → new hidden state
arr(ax, 10.05, 7.6, 10.05, 8.05, color='#2E7D32', lw=2)

# Hidden state box
rbox(ax, 12.2, 8.05, 2.2, 0.8, fc='#E8F5E9', ec='#2E7D32', zorder=5)
label(ax, 13.3, 8.55, 'Hidden State', fs=9, bold=True)
label(ax, 13.3, 8.2, 'h(t)', fs=9, color='#1B5E20')

# Arrow: new hidden → hidden (feedback)
arr(ax, 11.15, 8.45, 12.2, 8.45, color='#2E7D32', lw=1.8)
label(ax, 11.68, 8.7, 'feedback', fs=8, color='#555555', italic=True)

# Recurrent projection box
rbox(ax, 12.0, 6.6, 2.6, 0.75, fc='#E8F5E9', ec='#2E7D32', zorder=5)
label(ax, 13.3, 7.1, 'Recurrent Projection', fs=9, bold=True)
label(ax, 13.3, 6.8, 'U · h(t)', fs=9, color='#1B5E20')

# Arrow: hidden state down → recurrent projection
arr(ax, 13.3, 8.05, 13.3, 7.35, color='#2E7D32', lw=1.8)

# Arrow: recurrent projection → LNN
arr(ax, 12.0, 6.97, 11.05, 6.0, color='#2E7D32', lw=1.8)

# E(x) arrow into LNN
arr(ax, 8.75, 4.3, 8.95, 4.3, color='#1565C0', lw=2)
label(ax, 8.85, 4.6, 'E(x)', fs=9, bold=True, color='#1565C0')

# ── Output ────────────────────────────────────────────────────────────────
# Arrow from LNN bottom to output layer
arr(ax, 11.0, 3.5, 11.0, 2.95, color='#E65100', lw=2)

# Output layer
rbox(ax, 9.3, 2.35, 3.4, 0.6, fc='#FBE9E7', ec='#E65100', lw=1.8, zorder=5)
label(ax, 11.0, 2.65, 'Output Layer   Linear( H → 1 )', fs=9.5, bold=True,
      color='#BF360C')

# Arrows to two heads
arr(ax, 10.3, 2.35, 10.0, 1.7, color='#2E7D32', lw=1.8)
arr(ax, 11.7, 2.35, 12.0, 1.7, color='#283593', lw=1.8)

# Regression head
rbox(ax, 8.9, 0.55, 2.3, 1.15, fc='#E8F5E9', ec='#2E7D32', lw=1.8, zorder=5)
label(ax, 10.05, 1.42, 'REGRESSION HEAD', fs=9, bold=True, color='#1B5E20')
label(ax, 10.05, 1.13, 'Estimated Power (Watts)', fs=8.5, color='#333333')
label(ax, 10.05, 0.75, 'MAE  /  SAE', fs=9, bold=True, color='#1B5E20')

# Classification head
rbox(ax, 11.5, 0.55, 2.3, 1.15, fc='#E8EAF6', ec='#283593', lw=1.8, zorder=5)
label(ax, 12.65, 1.42, 'CLASSIFICATION HEAD', fs=9, bold=True, color='#1A237E')
label(ax, 12.65, 1.13, 'Appliance State  ON / OFF', fs=8.5, color='#333333')
label(ax, 12.65, 0.75, 'F1-SCORE', fs=9, bold=True, color='#1A237E')

# ── Caption ───────────────────────────────────────────────────────────────
cap = ('Diagram of the Transformer Encoder + LNN Architecture for NILM, integrating Multi-Head Self-Attention for '
       'global feature extraction with a Liquid Neural Network\n'
       '(LNN) for continuous-time temporal dynamics. '
       'Outputs: F1-SCORE (appliance state classification) and MAE/SAE (power regression).')
ax.text(8, 0.13, cap, ha='center', va='center', fontsize=7.8,
        color='#333333', style='italic', zorder=5)

plt.tight_layout(pad=0.3)
plt.savefig(r'c:\Users\MathK\OneDrive\Desktop\PhD_Slides\LNN\transformer_lnn_diagram.png',
            dpi=150, bbox_inches='tight', facecolor='#FFFBF0')
print('Saved!')
