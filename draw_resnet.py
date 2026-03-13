import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 16); ax.set_ylim(0, 11); ax.axis('off')
fig.patch.set_facecolor('#FFFBF0'); ax.set_facecolor('#FFFBF0')

C_CONV = '#4477BB'
C_BN   = '#DD8822'
C_RELU = '#44AA55'
C_SKIP = '#CC3333'
C_POOL = '#7744AA'
C_STEM = '#2299AA'
C_FC   = '#777777'
C_BG   = '#EEF4FF'

def rbox(x,y,w,h,fc,ec,lw=1.6,zo=4,alpha=1.0):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12',
        linewidth=lw,edgecolor=ec,facecolor=fc,zorder=zo,alpha=alpha))

def circ(x,y,r,fc,ec,lw=1.8,zo=5):
    ax.add_patch(Circle((x,y),r,linewidth=lw,edgecolor=ec,facecolor=fc,zorder=zo))

def txt(x,y,s,fs=9,c='black',bold=False,ha='center',va='center',zo=7):
    ax.text(x,y,s,ha=ha,va=va,fontsize=fs,color=c,
            fontweight='bold' if bold else 'normal',zorder=zo)

def arr(x1,y1,x2,y2,col='#333',lw=1.6,zo=6):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
        arrowprops=dict(arrowstyle='->',color=col,lw=lw,mutation_scale=13),zorder=zo)

def seg(x1,y1,x2,y2,col='#333',lw=1.5,zo=5,ls='-'):
    ax.plot([x1,x2],[y1,y2],color=col,lw=lw,zorder=zo,ls=ls)

# ── Title ──────────────────────────────────────────────────────────
txt(8, 10.6, '1D ResNet — Residual Block Architecture', fs=15, c='#222', bold=True)
txt(8, 10.15, r'$\mathbf{y} = \mathcal{F}(\mathbf{x},\{W_i\}) + \mathbf{x}$'
             r'     layers=[2,2,2],  base_width=32',
    fs=9.5, c='#555')

# ══════════════════════════════════════════════════════════════════
# LEFT: Full network  (x = 0.4 .. 4.4)
# ══════════════════════════════════════════════════════════════════
txt(2.3, 9.65, 'Full Network', fs=11, c='#333', bold=True)

# Input
rbox(0.5, 8.95, 3.5, 0.58, fc='#E8E8E8', ec='#888', lw=1.5, zo=4)
txt(2.25, 9.24, r'Input  $x_t \in \mathbb{R}^{L \times 1}$', fs=9, c='#333')

net_blocks = [
    (C_STEM, 'Stem\nConv1d k=7,s=2 + BN + ReLU + MaxPool', 8.1),
    (C_CONV, 'Layer 1:  2x ResBlock\n32 channels,  stride=1',  7.0),
    (C_CONV, 'Layer 2:  2x ResBlock\n64 channels,  stride=2',  5.9),
    (C_CONV, 'Layer 3:  2x ResBlock\n128 channels,  stride=2', 4.8),
    (C_POOL, 'Adaptive AvgPool1d(1)',                          3.7),
    (C_FC,   'FC(128 -> 1)',                                    2.7),
]

prev_cy = 9.24
for (bc, bl, by) in net_blocks:
    rbox(0.5, by-0.38, 3.5, 0.76, fc=bc, ec='white', lw=1.3, zo=4)
    txt(2.25, by, bl, fs=8, c='white', bold=True)
    arr(2.25, prev_cy-0.01, 2.25, by+0.38, col='#555', lw=1.5)
    prev_cy = by - 0.38

# ══════════════════════════════════════════════════════════════════
# RIGHT: Residual block detail  (x = 4.8 .. 15.5)
# ══════════════════════════════════════════════════════════════════
rbox(4.8, 1.5, 10.5, 8.25, fc=C_BG, ec='#8888BB', lw=2.0, zo=2, alpha=0.4)
txt(10.05, 9.55, 'Residual Block Detail  (single ResidualBlock)', fs=11, c='#333', bold=True)

MX = 9.2
SX = 13.5

layers = [
    (8.6,  C_CONV, 'Conv1d  (kernel=5, stride=s)', r'$C_{in} \to C_{out}$'),
    (7.4,  C_BN,   'BatchNorm1d', ''),
    (6.5,  C_RELU, 'ReLU', ''),
    (5.5,  C_CONV, 'Conv1d  (kernel=5, stride=1)', r'$C_{out} \to C_{out}$'),
    (4.4,  C_BN,   'BatchNorm1d', ''),
]
ADD_Y   = 3.2
RELU2_Y = 2.3

for (ly, lc, ln, lsub) in layers:
    rbox(MX-1.6, ly-0.36, 3.2, 0.72, fc=lc, ec='white', lw=1.4, zo=5)
    txt(MX, ly+0.08, ln, fs=9, c='white', bold=True)
    if lsub:
        txt(MX, ly-0.17, lsub, fs=8, c='white')

# Final ReLU
rbox(MX-1.6, RELU2_Y-0.36, 3.2, 0.72, fc=C_RELU, ec='white', lw=1.4, zo=5)
txt(MX, RELU2_Y+0.08, 'ReLU  (after add)', fs=9, c='white', bold=True)

# Main path arrows
ys = [l[0] for l in layers]
for i in range(len(ys)-1):
    arr(MX, ys[i]-0.36, MX, ys[i+1]+0.36, col='#555', lw=1.6)
arr(MX, ys[-1]-0.36, MX, ADD_Y+0.28,  col='#555', lw=1.6)
arr(MX, ADD_Y-0.28,  MX, RELU2_Y+0.36, col='#555', lw=1.6)

# Input x
arr(MX, 9.62, MX, ys[0]+0.36, col='#555', lw=1.8)
txt(MX, 9.78, r'$\mathbf{x}$', fs=12, c='#333', bold=True)

# ⊕ circle
circ(MX, ADD_Y, 0.28, fc='white', ec=C_SKIP, lw=2.2, zo=6)
txt(MX, ADD_Y, r'$\oplus$', fs=12, c=C_SKIP, bold=True, zo=7)

# ── Skip connection ───────────────────────────────────────────────
seg(MX, 9.62, SX, 9.62, col=C_SKIP, lw=2.0)
seg(SX, 9.62, SX, ADD_Y, col=C_SKIP, lw=2.0)
arr(SX, ADD_Y, MX+0.28, ADD_Y, col=C_SKIP, lw=2.0)

# Skip path: Conv1d 1x1 + BN (when dims differ)
rbox(SX-0.85, 6.35, 1.7, 0.9, fc=C_SKIP, ec='white', lw=1.3, zo=5)
txt(SX, 6.75, 'Conv1d 1x1, stride=s', fs=7.5, c='white', bold=True)
txt(SX, 6.48, '+ BatchNorm1d', fs=7.5, c='white')
txt(SX, 6.22, '(if dims differ)', fs=7, c='#FFCCCC')

txt(SX+0.1, 8.3, 'Skip\nConnection', fs=9, c=C_SKIP, bold=True, ha='left')

# Output
arr(MX, RELU2_Y-0.36, MX, 1.72, col='#555', lw=1.8)
txt(MX, 1.62, r'$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$', fs=9, c='#333')

# stride note
txt(10.05, 1.88, r'stride $s=1$ (same dims) or $s=2$ (downsamples)',
    fs=8, c='#666')

# ── Legend ────────────────────────────────────────────────────────
items = [
    (C_STEM, r'Stem — initial Conv1d k=7'),
    (C_CONV, r'Conv1d — kernel=5'),
    (C_BN,   r'BatchNorm1d'),
    (C_RELU, r'ReLU — activation'),
    (C_SKIP, r'Skip — identity or Conv1d 1x1+BN'),
    (C_POOL, r'Adaptive AvgPool'),
    (C_FC,   r'FC — output'),
]
for i,(lc,lt) in enumerate(items):
    lx = 0.3 + i*2.26
    rbox(lx, 0.28, 0.26, 0.26, fc=lc, ec='white', lw=0, zo=6)
    txt(lx+0.16, 0.41, lt, fs=7.5, c='#333', ha='left')

plt.tight_layout()
plt.savefig('resnet_diagram.png', dpi=180, bbox_inches='tight')
plt.close()
print('Saved: resnet_diagram.png')
