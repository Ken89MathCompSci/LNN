
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 16); ax.set_ylim(0, 11); ax.axis('off')
fig.patch.set_facecolor('#FFFBF0'); ax.set_facecolor('#FFFBF0')

C_CONV  = '#4477BB'
C_RELU  = '#44AA55'
C_DROP  = '#CC7700'
C_POOL  = '#7744AA'
C_FC    = '#777777'
C_BG    = '#EEF4FF'
C_DIL   = '#CC3333'

def rbox(x,y,w,h,fc,ec,lw=1.6,zo=4,alpha=1.0):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12',
        linewidth=lw,edgecolor=ec,facecolor=fc,zorder=zo,alpha=alpha))

def txt(x,y,s,fs=9,c='black',bold=False,ha='center',va='center',zo=7):
    ax.text(x,y,s,ha=ha,va=va,fontsize=fs,color=c,
            fontweight='bold' if bold else 'normal',zorder=zo)

def arr(x1,y1,x2,y2,col='#333',lw=1.6,zo=6):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
        arrowprops=dict(arrowstyle='->',color=col,lw=lw,mutation_scale=13),zorder=zo)

def seg(x1,y1,x2,y2,col='#333',lw=1.5,zo=5,ls='-'):
    ax.plot([x1,x2],[y1,y2],color=col,lw=lw,zorder=zo,ls=ls)

# Title
txt(8, 10.6, 'Temporal Convolutional Network (TCN)', fs=15, c='#222', bold=True)
txt(8, 10.18, r'num_channels=[32,64,128],  kernel_size=3,  dilation=2$^i$,  dropout=0.2',
    fs=9.5, c='#555')

# LEFT: Full network
txt(2.3, 9.7, 'Full Network', fs=11, c='#333', bold=True)

rbox(0.4, 8.98, 3.7, 0.56, fc='#E8E8E8', ec='#888', lw=1.5, zo=4)
txt(2.25, 9.26, r'Input  $x_t \in \mathbb{R}^{L \times 1}$', fs=9, c='#333')

net_blocks = [
    (C_CONV, 'TCNBlock 1\nConv1d(1->32, k=3, d=1) + ReLU + Dropout', 8.1),
    (C_CONV, 'TCNBlock 2\nConv1d(32->64, k=3, d=2) + ReLU + Dropout', 6.85),
    (C_CONV, 'TCNBlock 3\nConv1d(64->128, k=3, d=4) + ReLU + Dropout', 5.6),
    (C_POOL, 'Global AvgPool\ntorch.mean(dim=2)', 4.35),
    (C_FC,   'FC(128 -> 1)', 3.2),
]

prev_cy = 9.26
for (bc, bl, by) in net_blocks:
    rbox(0.4, by-0.46, 3.7, 0.92, fc=bc, ec='white', lw=1.3, zo=4)
    txt(2.25, by, bl, fs=8, c='white', bold=True)
    arr(2.25, prev_cy-0.01, 2.25, by+0.46, col='#555', lw=1.5)
    prev_cy = by - 0.46

rbox(0.4, 2.38, 3.7, 0.62, fc='#F0F0F0', ec='#AAAAAA', lw=1.2, zo=4)
txt(2.25, 2.69, 'Receptive field = 1+(k-1)(d1+d2+d3) = 1+2(7) = 15 steps', fs=7.5, c='#444')

# RIGHT: TCNBlock detail + dilation
rbox(4.7, 1.5, 11.0, 8.25, fc=C_BG, ec='#8888BB', lw=2.0, zo=2, alpha=0.4)
txt(10.2, 9.55, 'TCNBlock Detail  (single block)', fs=11, c='#333', bold=True)

BX = 7.2

block_layers = [
    (8.3,  C_CONV, 'Conv1d  (kernel=3, dilation=d)',  r'$C_{in} \to C_{out}$,  padding=(k-1)*d//2'),
    (6.9,  C_RELU, 'ReLU', ''),
    (5.7,  C_DROP, 'Dropout  (p=0.2)', ''),
]

prev_by = 9.26
for (ly, lc, ln, lsub) in block_layers:
    rbox(BX-1.7, ly-0.38, 3.4, 0.76, fc=lc, ec='white', lw=1.4, zo=5)
    txt(BX, ly+0.08, ln, fs=9.5, c='white', bold=True)
    if lsub:
        txt(BX, ly-0.18, lsub, fs=7.5, c='white')
    arr(BX, prev_by-0.01, BX, ly+0.38, col='#555', lw=1.6)
    prev_by = ly - 0.38

arr(BX, prev_by, BX, 4.75, col='#555', lw=1.6)
txt(BX, 4.6, r'Output  $\in \mathbb{R}^{L \times C_{out}}$', fs=9, c='#333')
txt(BX, 9.42, r'Input  $\in \mathbb{R}^{L \times C_{in}}$', fs=9, c='#333')

# Dilation illustration
txt(12.7, 9.55, 'Dilation Illustration', fs=10, c='#333', bold=True)
txt(12.7, 9.2,  'kernel=3 — how receptive field grows', fs=8, c='#666')

DIL_CONFIGS = [
    (8.0, 1, C_CONV,   'Block 1: d=1'),
    (6.4, 2, '#2299AA','Block 2: d=2'),
    (4.8, 4, C_DIL,    'Block 3: d=4'),
]

for (dy, dil, dc, dlabel) in DIL_CONFIGS:
    xs = [10.7 + i*0.42 for i in range(7)]
    for xi in xs:
        ax.add_patch(plt.Circle((xi, dy), 0.14, fc=dc, ec='white', lw=1.2, zorder=5))
    centre = xs[3]
    for offset in [-dil, 0, dil]:
        src_x = centre + offset * 0.42
        if xs[0]-0.1 <= src_x <= xs[-1]+0.1:
            seg(src_x, dy+0.14, centre, dy+0.50, col=dc, lw=1.8, zo=4)
    ax.add_patch(plt.Circle((centre, dy+0.62), 0.17, fc=dc, ec='white', lw=1.5, zorder=6))
    txt(14.7, dy+0.2, dlabel, fs=8, c=dc, bold=True, ha='left')
    rf = 1 + (3-1)*dil
    txt(14.7, dy-0.05, 'RF=%d'%rf, fs=7.5, c='#555', ha='left')

txt(12.7, 3.8, 'Stacking blocks multiplies\nreceptive field coverage', fs=8, c='#555')

# Legend
items = [
    (C_CONV, 'Conv1d -- dilated'),
    (C_RELU, 'ReLU -- activation'),
    (C_DROP, 'Dropout -- regularisation'),
    (C_POOL, 'Global AvgPool'),
    (C_FC,   'FC -- output'),
]
for i,(lc,lt) in enumerate(items):
    lx = 0.3 + i*3.1
    ax.add_patch(FancyBboxPatch((lx, 0.25), 0.28, 0.28,
        boxstyle='round,pad=0.05', fc=lc, ec='white', lw=0, zorder=6))
    txt(lx+0.18, 0.39, lt, fs=7.5, c='#333', ha='left')

plt.tight_layout()
plt.savefig('tcn_diagram.png', dpi=180, bbox_inches='tight')
plt.close()
print('Saved: tcn_diagram.png')
