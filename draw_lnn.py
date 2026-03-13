import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16); ax.set_ylim(0, 10); ax.axis('off')
fig.patch.set_facecolor('white'); ax.set_facecolor('white')

C_TAU   = '#2277CC'
C_DRIVE = '#33AA33'
C_GATE  = '#CC3333'
C_ODE   = '#CC7700'
C_IN    = '#F5F5F5'

def neuron(x, y, label, fc='white', ec='black', r=0.38, fs=9, lw=1.8, bold=False):
    ax.add_patch(Circle((x,y), r, fc=fc, ec=ec, lw=lw, zorder=5))
    ax.text(x, y, label, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', zorder=6)

def conn(x1,y1,x2,y2,col='#AAAAAA',lw=0.8,r=0.38,label='',lfs=7.5,rad=0,zo=3):
    dx=x2-x1; dy=y2-y1; d=np.hypot(dx,dy)
    ux,uy = dx/d,dy/d
    sx,sy = x1+ux*r, y1+uy*r
    ex,ey = x2-ux*r, y2-uy*r
    style = 'arc3,rad=%.2f'%rad
    ax.annotate('',xy=(ex,ey),xytext=(sx,sy),
                arrowprops=dict(arrowstyle='->',color=col,lw=lw,mutation_scale=9,
                                connectionstyle=style),zorder=zo)
    if label:
        mx=(sx+ex)/2; my=(sy+ey)/2
        ax.text(mx,my,label,fontsize=lfs,color=col,ha='center',va='center',
                bbox=dict(boxstyle='round,pad=0.1',fc='white',ec='none',alpha=0.92),zorder=8)

X1 = 1.8; X2 = 6.0; X3 = 10.5; X4 = 13.8

NX = 5; NH = 3
XY = [8.5 - i*1.1 for i in range(NX)]
HY = [3.5 - i*1.0 for i in range(NH)]
YTAU=7.8; YFT=5.5; YGT=3.2; YODE=5.5; YOUT=5.5

ax.text(8, 9.65, 'Advanced Liquid Neural Network (Advanced LNN) Cell',
        ha='center', fontsize=13, fontweight='bold')
ax.text(X1, 9.2, 'Input Layer',        ha='center', fontsize=10, fontweight='bold')
ax.text(X2, 9.2, 'Computation Layer',  ha='center', fontsize=10, fontweight='bold')
ax.text(X3, 9.2, 'ODE',                ha='center', fontsize=10, fontweight='bold')
ax.text(X4, 9.2, 'Output',             ha='center', fontsize=10, fontweight='bold')

# Group labels and divider
ax.text(X1-0.85, (XY[-1]+HY[0])/2+0.1, r'$x_t$',       ha='center', fontsize=9, color='#555')
ax.text(X1-0.85, (HY[0]+HY[-1])/2,     r'$h_{t-1}$',   ha='center', fontsize=9, color='#555')
ax.plot([X1-0.5,X1-0.5],[HY[0]+0.55,XY[-1]-0.55],color='#BBBBBB',lw=1,ls='--',zorder=2)

# Input neurons
for i,y in enumerate(XY):
    neuron(X1, y, r'$x_{%d}$'%(i+1), fc=C_IN, r=0.38, fs=9)
for i,y in enumerate(HY):
    neuron(X1, y, r'$h_{%d}$'%(i+1), fc='#E8E8FF', ec='#6666AA', r=0.38, fs=9)

# Thin background connections
for xy in XY:
    conn(X1,xy, X2,YTAU, col='#AACCEE', lw=0.9)
    conn(X1,xy, X2,YFT,  col='#AADDAA', lw=0.9)
    conn(X1,xy, X2,YGT,  col='#FFAAAA', lw=0.9)
for hy in HY:
    conn(X1,hy, X2,YFT,  col='#AADDAA', lw=0.9)
    conn(X1,hy, X2,YGT,  col='#FFAAAA', lw=0.9)
    conn(X1,hy, X3,YODE, col='#CCCCCC', lw=0.7, rad=-0.15)

# Highlighted weight labels
conn(X1,XY[0], X2,YTAU, col=C_TAU,   lw=2.0, label=r'$W_{\tau}$', lfs=8, zo=7)
conn(X1,XY[2], X2,YFT,  col=C_DRIVE, lw=2.0, label=r'$W_{in}$',   lfs=8, zo=7)
conn(X1,XY[4], X2,YGT,  col=C_GATE,  lw=2.0, label=r'$W_g$', lfs=8, rad=0.05, zo=7)
conn(X1,HY[0], X2,YFT,  col=C_DRIVE, lw=2.0, label=r'$W_{rec}$',  lfs=8, zo=7)
conn(X1,HY[2], X2,YGT,  col=C_GATE,  lw=2.0, label=r'$W_g$', lfs=8, rad=-0.05, zo=7)

# Computation neurons
neuron(X2,YTAU, r'$\tau$',  fc='#DDEEFF', ec=C_TAU,   r=0.5, lw=2.2, fs=13, bold=True)
ax.text(X2,YTAU+0.72, r'$\mathrm{softplus}(\tau_b)\odot\sigma(W_{\tau}x_t)$',
        ha='center', fontsize=7.5, color=C_TAU)

neuron(X2,YFT,  r'$f_t$',  fc='#DDFFDD', ec=C_DRIVE,  r=0.5, lw=2.2, fs=13, bold=True)
ax.text(X2,YFT+0.72, r'$\tanh(W_{in}x_t + h_{t-1}W_{rec})$',
        ha='center', fontsize=7.5, color=C_DRIVE)

neuron(X2,YGT,  r'$g_t$',  fc='#FFDDDD', ec=C_GATE,   r=0.5, lw=2.2, fs=13, bold=True)
ax.text(X2,YGT-0.72, r'$\sigma(W_g[h_{t-1},\,x_t])$',
        ha='center', fontsize=7.5, color=C_GATE)

# Computation -> ODE
conn(X2,YTAU, X3,YODE, col=C_TAU,   lw=2.0, label=r'$-h/\tau$',    lfs=8, zo=7)
conn(X2,YFT,  X3,YODE, col=C_DRIVE, lw=2.0, zo=7)
conn(X2,YGT,  X3,YODE, col=C_GATE,  lw=2.0, label=r'$g{\odot}f$', lfs=8, rad=-0.12, zo=7)

# ODE node
neuron(X3,YODE, r'$dh/dt$', fc='#FFF5DD', ec=C_ODE, r=0.52, lw=2.2, fs=10, bold=True)
ax.text(X3,YODE+0.78, r'$= -h_{t-1}/\tau + g_t\odot f_t$',
        ha='center', fontsize=8, color=C_ODE)

# ODE -> output
conn(X3,YODE, X4,YOUT, col=C_ODE, lw=2.2, label=r'$\times\Delta t$', lfs=9, zo=7)

# Output neuron
neuron(X4,YOUT, r'$h_t$', fc='#EEFFEE', ec='#228822', r=0.5, lw=2.2, fs=13, bold=True)
ax.text(X4,YOUT+0.75, r'$h_{t-1}+\Delta t\cdot(dh/dt)$',
        ha='center', fontsize=8, color='#228822')
ax.text(X4,YOUT-0.75, r'clamp$(\pm10)$',
        ha='center', fontsize=7.5, color='#228822')

# h_{t-1} skip (Euler residual)
ax.annotate('',xy=(X4,YOUT-0.5),xytext=(X1,HY[-1]-0.38),
            arrowprops=dict(arrowstyle='->',color='#888888',lw=1.8,
                           connectionstyle='arc3,rad=-0.25',mutation_scale=12),zorder=4)
ax.text(8.0, 1.08, r'$h_{t-1}$ skip connection (Euler residual)',
        ha='center', fontsize=8.5, color='#666666', style='italic')

# Equation banner
ax.text(8, 0.38,
    r'$h_t = \mathrm{clamp}\!\left(h_{t-1} + \Delta t\!\left(\frac{-h_{t-1}}{\tau} + g_t \odot f_t\right),\;\pm10\right)$',
    ha='center', fontsize=10, color='#333333',
    bbox=dict(boxstyle='round,pad=0.4', fc='#F5F5FF', ec='#AAAACC', lw=1.2))

# Legend
items = [
    (C_TAU,   r'$\tau$: adaptive time constant'),
    (C_DRIVE, r'$f_t$: drive signal ($\tanh$)'),
    (C_GATE,  r'$g_t$: input gate ($\sigma$)'),
    (C_ODE,   r'ODE: $dh/dt = -h/\tau + g_t f_t$'),
]
for i,(lc,lt) in enumerate(items):
    lx = 0.5 + i*3.9
    ax.add_patch(Circle((lx+0.14, 0.88), 0.12, fc=lc, ec='none', zorder=6))
    ax.text(lx+0.32, 0.88, lt, fontsize=8, color='#333', ha='left', va='center')

plt.tight_layout()
plt.savefig('lnn_diagram.png', dpi=180, bbox_inches='tight')
plt.close()
print('Saved: lnn_diagram.png')
