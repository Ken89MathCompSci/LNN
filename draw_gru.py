import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 16); ax.set_ylim(0, 11); ax.axis("off")
fig.patch.set_facecolor("#FFFBF0"); ax.set_facecolor("#FFFBF0")

C_RESET ="#E07070"; C_UPDATE="#70A0E0"; C_CAND  ="#70C070"
C_STATE ="#9966CC"; C_HIDDEN="#444444"; C_BG    ="#EEF2FF"

def rbox(x,y,w,h,fc,ec,lw=1.6,zo=4,alpha=1.0):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.1",linewidth=lw,edgecolor=ec,facecolor=fc,zorder=zo,alpha=alpha))
def circ(x,y,r,fc,ec,lw=1.8,zo=5):
    ax.add_patch(Circle((x,y),r,linewidth=lw,edgecolor=ec,facecolor=fc,zorder=zo))
def txt(x,y,s,fs=9,c="black",bold=False,ha="center",va="center",zo=7):
    ax.text(x,y,s,ha=ha,va=va,fontsize=fs,color=c,fontweight="bold" if bold else "normal",zorder=zo)
def arr(x1,y1,x2,y2,col="#333",lw=1.6,zo=6):
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="->",color=col,lw=lw,mutation_scale=13),zorder=zo)
def seg(x1,y1,x2,y2,col="#333",lw=1.5,zo=5,ls="-"):
    ax.plot([x1,x2],[y1,y2],color=col,lw=lw,zorder=zo,ls=ls)

# Title
txt(8,10.65,r"Gated Recurrent Unit (GRU) Cell",fs=15,c="#222",bold=True)
txt(8,10.2,r"$h_t = (1-z_t)\odot h_{t-1} + z_t\odot\tilde{h}_t$          $\tilde{h}_t = \tanh(W[r_t\odot h_{t-1},\, x_t])$",fs=10,c="#555")

# Cell background
rbox(0.5,1.8,15.0,8.0,fc=C_BG,ec="#8888BB",lw=2.0,zo=2,alpha=0.4)
txt(8.0,2.1,"GRU Cell",fs=8.5,c="#8888BB")

CY  = 9.2   # hidden state highway
GY  = 6.5   # gate box centres
BUS = 4.2   # [h,x] input bus

# -- Hidden state highway -----------------------------------------
arr(0.0,CY,1.2,CY,col=C_STATE,lw=2.4)
txt(0.55,CY+0.32,r"$h_{t-1}$",fs=10,c=C_STATE,bold=True)
arr(14.2,CY,15.8,CY,col=C_STATE,lw=2.4)
txt(15.1,CY+0.32,r"$h_t$",fs=10,c=C_STATE,bold=True)

# Operations on highway: x(1-z) and +(blend)
OPX1 = 5.0   # x circle: (1-z)*h_{t-1}
OPX2 = 13.0  # + circle: blend

circ(OPX1,CY,0.34,fc="white",ec=C_UPDATE,lw=2.2)
txt(OPX1,CY,"x",fs=14,c=C_UPDATE,bold=True)
circ(OPX2,CY,0.34,fc="white",ec=C_UPDATE,lw=2.2)
txt(OPX2,CY,"+",fs=14,c=C_UPDATE,bold=True)

# Highway segments
seg(1.2,CY,OPX1-0.34,CY,col=C_STATE,lw=2.4)
seg(OPX1+0.34,CY,OPX2-0.34,CY,col=C_STATE,lw=2.4)
seg(OPX2+0.34,CY,14.2,CY,col=C_STATE,lw=2.4)

# (1-z) label on highway segment after x circle
rbox(5.8,CY-0.3,1.1,0.6,fc="#EEE",ec="#888",lw=1.1,zo=5)
txt(6.35,CY,r"$(1\!-\!z_t)$",fs=8,c="#444")

# -- 3 Gate / candidate boxes -------------------------------------
GX=[3.0, 7.5, 11.5]

gates=[(C_RESET, "Reset Gate",  r"$\sigma$",  r"$r_t$"),
       (C_UPDATE,"Update Gate", r"$\sigma$",  r"$z_t$"),
       (C_CAND,  "Candidate",   r"$\tanh$",  r"$\tilde{h}_t$")]

for j,(gc,glabel,gfunc,gout) in enumerate(gates):
    x=GX[j]
    txt(x,GY+1.05,glabel,fs=8.5,c=gc,bold=True)
    rbox(x-0.46,GY-0.38,0.92,0.76,fc=gc,ec="white",lw=1.5,zo=5)
    txt(x,GY,gfunc,fs=11,c="white",bold=True,zo=6)
    txt(x+0.62,GY+0.08,gout,fs=9,c=gc,ha="left")

# -- Reset path: r_t x h_{t-1} intermediate circle ----------------
RHX=2.2; RHY=5.2
circ(RHX,RHY,0.28,fc="white",ec=C_RESET,lw=1.9)
txt(RHX,RHY,"x",fs=12,c=C_RESET,bold=True)

# h_{t-1} taps down to r x h circle
seg(1.2,CY,1.2,RHY,col=C_STATE,lw=1.5,ls="--")
arr(1.2,RHY,RHX-0.28,RHY,col=C_STATE,lw=1.5)
txt(1.0,RHY+0.25,r"$h_{t-1}$",fs=7.5,c=C_STATE,ha="right")

# Reset gate ? r x h circle
arr(GX[0],GY-0.38,RHX,RHY+0.28,col=C_RESET,lw=1.7)

# r x h ? candidate tanh
arr(RHX+0.28,RHY,GX[2]-0.46,GY-0.1,col=C_RESET,lw=1.7)
txt(5.5,4.8,r"$r_t \odot h_{t-1}$",fs=8,c=C_RESET)

# -- Update gate connections ---------------------------------------
# Update gate ? (1-z) x circle on highway
arr(GX[1],GY+0.38,OPX1,CY-0.34,col=C_UPDATE,lw=1.8)
txt(5.8,8.15,r"$(1-z_t)$",fs=7.5,c=C_UPDATE)

# z x h~ intermediate circle
ZHX=12.0; ZHY=8.1
circ(ZHX,ZHY,0.28,fc="white",ec=C_UPDATE,lw=1.9)
txt(ZHX,ZHY,"x",fs=12,c=C_UPDATE,bold=True)

# Update gate ? z x h~ circle
arr(GX[1],GY+0.38,ZHX-0.2,ZHY-0.2,col=C_UPDATE,lw=1.7)
# Candidate ? z x h~ circle
arr(GX[2],GY+0.38,ZHX+0.1,ZHY-0.22,col=C_CAND,lw=1.7)
txt(ZHX+0.45,ZHY,r"$z_t\!\odot\!\tilde{h}_t$",fs=8,c=C_UPDATE,ha="left")
# z x h~ ? + blend circle
arr(ZHX,ZHY+0.28,OPX2,CY-0.34,col=C_UPDATE,lw=1.8)

# -- Inputs: [h_{t-1}, x_t] --------------------------------------
arr(0.0,BUS,1.5,BUS,col=C_HIDDEN,lw=2.2)
txt(0.65,BUS+0.3,r"$h_{t-1}$",fs=10,c=C_HIDDEN,bold=True)
arr(7.5,1.1,7.5,2.7,col="#336633",lw=2.2)
txt(6.9,0.95,r"$x_t$",fs=11,c="#336633",bold=True)

rbox(1.5,BUS-0.3,5.2,0.6,fc="#E0E0E0",ec="#999",lw=1.3,zo=4)
txt(4.1,BUS,r"$[h_{t-1},\, x_t]$  concatenated",fs=9,c="#444")

seg(6.7,BUS,14.5,BUS,col="#888",lw=1.3)
for x in GX:
    seg(x,BUS,x,GY-0.38,col="#888",lw=1.4,ls="--")
seg(7.5,2.7,7.5,BUS-0.3,col="#336633",lw=1.4,ls="--")

# -- h_t feedback loop --------------------------------------------
seg(14.2,CY,15.4,CY,col=C_STATE,lw=0)   # already drawn by highway arr
seg(15.4,CY,15.4,BUS,col=C_HIDDEN,lw=1.6,ls="--")
arr(15.4,BUS,14.5,BUS,col=C_HIDDEN,lw=1.8)
txt(15.82,6.8,r"$h_t \to h_{t-1}$",fs=8,c=C_HIDDEN,ha="left")

# -- Legend -------------------------------------------------------
items=[(C_RESET, r"Reset gate    $r_t=\sigma(W_r[h_{t-1},x_t])$"),
       (C_UPDATE,r"Update gate   $z_t=\sigma(W_z[h_{t-1},x_t])$"),
       (C_CAND,  r"Candidate     $\tilde{h}_t=\tanh(W[r_t\odot h_{t-1},x_t])$"),
       (C_STATE, r"Output        $h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde{h}_t$")]
for i,(lc,lt) in enumerate(items):
    lx=0.4+i*3.9
    rbox(lx,0.2,0.28,0.28,fc=lc,ec="white",lw=0,zo=6)
    txt(lx+0.2,0.34,lt,fs=7.6,c="#333",ha="left")

plt.tight_layout()
plt.savefig("gru_diagram.png",dpi=180,bbox_inches="tight")
plt.close()
print("Saved: gru_diagram.png")
