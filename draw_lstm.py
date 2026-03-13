import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 16); ax.set_ylim(0, 11); ax.axis("off")
fig.patch.set_facecolor("#FFFBF0"); ax.set_facecolor("#FFFBF0")

C_FORGET="#E07070"; C_INPUT="#70A0E0"; C_CELL="#70C070"; C_OUTPUT="#E0A030"
C_STATE="#9966CC"; C_HIDDEN="#444444"; C_BG="#EEF2FF"

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

txt(8,10.65,r"Long Short-Term Memory (LSTM) Cell",fs=15,c="#222",bold=True)
txt(8,10.2,r"$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$          $h_t = o_t \odot \tanh(C_t)$",fs=10,c="#555")

rbox(0.5,1.8,15.0,8.0,fc=C_BG,ec="#8888BB",lw=2.0,zo=2,alpha=0.4)
txt(8.0,2.1,"LSTM Cell",fs=8.5,c="#8888BB")

CY=9.2; GY=6.8; BUS=4.5

arr(0.0,CY,1.2,CY,col=C_STATE,lw=2.4)
txt(0.55,CY+0.32,r"$C_{t-1}$",fs=10,c=C_STATE,bold=True)
arr(14.1,CY,15.8,CY,col=C_STATE,lw=2.4)
txt(15.1,CY+0.32,r"$C_t$",fs=10,c=C_STATE,bold=True)

GX=[3.2,6.0,8.8,11.6]; OPX=[4.4,7.2,13.0]

for ox,sym,oc in [(OPX[0],"x",C_FORGET),(OPX[1],"+",C_INPUT),(OPX[2],"x",C_OUTPUT)]:
    circ(ox,CY,0.34,fc="white",ec=oc,lw=2.2)
    txt(ox,CY,sym,fs=14,c=oc,bold=True)

seg(1.2,CY,OPX[0]-0.34,CY,col=C_STATE,lw=2.4)
seg(OPX[0]+0.34,CY,OPX[1]-0.34,CY,col=C_STATE,lw=2.4)
seg(OPX[1]+0.34,CY,10.6,CY,col=C_STATE,lw=2.4)
rbox(10.6,CY-0.38,1.0,0.76,fc="#F5E09A",ec="#AA8800",lw=1.6,zo=5)
txt(11.1,CY,r"$\tanh$",fs=9.5,c="#664400",bold=True)
seg(11.6,CY,OPX[2]-0.34,CY,col=C_STATE,lw=2.4)

gates=[(C_FORGET,"Forget Gate",r"$\sigma$",r"$f_t$"),(C_INPUT,"Input Gate",r"$\sigma$",r"$i_t$"),(C_CELL,"Cell Gate",r"$\tanh$",r"$\tilde{C}_t$"),(C_OUTPUT,"Output Gate",r"$\sigma$",r"$o_t$")]
for j,(gc,glabel,gfunc,gout) in enumerate(gates):
    x=GX[j]
    txt(x,GY+1.05,glabel,fs=8.5,c=gc,bold=True)
    rbox(x-0.46,GY-0.38,0.92,0.76,fc=gc,ec="white",lw=1.5,zo=5)
    txt(x,GY,gfunc,fs=11,c="white",bold=True,zo=6)
    txt(x+0.62,GY+0.08,gout,fs=9,c=gc,ha="left")

arr(GX[0],GY+0.38,OPX[0],CY-0.34,col=C_FORGET,lw=1.8)

MX=(GX[1]+GX[2])/2; MY=7.9
circ(MX,MY,0.28,fc="white",ec="#5566AA",lw=1.9)
txt(MX,MY,"x",fs=12,c="#5566AA",bold=True)
arr(GX[1],GY+0.38,MX-0.15,MY-0.15,col=C_INPUT,lw=1.7)
arr(GX[2],GY+0.38,MX+0.15,MY-0.15,col=C_CELL,lw=1.7)
arr(MX,MY+0.28,OPX[1],CY-0.34,col="#5566AA",lw=1.8)

arr(GX[3],GY+0.38,OPX[2],CY-0.34,col=C_OUTPUT,lw=1.8)

arr(0.0,BUS,1.5,BUS,col=C_HIDDEN,lw=2.2)
txt(0.65,BUS+0.3,r"$h_{t-1}$",fs=10,c=C_HIDDEN,bold=True)
arr(8.0,1.1,8.0,2.7,col="#336633",lw=2.2)
txt(7.4,0.95,r"$x_t$",fs=11,c="#336633",bold=True)

rbox(1.5,BUS-0.3,5.2,0.6,fc="#E0E0E0",ec="#999",lw=1.3,zo=4)
txt(4.1,BUS,r"$[h_{t-1},\, x_t]$  concatenated",fs=9,c="#444")

seg(6.7,BUS,14.5,BUS,col="#888",lw=1.3)
for x in GX:
    seg(x,BUS,x,GY-0.38,col="#888",lw=1.4,ls="--")
seg(8.0,2.7,8.0,BUS-0.3,col="#336633",lw=1.4,ls="--")

arr(OPX[2]+0.34,CY,15.4,CY,col=C_HIDDEN,lw=2.2)
txt(15.4,CY+0.32,r"$h_t$",fs=10,c=C_HIDDEN,bold=True)
seg(15.4,CY,15.4,BUS,col=C_HIDDEN,lw=1.6,ls="--")
arr(15.4,BUS,14.5,BUS,col=C_HIDDEN,lw=1.8)
txt(15.85,6.8,r"$h_t \to h_{t-1}$",fs=8,c=C_HIDDEN,ha="left")

items=[(C_FORGET,r"Forget gate  $f_t=\sigma(W_f[h_{t-1},x_t]+b_f)$"),(C_INPUT,r"Input gate   $i_t=\sigma(W_i[h_{t-1},x_t]+b_i)$"),(C_CELL,r"Cell gate    $\tilde{C}_t=\tanh(W_C[h_{t-1},x_t]+b_C)$"),(C_OUTPUT,r"Output gate  $o_t=\sigma(W_o[h_{t-1},x_t]+b_o)$"),(C_STATE,r"Cell state   $C_t=f_t\odot C_{t-1}+i_t\odot\tilde{C}_t$")]
for i,(lc,lt) in enumerate(items):
    lx=0.4+i*3.1
    rbox(lx,0.2,0.28,0.28,fc=lc,ec="white",lw=0,zo=6)
    txt(lx+0.2,0.34,lt,fs=7.6,c="#333",ha="left")

plt.tight_layout()
plt.savefig("lstm_diagram.png",dpi=180,bbox_inches="tight")
plt.close()
print("Saved: lstm_diagram.png")
