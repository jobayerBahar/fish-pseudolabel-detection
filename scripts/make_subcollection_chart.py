import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
envs=["Aldabra","Tanganyika","Protea","Pondoland"]
hum={"Aldabra":[.4374,.4843,.4796],"Tanganyika":[.6103,.6499,.6538],"Protea":[.6614,.6837,.6643],"Pondoland":[.7081,.7353,.7340]}
gd={"Aldabra":[.2242,.2244,.2211,.2240,.2445,.2425],"Tanganyika":[.3977,.3753,.4179,.3957,.3978,.3708],
    "Protea":[.4948,.4803,.4933,.4894,.5040,.5005],"Pondoland":[.5271,.5598,.5273,.5665,.5338,.5717]}
def ret(e):
    h=hum[e]; g=gd[e]
    return np.mean([g[0]/h[0],g[1]/h[0],g[2]/h[1],g[3]/h[1],g[4]/h[2],g[5]/h[2]])*100
ratio={"Aldabra":(70+66)/2,"Tanganyika":(75+76)/2,"Protea":75,"Pondoland":(86+87)/2}
BLUE,RED,INK="#4a78c2","#d62728","#222222"
fig,(b,a)=plt.subplots(1,2,figsize=(7.8,3.2),gridspec_kw={"width_ratios":[1,1.5]})

# left: scatter
for e in envs:
    r=ret(e)
    b.scatter(ratio[e],r,s=55,color=INK,zorder=3,edgecolor="white",linewidth=1.5)
    dy=-4.2 if e=="Protea" else 2.2
    b.text(ratio[e]+1.0,r+dy,e,fontsize=8.5,color=INK)
b.set_xlabel("Pseudo-label coverage (%)"); b.set_ylabel("Retention (%)")
b.set_xlim(63,92); b.set_ylim(40,85)
b.grid(True,color="#e5e5e5",zorder=0); b.set_axisbelow(True)
b.set_title("(a) Coverage and retention",fontsize=9.5,loc="left",pad=8)

# right: bars
w=0.36
for i,e in enumerate(envs):
    for vals,off,col in [(hum[e],-w/2-0.01,BLUE),(gd[e],w/2+0.01,RED)]:
        m=np.mean(vals)
        a.bar(i+off,m,w,color=col,zorder=2)
        a.plot([i+off]*2,[min(vals),max(vals)],color=INK,lw=1,zorder=3)
        a.text(i+off,max(vals)+0.015,f"{m:.2f}",ha="center",va="bottom",fontsize=7.5,color=INK)
    a.text(i,0.9,f"{ret(e):.0f}%",ha="center",va="bottom",fontsize=9,color=INK,fontweight="bold")
a.set_xticks(range(4)); a.set_xticklabels(envs)
a.set_ylabel("mAP$_{50:95}$ (human test labels)"); a.set_ylim(0,0.98)
a.set_yticks([0,0.2,0.4,0.6,0.8])
a.yaxis.grid(True,color="#e5e5e5",zorder=0); a.set_axisbelow(True)
handles=[Patch(color=BLUE,label="Human-trained"),Patch(color=RED,label="GD-trained"),
         Line2D([],[],color=INK,marker="$\\%$",linestyle="None",markersize=9,label="Retention")]
a.legend(handles=handles,frameon=False,fontsize=8,loc="upper center",ncol=3,bbox_to_anchor=(0.5,-0.13),handlelength=1.2,columnspacing=1.0)
a.set_title("(b) mAP per sub-collection",fontsize=9.5,loc="left",pad=8)
fig.tight_layout()
fig.savefig("envfig/subcollection_chart.pdf"); fig.savefig("envfig/subcollection_chart.png",dpi=130)
