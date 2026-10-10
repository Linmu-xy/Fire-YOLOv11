"""Plot stored dataset measurements and historical validation observations."""
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--artifacts',type=Path,default=Path(__file__).resolve().parent)
root=p.parse_args().artifacts
data=json.loads((root/'module_dataset_sizes_cmj.json').read_text())
data.update(json.loads((root/'module_dataset_sizes_lxy.json').read_text()))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
labels=['D-Fire','FASDD old','FASDD clean','FS-YOLO']
ids=['dfire','fasdd','fasdd_cv_clean_v1','fsyolo'];x=np.arange(4)
for offset,c,color in [(-.18,'fire','#ea580c'),(.18,'smoke','#64748b')]:
    values=[]
    for ds in ids:
        s=data[ds]['train']['classes'][c]
        values.append(100*sum(s['long_buckets_0_8_16_32_64_inf'][:2])/s['boxes'])
    bars=axes[0].bar(x+offset,values,.35,label=c.capitalize(),color=color)
    for bar,v in zip(bars,values):axes[0].text(bar.get_x()+bar.get_width()/2,v+.25,f'{v:.2f}',ha='center',fontsize=8)
axes[0].set(title='Training boxes with resized long side <16px',ylabel='Fraction of class boxes (%)',xticks=x,xticklabels=labels,ylim=(0,14))
axes[0].legend(frameon=False)
v=[.678,.401,.680]
bars=axes[1].bar(['Seed 0','Seed 1','Seed 2'],v,color='#2563eb',width=.55)
for b,value in zip(bars,v):axes[1].text(b.get_x()+b.get_width()/2,value+.02,f'+{value:.3f}',ha='center')
axes[1].set(title='P2 minus baseline: best validation mAP50-95',ylabel='Change (percentage points)',ylim=(0,.9))
v=[-.286,-.224,.468,-.530]
bars=axes[2].bar(['SRDG s0','DCBR s0','DCBR s2','ACR s0'],v,color=['#dc2626','#dc2626','#16a34a','#dc2626'],width=.6)
for b,value in zip(bars,v):axes[2].text(b.get_x()+b.get_width()/2,value+(.025 if value>0 else -.055),f'{value:+.3f}',ha='center',fontsize=9)
axes[2].set(title='Each module minus parent P2: best val',ylabel='Change (percentage points)',ylim=(-.7,.65))
axes[2].axhline(0,color='#475569',lw=.8)
for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('P2 and independent module diagnosis | static whole-image resize to max side 640',fontsize=13)
fig.savefig(root/'p2_module_diagnosis.png',dpi=170)
plt.close(fig)
