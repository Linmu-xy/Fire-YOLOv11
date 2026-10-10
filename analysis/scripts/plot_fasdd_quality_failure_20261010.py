#!/usr/bin/env python3
"""Render audited quality failure diagnostics from saved JSON, no inference."""
from pathlib import Path
import json
import os

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'analysis/artifacts/fasdd_quality_failure_20261010_v2'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT.parent/'Fire-YOLOv11-runtime/quality_failure_plot/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

d=json.loads((DATA/'summary.json').read_text())
fig,ax=plt.subplots(1,3,figsize=(14.8,4.5),constrained_layout=True)
variants=['product_pq_gamma025','product_pq_gamma050','product_pq','quality_only_q','GT_required_TP50_reorder_by_q']
labels=['p*q^0.25','p*q^0.50','p*q','q alone','TP50 only: rank by q*']
gains=[d['fixed_candidate_AP'][name]['gain_pp'] for name in variants]
ax[0].barh(labels,gains,color=['#14856b' if v>0 else '#c65350' for v in gains])
ax[0].axvline(0,color='#555',lw=1)
ax[0].invert_yaxis()
ax[0].set(xlim=(-4.05,.6),xlabel='mAP50-95 change (percentage points)',title='Same original boxes: score changes')
for i,gain in enumerate(gains):ax[0].text(gain-.06 if gain<0 else gain+.05,i,f'{gain:+.3f}',ha='right' if gain<0 else 'left',va='center',fontsize=9)
names=['fire','smoke'];x=np.arange(2);w=.33
p=[d['val_TP50_associations'][c]['p_IoU_Pearson'] for c in names]
q=[d['val_TP50_associations'][c]['q_IoU_Pearson'] for c in names]
ax[1].bar(x-w/2,p,w,label='Original p',color='#4b78a8')
ax[1].bar(x+w/2,q,w,label='Learned q',color='#b67430')
ax[1].set(xticks=x,xticklabels=names,ylim=(0,.82),ylabel='Pearson correlation with true IoU',title='Within original true positives (TP50)')
ax[1].legend(frameon=False)
for i,(v,z) in enumerate(zip(p,q)):
 ax[1].text(i-w/2,v+.025,f'{v:.3f}',ha='center',fontsize=9)
 ax[1].text(i+w/2,z+.025,f'{z:.3f}',ha='center',fontsize=9)
base=d['operational_fixed_candidates_conf025']['baseline_p'];new=d['operational_fixed_candidates_conf025']['product_pq']
keys=['fire_16to64_recall75','mixed_smoke_recall50','negative_alarm_rate']
y=np.arange(3)
ax[2].bar(y-w/2,[100*base[k] for k in keys],w,label='Original p',color='#4b78a8')
ax[2].bar(y+w/2,[100*new[k] for k in keys],w,label='p*q',color='#b67430')
ax[2].set(xticks=y,xticklabels=['Fire 16-64px\nRecall75','Mixed smoke\nRecall50','Negative images\nAlarm rate'],ylim=(0,66),ylabel='Percent',title='Same score threshold 0.25**')
ax[2].legend(frameon=False)
fig.suptitle('Frozen YOLO11n quality head: failure audit on 29,804 FASDD val images',fontsize=14)
fig.text(.01,-.055,'* TP50 identity requires GT: diagnostic only.  ** Same numeric threshold, not matched false alarm rates.\nAll score variants are exploratory fixed-candidate audits; the registered pre-NMS result was -0.501 pp.',fontsize=9)
fig.savefig(DATA/'quality_failure_diagnostics.png',dpi=180,bbox_inches='tight')
plt.close(fig)
print(DATA/'quality_failure_diagnostics.png')
