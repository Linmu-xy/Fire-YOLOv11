"""Read existing registered training curves; no model inference or training."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

KEYS=['train/box_loss','train/cls_loss','train/dfl_loss','val/box_loss','val/cls_loss','val/dfl_loss',
      'metrics/mAP50(B)','metrics/mAP50-95(B)','metrics/precision(B)','metrics/recall(B)','lr/pg0']

def diagnostic(root):
    result=[]
    for file in sorted((root/'results').glob('*/*/*/seed*/run.json')):
        relative=str(file.relative_to(root))
        if '/fasdd_cv_clean_v1/' in relative:continue
        cfg=json.loads(file.read_text());curve=file.parent/'results.csv'
        if not curve.exists():continue
        rows=[]
        for row in csv.DictReader(curve.open()):
            row={k.strip():v.strip() for k,v in row.items() if k}
            rows.append({k:float(v) for k,v in row.items() if v})
        if not rows:continue
        score='metrics/mAP50-95(B)'
        finite=[r for r in rows if math.isfinite(r.get(score,float('nan')))]
        best=max(finite,key=lambda r:r[score])
        epochs=[int(r['epoch']) for r in rows]
        windows={}
        for lo,hi in [(1,10),(20,30),(45,55),(80,89),(91,100)]:
            part=[r for r in rows if lo<=r['epoch']<=hi]
            if part:windows[f'{lo}-{hi}']={k:statistics.mean(r[k] for r in part if k in r) for k in KEYS}
        run={'path':relative,'dataset':cfg['dataset'],'method':cfg['method'],'seed':cfg['seed'],
             'training':cfg['training'],'rows':len(rows),'last_epoch':max(epochs),
             'missing_epochs':sorted(set(range(1,max(epochs)+1))-set(epochs)),
             'nonfinite_cells':sum(not math.isfinite(v) for r in rows for v in r.values()),
             'best':best,'last':rows[-1],'windows':windows,'curve':rows,
             'samples':[{k:r.get(k) for k in ['epoch',*KEYS]} for r in rows if int(r['epoch']) in (1,10,25,50,75,89,90,91,95,100)],
             'training_code_sha256':{k:v for k,v in cfg.get('provenance',{}).get('source_sha256',{}).items() if k.startswith('firesmoke/')},
             'source_names':cfg.get('data_metadata',{}).get('names')}
        result.append(run)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();r=diagnostic(a.root.resolve());a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(r,f,indent=2,ensure_ascii=False)
    print(json.dumps([{'path':v['path'],'rows':v['rows'],'best_epoch':v['best']['epoch'],
                      'map50':v['best']['metrics/mAP50(B)'],'map5095':v['best']['metrics/mAP50-95(B)'],
                      'nonfinite_cells':v['nonfinite_cells']} for v in r],indent=2))
