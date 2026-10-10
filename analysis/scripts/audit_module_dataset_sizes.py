"""Read manifest boxes and image headers; no image edits or model execution.

Size convention: aspect-preserving whole-image resize with max side 640.
This describes static source geometry, not augmented Mosaic/rect validation.
"""
import argparse
import json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from PIL import Image

def measure(root,datasets):
    out={}
    for ds in datasets:
        p=root/'data/prepared'/ds/'manifest.jsonl'
        groups=defaultdict(lambda:defaultdict(list)); imgs=defaultdict(Counter);dims=defaultdict(list)
        with p.open() as f:
            for line in f:
                row=json.loads(line)
                if row.get('excluded'):continue
                split=row['split'];boxes=row['boxes'];classes={int(b[0]) for b in boxes}
                imgs[split]['images']+=1
                imgs[split]['background' if not classes else 'both' if len(classes)==2 else 'fire_only' if 0 in classes else 'smoke_only']+=1
                with Image.open(row['prepared_image']) as im:
                    w,h=im.size
                    if im.getexif().get(274,1) in (5,6,7,8):w,h=h,w
                dims[split].append(max(w,h));ratio=640/max(w,h)
                for c,_,_,bw,bh in boxes:
                    bw,bh=bw*w*ratio,bh*h*ratio
                    groups[split][int(c)].append([max(bw,bh),min(bw,bh),(bw*bh)**.5])
        summary={}
        for split in imgs:
            result={'images':dict(imgs[split]),'native_longside_percentiles':np.percentile(dims[split],[10,50,90]).tolist(),'classes':{}}
            for c,entries in groups[split].items():
                arr=np.array(entries); long,short,area=arr.T
                edges=[0,8,16,32,64,float('inf')]
                buckets=[int(((long>=lo)&(long<hi)).sum()) for lo,hi in zip(edges,edges[1:])]
                result['classes'][['fire','smoke'][c]]={'boxes':len(arr),'long_buckets_0_8_16_32_64_inf':buckets,
                 'long_percentiles_10_50_90':np.percentile(long,[10,50,90]).tolist(),
                 'short_lt8':int((short<8).sum()),'short_lt16':int((short<16).sum()),
                 'sqrt_area_lt16':int((area<16).sum()),'sqrt_area_lt32':int((area<32).sum())}
            summary[split]=result
        out[ds]=summary
        print(json.dumps({'dataset':ds,'summary':summary},ensure_ascii=False),flush=True)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);p.add_argument('datasets',nargs='+');a=p.parse_args()
    result=measure(a.root,a.datasets)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
