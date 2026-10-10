"""Read trusted local historical checkpoints on CPU; no forward/inference/training."""
import argparse
import json
import statistics
from pathlib import Path
import torch
from torch import nn
import firesmoke.models  # register existing research classes for checkpoint loading

def tensor_stats(t):
    t=t.detach().float().cpu()
    return {'shape':list(t.shape),'rms':float(t.square().mean().sqrt()),
            'max_abs':float(t.abs().max()),'nonzero':int(torch.count_nonzero(t))}

def audit(root):
    out=[]
    for f in sorted((root/'results/dfire').glob('*/*/seed*/run.json')):
        cfg=json.loads(f.read_text()); method=cfg['method']
        if cfg['training']['imgsz']!=640:continue
        w=f.parent/'weights/best.pt'
        if not w.exists():continue
        ckpt=torch.load(w,map_location='cpu',weights_only=False)
        model=ckpt.get('ema')
        if model is None:model=ckpt['model']
        head=model.model[-1]
        layers=[]
        for i,m in enumerate(model.model):
            layers.append({'index':i,'type':type(m).__name__,'from':getattr(m,'f',None),
                           'parameters':sum(p.numel() for p in m.parameters())})
        cls=[]
        for branch in head.cv3:
            cls.append([{'shape':list(m.weight.shape),'groups':m.groups}
                        for m in branch.modules() if isinstance(m,nn.Conv2d)])
        modules=[]
        for m in model.modules():
            if type(m).__name__ in ('SemanticResidualDetailGate','DetailConditionedBandResidual','AlignedContextResidual'):
                modules.append({'type':type(m).__name__,
                                'parameters':{n:tensor_stats(p) for n,p in m.named_parameters()}})
        exposure=[]
        exp=f.parent/'background_exposure.jsonl'
        if exp.exists():
            rows=[json.loads(l) for l in exp.read_text().splitlines() if l.strip()]
            for lo,hi in [(1,10),(80,89),(91,100)]:
                part=[r for r in rows if lo<=r['epoch']<=hi]
                if part:
                    total=sum(r['images'] for r in part); empty=sum(r['post_augmentation_empty_images'] for r in part)
                    exposure.append({'epochs':[lo,hi],'images':total,'empty':empty,'fraction':empty/total})
        out.append({'path':str(f.relative_to(root)),'method':method,'seed':cfg['seed'],
                    'checkpoint_epoch':ckpt.get('epoch'),'total_parameters':sum(p.numel() for p in model.parameters()),
                    'head_parameters':sum(p.numel() for p in head.parameters()),
                    'head_stride':head.stride.tolist(),'head_input_channels':[b[0].conv.in_channels for b in head.cv2],
                    'classification_convs':cls,'layers':layers,'module_parameters':modules,
                    'background_exposure':exposure,'provenance':cfg.get('provenance'),
                    'initialization':cfg.get('initialization'),'data_metadata':cfg.get('data_metadata')})
        del ckpt,model
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=audit(a.root)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps([{'path':r['path'],'params':r['total_parameters'],'head_params':r['head_parameters'],
                       'channels':r['head_input_channels'],'cls_shapes':r['classification_convs'][0],
                       'background_exposure':r['background_exposure'],
                       'modules':r['module_parameters']} for r in result],ensure_ascii=False,indent=2))
