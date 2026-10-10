"""Recover recorded source hashes from local Git history without changing checkout."""
import argparse
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

def audit(root,out):
    evidence=json.loads((root/'analysis/artifacts/historical_diagnosis_20261008/module_checkpoint_audit.json').read_text())
    targets={}
    for r in evidence:
        for name in ['firesmoke/models.py','firesmoke/data.py','firesmoke/experiment.py']:
            sha=r['provenance']['source_sha256'][name]
            targets[(name,sha)]=None
    recovered={}
    for name in ['firesmoke/models.py','firesmoke/data.py','firesmoke/experiment.py']:
        commits=subprocess.check_output(['git','log','--all','--format=%H','--',name],cwd=root,text=True).splitlines()
        commits+=list({r['provenance']['git_commit'] for r in evidence})
        for commit in dict.fromkeys(commits):
            p=subprocess.run(['git','show',commit+':'+name],cwd=root,capture_output=True)
            if p.returncode:continue
            sha=hashlib.sha256(p.stdout).hexdigest();key=(name,sha)
            if key in targets and key not in recovered:
                recovered[key]={'commit':commit,'text':p.stdout.decode()}
    out.mkdir(parents=True,exist_ok=True)
    result=[]
    for (name,sha) in targets:
        hit=recovered.get((name,sha));record={'file':name,'recorded_sha256':sha,'recovered':bool(hit)}
        if hit:
            record['commit']=hit['commit'];f=out/(Path(name).stem+'_'+sha[:12]+'.py')
            if not f.exists():f.write_text(hit['text'],encoding='utf-8')
        result.append(record)
    lookup={(r['method'],r['seed']):r for r in evidence}
    diffs={}
    for a,b in [(('p2',0),('baseline',0)),(('p2',0),('p2_srdg',0)),(('p2_srdg',0),('p2_dcbr',0)),(('p2_dcbr',0),('p2_acr',0))]:
        for name in ['firesmoke/models.py','firesmoke/data.py']:
            ha=lookup[a]['provenance']['source_sha256'][name];hb=lookup[b]['provenance']['source_sha256'][name]
            if ha==hb:continue
            va=recovered.get((name,ha));vb=recovered.get((name,hb))
            if va and vb:
                key=f'{a[0]}{a[1]}_to_{b[0]}{b[1]}_{Path(name).stem}'
                diffs[key]=''.join(difflib.unified_diff(va['text'].splitlines(True),vb['text'].splitlines(True),fromfile=ha[:12],tofile=hb[:12]))
    payload={'hash_recovery':result,'diffs':diffs}
    with (out/'source_audit.json').open('x',encoding='utf-8') as f:json.dump(payload,f,indent=2)
    print(json.dumps(payload,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();audit(a.root,a.output)
