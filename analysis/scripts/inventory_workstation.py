"""Read-only two-workstation repository inventory; no network mutations."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def inventory(root):
    def git(*args):
        p=subprocess.run(['git',*args],cwd=root,capture_output=True,text=True)
        return p.stdout.strip() if p.returncode==0 else None
    ignored={'__pycache__','.pytest_cache','.git','.runtime'}
    code={}
    candidates=[p for p in root.iterdir() if p.is_file()]
    for directory in ['firesmoke','configs','scripts','tests','docs','weights','analysis/scripts','analysis/reports']:
        base=root/directory
        if base.exists(): candidates.extend(p for p in base.rglob('*') if p.is_file())
    for p in sorted(set(candidates)):
        rel=p.relative_to(root)
        if ignored.intersection(rel.parts) or p.suffix in ['.pyc','.pyo']: continue
        before=p.stat(); value=sha(p); after=p.stat()
        code[str(rel)]={'size':after.st_size,'sha256':value,
                        'changed_while_reading':(before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns)}
    results={}
    active_prefix='results/fasdd_cv_clean_v1/fasdd-cv-clean-v1/baseline/seed0/'
    for p in sorted((root/'results').rglob('*')):
        if not p.is_file():continue
        relative=str(p.relative_to(root))
        if ignored.intersection(p.relative_to(root).parts) or relative.startswith(active_prefix):continue
        before=p.stat(); value=sha(p); after=p.stat()
        results[relative]={'size':after.st_size,'sha256':value,
                           'changed_while_reading':(before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns)}
    prepared={}
    for p in sorted((root/'data/prepared').glob('*/metadata.json')):
        m=json.loads(p.read_text())
        prepared[p.parent.name]={'metadata_sha256':sha(p),
          **{k:m.get(k) for k in ['counts','portable_inventory_sha256','portable_content_sha256','source_names']}}
    versions={p:importlib.metadata.version(p) for p in ['ultralytics','torch','torchvision','numpy','Pillow','PyYAML']}
    return {'root':str(root),'captured_at_unix':time.time(),'git_commit':git('rev-parse','HEAD'),
            'git_status':git('status','--short'),'branches':git('branch','-vv'),
            'tracked_files':git('ls-files').splitlines(),'code_and_weights':code,
            'finished_result_files':results,'active_result_excluded':active_prefix,
            'prepared':prepared,'versions':versions,
            'instructions_found':[str(p/'AGENTS.md') for p in [root,*root.parents] if (p/'AGENTS.md').exists()]}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args(); result=inventory(args.root.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({'output':str(args.output),'git_commit':result['git_commit'],
                      'code_files':len(result['code_and_weights']),
                      'result_files':len(result['finished_result_files']),
                      'instructions_found':result['instructions_found']},ensure_ascii=False))
