"""Portable digest of version evidence, ignoring machine paths and mtimes."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

def compact(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()+b'\n'

def evidence(root):
    prepared=root/'data/prepared/fasdd_cv_clean_v1'
    version=json.loads((prepared/'version.json').read_text())
    hashes={name:hashlib.sha256() for name in ['dfire','fasdd']}
    counts=Counter()
    with (prepared/'audit/content_hashes.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            entry={k:r.get(k) for k in ['source_id','dataset','split','sha256','decoded_sha256','native_size','display_size']}
            hashes[r['dataset']].update(compact(entry));counts[r['dataset']]+=1
    selected=hashlib.sha256()
    for line in (prepared/'manifest.jsonl').open():
        r=json.loads(line)
        selected.update(compact({k:r[k] for k in ['id','source_id','source_split','boxes','image_sha256','decoded_sha256']}))
    files={}
    for name in ['exclusions.jsonl','retained_source_ids.jsonl','perceptual_candidates.csv']:
        h=hashlib.sha256()
        with (prepared/'audit'/name).open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        files[name]=h.hexdigest()
    return {'portable_content_sha256':version['portable_content_sha256'],
            'protocol_sha256':version['protocol_sha256'],'builder_sha256':version['builder_sha256'],
            'source_counts':dict(counts),'source_content_digest':{k:v.hexdigest() for k,v in hashes.items()},
            'selected_image_ids_labels_content_digest':selected.hexdigest(),
            'audit_files_sha256':files,'split_counts':{s:c['images'] for s,c in version['counts'].items()},
            'excluded_counts':version['excluded_counts'],'exclusion_reasons':version['exclusion_reasons'],
            'perceptual_audit':version['perceptual_audit']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
    print(json.dumps(evidence(args.root.resolve()),ensure_ascii=False,indent=2))
