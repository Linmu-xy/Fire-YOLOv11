#!/usr/bin/env python3
"""Build an immutable FASDD_CV version; this script never starts training.

Exact file bytes and EXIF-oriented RGB pixels are exclusion evidence.
Perceptual hashes produce review candidates, never automatic exclusions.
"""
from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import io
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATASET = 'fasdd_cv_clean_v1'
CONFIG = 'configs/protocol-fasdd-cv-clean-v1.yaml'
PRIORITY = {'test': 0, 'val': 1, 'train': 2}
POLICY = {
    'version': 1,
    'source': 'canonical fasdd (FASDD_CV; invalid-label images already excluded)',
    'reference': 'canonical dfire, all train/val/test splits',
    'cross_dataset': 'exclude FASDD members of byte/RGB-pixel equivalence components',
    'within_fasdd': 'one image per equivalence component; test > val > train, then source id',
    'pixel_signature': 'SHA256 of EXIF-transposed RGB width,height and pixel bytes',
    'perceptual_screen': {'phash_distance': 4, 'dhash_distance': 4, 'global_gray32_ssim_min': 0.95},
    'perceptual_action': 'audit candidates only; no automatic exclusion or scene-id assignment',
    'split_action': 'retain official split of survivor; never move images between splits',
}


def dump(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bits_hash(bits):
    value = 0
    for b in np.asarray(bits).reshape(-1):
        value = (value << 1) | int(b)
    return value


def scan_one(job):
    identity, dataset, split, filename = job
    record = {'source_id': identity, 'dataset': dataset, 'split': split, 'image': filename}
    before = Path(filename).stat()
    data = Path(filename).read_bytes()
    after = Path(filename).stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Source image changed during scan: ' + filename)
    record.update(sha256=hashlib.sha256(data).hexdigest(), size=after.st_size, mtime_ns=after.st_mtime_ns)
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()  # Complete decode; truncated files are not silently accepted.
            record['native_size'] = list(image.size)
            rgb = ImageOps.exif_transpose(image).convert('RGB')
            w, h = rgb.size
            pixels = hashlib.sha256(f'RGB:{w}:{h}:'.encode())
            pixels.update(rgb.tobytes())
            gray = rgb.convert('L')
            small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS))
            horizontal = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
            k = np.arange(32, dtype=np.float64)[:, None]
            x = np.arange(32, dtype=np.float64)[None, :]
            dct = np.cos(np.pi * (x + 0.5) * k / 32) * np.sqrt(2.0 / 32)
            dct[0] = np.sqrt(1.0 / 32)
            low = (dct @ small.astype(np.float64) @ dct.T)[:8, :8].reshape(-1)
            record.update(decoded_sha256=pixels.hexdigest(), display_size=[w, h],
                          dhash=bits_hash(horizontal[:, :-1] > horizontal[:, 1:]),
                          phash=bits_hash(low > np.median(low[1:])),
                          gray_std=float(small.std()),
                          gray32=base64.b64encode(small.tobytes()).decode(), decode_error=None)
    except Exception as error:
        record.update(decoded_sha256=None, decode_error=f'{type(error).__name__}: {error}')
    return record


def select_records(records):
    """Pure deterministic selection; only proven byte/pixel identity can exclude."""
    parents = list(range(len(records)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    tables = [dict(), dict()]
    for i, row in enumerate(records):
        for table, value in zip(tables, (row['sha256'], row.get('decoded_sha256'))):
            if value is None:
                continue
            if value in table:
                parents[find(i)] = find(table[value])
            else:
                table[value] = i
    components = defaultdict(list)
    for i in range(len(records)):
        components[find(i)].append(records[i])
    kept, excluded = [], []
    for members in components.values():
        targets = [r for r in members if r['dataset'] == 'fasdd']
        references = [r for r in members if r['dataset'] == 'dfire']
        good = []
        for row in targets:
            if row.get('decode_error'):
                excluded.append({'source_id': row['source_id'], 'split': row['split'],
                                 'reason': 'image_decode_failure', 'details': row['decode_error']})
            else:
                good.append(row)
        if references:
            ref_bytes = {r['sha256'] for r in references}
            for row in good:
                excluded.append({'source_id': row['source_id'], 'split': row['split'],
                                 'reason': 'cross_dataset_byte_duplicate' if row['sha256'] in ref_bytes
                                 else 'cross_dataset_decoded_duplicate',
                                 'matched_ids': sorted(r['source_id'] for r in references)})
        elif good:
            survivor = min(good, key=lambda r: (PRIORITY[r['split']], r['source_id']))
            kept.append(survivor)
            for row in good:
                if row is survivor:
                    continue
                scope = 'within_split' if row['split'] == survivor['split'] else 'cross_split'
                kind = 'byte' if row['sha256'] == survivor['sha256'] else 'decoded'
                excluded.append({'source_id': row['source_id'], 'split': row['split'],
                                 'reason': f'{scope}_{kind}_duplicate',
                                 'kept_id': survivor['source_id'], 'kept_split': survivor['split']})
    return sorted(kept, key=lambda r: r['source_id']), sorted(excluded, key=lambda r: r['source_id'])


def similarity(a, b):
    a, b = a.astype(float), b.astype(float)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a-ma)*(b-mb)).mean()
    c1, c2 = (0.01*255)**2, (0.03*255)**2
    return float(((2*ma*mb+c1)*(2*cov+c2))/((ma*ma+mb*mb+c1)*(va+vb+c2)))


def audit_near(records, output):
    from firesmoke.data import BKTree
    tree, prior = BKTree(), []
    counts, impacted = Counter(), defaultdict(set)
    hash_candidates = accepted = low_texture = 0
    with Path(output).open('x', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['a_id', 'b_id', 'phash_distance', 'dhash_distance', 'global_gray32_ssim', 'low_texture'])
        for i, row in enumerate(records):
            thumb = np.frombuffer(base64.b64decode(row['gray32']), dtype=np.uint8).reshape(32, 32)
            for j, distance in tree.query(row['phash'], 4):
                other, other_thumb = prior[j]
                if row['dataset'] == other['dataset'] == 'dfire':
                    continue
                if row['dataset'] == other['dataset'] and row['split'] == other['split']:
                    continue
                dd = (row['dhash'] ^ other['dhash']).bit_count()
                if dd > 4:
                    continue
                hash_candidates += 1
                score = similarity(thumb, other_thumb)
                if score < 0.95:
                    continue
                low = min(row['gray_std'], other['gray_std']) < 5
                low_texture += low
                accepted += 1
                a, b = sorted((row['source_id'], other['source_id']))
                writer.writerow([a, b, distance, dd, f'{score:.8f}', int(low)])
                scope = ' <-> '.join(sorted((row['dataset']+'/'+row['split'], other['dataset']+'/'+other['split'])))
                counts[scope] += 1
                impacted[row['dataset']+'/'+row['split']].add(row['source_id'])
                impacted[other['dataset']+'/'+other['split']].add(other['source_id'])
            tree.add(row['phash'], i)
            prior.append((row, thumb))
            if (i+1) % 5000 == 0:
                print(f'NEAR_PROGRESS {i+1}/{len(records)} candidates={accepted}', flush=True)
    return {'registered_screen': POLICY['perceptual_screen'], 'hash_candidates': hash_candidates,
            'review_candidate_pairs': accepted, 'low_texture_pairs': low_texture,
            'pairs_by_scope': dict(counts), 'impacted_images_by_scope': {k:len(v) for k,v in impacted.items()},
            'review_status': 'pending' if accepted else 'no candidates under registered screen',
            'scope': 'FASDD across splits and FASDD vs D-Fire; same-split perceptual redundancy not screened',
            'action': 'No perceptual candidate is automatically excluded; no scene identity is inferred'}


def verify(root=ROOT, config_file=CONFIG):
    from firesmoke.data import load_protocol, load_prepared
    cfg = load_protocol(root / config_file)
    out, metadata, rows = load_prepared(cfg, DATASET)
    audit = out / 'audit'
    scan = [json.loads(l) for l in (audit/'content_hashes.jsonl').open()]
    refs = [r for r in scan if r['dataset']=='dfire']
    bytes_seen, pixels_seen = set(), set()
    ref_bytes = {r['sha256'] for r in refs}
    ref_pixels = {r['decoded_sha256'] for r in refs}
    for i, row in enumerate(rows, 1):
        if file_hash(row['prepared_image']) != row['image_sha256']:
            raise ValueError('Image bytes changed: ' + row['id'])
        sha, pixel = row['image_sha256'], row['decoded_sha256']
        if sha in bytes_seen or pixel in pixels_seen or sha in ref_bytes or pixel in ref_pixels:
            raise ValueError('Content identity collision in new version: ' + row['id'])
        bytes_seen.add(sha); pixels_seen.add(pixel)
        if i % 20000 == 0:
            print(f'VERIFY_IMAGES {i}/{len(rows)}', flush=True)
    for row in refs:
        if file_hash(row['image']) != row['sha256']:
            raise ValueError('D-Fire reference changed: ' + row['source_id'])
    for name, expected in metadata['audit_sha256'].items():
        if file_hash(audit/name) != expected:
            raise ValueError('Audit file changed: ' + name)
    if cfg['dataset_version']['portable_content_sha256'] != metadata['portable_content_sha256']:
        raise ValueError('Protocol content fingerprint mismatch')
    result = {'dataset': DATASET, 'canonical_labels_and_manifests': 'passed',
              'full_kept_and_reference_byte_rehash': 'passed', 'image_count': len(rows),
              'zero_byte_and_decoded_pixel_collisions_within_version_and_against_dfire': True,
              'counts': metadata['counts'], 'portable_content_sha256': metadata['portable_content_sha256'],
              'perceptual_review': metadata['perceptual_audit']['review_status'],
              'scene_independence': 'unverified'}
    return result


def build(root=ROOT, source_config='configs/protocol.yaml', workers=4):
    import yaml
    from firesmoke.common import path, digest, stable_digest
    from firesmoke.data import load_protocol, load_prepared, prepare
    config_path = root / CONFIG
    view = root/'data/raw/FASDD_CV_clean_v1'
    prepared = root/'data/prepared'/DATASET
    stage = root/'analysis/work'/f'{DATASET}_build'
    for dest in (view, prepared, config_path, stage):
        if dest.exists() or dest.is_symlink():
            raise FileExistsError(f'Refuse to overwrite existing version/staging path: {dest}')
    cfg = load_protocol(root/source_config)
    snapshots, source_rows = {}, {}
    protected = [root/source_config]
    for ds in ('fasdd','dfire'):
        directory, metadata, rows = load_prepared(cfg, ds)
        snapshots[ds] = metadata
        source_rows[ds] = rows
        protected += [directory/n for n in ('metadata.json','manifest.jsonl','excluded.json','data.yaml','train.txt','val.txt','test.txt')]
    before = {str(p.relative_to(root)):digest(p) for p in protected}
    stage.mkdir(parents=True)
    dump(stage/'build_input.json', {'source_config_sha256': digest(root/source_config), 'source_metadata':snapshots,
                                  'policy':POLICY, 'builder_sha256':digest(Path(__file__))})
    jobs = [(r['id'], ds, r['split'], r['image']) for ds in ('dfire','fasdd') for r in source_rows[ds]]
    scanned=[]; started=time.monotonic()
    with (stage/'content_hashes.jsonl').open('x', encoding='utf-8') as stream:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, row in enumerate(pool.map(scan_one, jobs, chunksize=16), 1):
                scanned.append(row)
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                if i % 2000 == 0:
                    stream.flush()
                    print(f'SCAN_PROGRESS {i}/{len(jobs)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    failures=[r for r in scanned if r.get('decode_error')]
    dump(stage/'decode_failures.json',failures)
    if any(r['dataset']=='dfire' for r in failures):
        raise ValueError('Reference images failed decode; inspect staging/decode_failures.json before publishing')
    for r in scanned:
        st=Path(r['image']).stat()
        if (st.st_size,st.st_mtime_ns)!=(r['size'],r['mtime_ns']):
            raise ValueError('Source changed after scan: '+r['image'])
    kept, excluded = select_records(scanned)
    old_excluded=json.loads((path(cfg['prepared_root'])/'fasdd/excluded.json').read_text())
    excluded += [{'source_id':r['id'],'split':r['split'],'reason':'invalid_source_label',
                  'label_issues':r['label_issues']} for r in old_excluded]
    excluded=sorted(excluded,key=lambda r:r['source_id'])
    jsonl(stage/'exclusions.jsonl',excluded)
    jsonl(stage/'retained_source_ids.jsonl',[{'source_id':r['source_id'],'split':r['split']} for r in kept])
    print('SELECTED_COUNTS',dict(Counter(r['split'] for r in kept)),flush=True)
    print('EXCLUSION_REASONS',dict(Counter(r['reason'] for r in excluded)),flush=True)
    near=audit_near([r for r in scanned if r['dataset']=='dfire']+kept,stage/'perceptual_candidates.csv')
    dump(stage/'perceptual_audit.json',near)
    ann=view/'annotations/YOLO_CV'; ann.mkdir(parents=True)
    source_root=path(cfg['datasets']['fasdd']['root'])
    (view/'images').symlink_to(source_root/'images', target_is_directory=True)
    (ann/'labels').symlink_to(source_root/'annotations/YOLO_CV/labels',target_is_directory=True)
    (ann/'images').symlink_to('../../images',target_is_directory=True)
    for split in ('train','val','test'):
        (ann/f'{split}.txt').write_text(''.join('./images/'+Path(r['image']).name+'\n' for r in kept if r['split']==split),encoding='utf-8')
    new=copy.deepcopy(cfg)
    new['datasets']={'dfire':cfg['datasets']['dfire'],DATASET:{'root':'data/raw/FASDD_CV_clean_v1',
        'layout':'fasdd','source_names':['fire','smoke'],'invalid_box_policy':'error'}}
    new['prepared_root']='data/prepared'
    prepare(new,DATASET)
    audit=prepared/'audit';audit.mkdir()
    import shutil
    for f in stage.iterdir():
        if f.is_file():shutil.copyfile(f,audit/f.name)
    mapping={Path(r['image']).name:r for r in kept}
    manifest=prepared/'manifest.jsonl'
    rows=[json.loads(l) for l in manifest.open()]
    for r in rows:
        original=mapping[Path(r['image']).name]
        r.update(source_id=original['source_id'],source_split=original['split'],
                 image_sha256=original['sha256'],decoded_sha256=original['decoded_sha256'])
    manifest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    metadata_path=prepared/'metadata.json'
    metadata=json.loads(metadata_path.read_text())
    content=stable_digest([{k:r[k] for k in ('id','boxes','image_sha256','decoded_sha256')} for r in rows])
    metadata.update(manifest_sha256=digest(manifest),image_content_hashed=True,
        portable_content_sha256=content,selection_policy=POLICY,
        original_source_counts=snapshots['fasdd']['counts'],source_metadata=snapshots,
        excluded_source_counts=dict(Counter(r['split'] for r in excluded)),
        exclusion_reasons=dict(Counter(r['reason'] for r in excluded)),
        perceptual_audit=near,audit_sha256={f.name:digest(f) for f in audit.iterdir() if f.is_file()})
    metadata_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    new['dataset_version']={'name':DATASET,'portable_content_sha256':content,'policy':POLICY,
        'perceptual_review':near['review_status'],'scene_independence':'unverified',
        'test_history':'Original FASDD and D-Fire tests were previously inspected; not a fresh confirmatory holdout'}
    config_path.write_text('# Generated immutable data version; this config does not start training.\n'+yaml.safe_dump(new,sort_keys=False,allow_unicode=True),encoding='utf-8')
    after={str(p.relative_to(root)):digest(p) for p in protected}
    if before!=after:
        raise ValueError('An original manifest/config changed during build')
    result=verify(root)
    stats={s:Counter() for s in ('train','val','test')}
    for r in rows:
        c=stats[r['split']];c['images']+=1;c['background']+=not r['boxes']
        classes={b[0] for b in r['boxes']}
        c['both' if classes=={0,1} else 'fire_only' if classes=={0} else 'smoke_only' if classes=={1} else 'none']+=1
        for b in r['boxes']:c[['fire_boxes','smoke_boxes'][b[0]]]+=1
    version={'status':'built_and_integrity_verified','dataset':DATASET,'policy':POLICY,
        'counts':dict(stats),'exclusion_reasons':metadata['exclusion_reasons'],'excluded_counts':metadata['excluded_source_counts'],
        'source_unchanged':before==after,'source_hashes':before,'verification':result,
        'perceptual_audit':near,'portable_content_sha256':content,'protocol_sha256':digest(config_path),
        'builder_sha256':digest(Path(__file__)),'elapsed_seconds':round(time.monotonic()-started,2)}
    dump(prepared/'version.json',version)
    table='\n'.join(f"| {s} | {c['images']} | {c['background']} | {c['fire_boxes']} | {c['smoke_boxes']} |" for s,c in stats.items())
    doc=f'''# FASDD_CV clean v1\n\nVersion: {DATASET}\n\n| split | images | background | fire boxes | smoke boxes |\n|---|---:|---:|---:|---:|\n{table}\n\nExclusions: {json.dumps(metadata['exclusion_reasons'],ensure_ascii=False)}\n\nAll retained and D-Fire reference files were decoded and byte-hashed. There are zero retained byte or decoded-pixel identity collisions within this version and against the canonical D-Fire reference. Official split membership is preserved; byte/pixel duplicate priority is test > val > train.\n\nPerceptual review candidates: {near['review_candidate_pairs']}; status: {near['review_status']}. Candidate pairs are NOT confirmed duplicates and have not been automatically excluded. Scene independence remains unverified. Existing tests have already influenced development; this version is not a new blind confirmatory holdout.\n\nData: data/prepared/{DATASET}/data.yaml\n\nProtocol: {CONFIG}\n\nPortable content fingerprint: {content}\n\nInspect audit/exclusions.jsonl, content_hashes.jsonl, perceptual_candidates.csv and version.json for provenance. Source images are linked, canonical labels are separate files. Original data/configs remain unchanged.\n\nPreview (no training):\n\n```bash\npython -m firesmoke --config {CONFIG} plan --dataset {DATASET} --method baseline --seed 0 --tag fasdd-cv-clean-v1\npython scripts/prepare_fasdd_clean_v1.py --verify-only\n```\n\nTraining entry after reviewing the registered protocol:\n\n```bash\npython -m firesmoke --config {CONFIG} train --dataset {DATASET} --method baseline --seed 0 --tag fasdd-cv-clean-v1 --execute\n```\n\nThis builder has not launched training. For another workstation, copy the original datasets and rebuild symlinks locally; do not copy absolute symlinks as portable data.\n'''
    (prepared/'VERSION.md').write_text(doc,encoding='utf-8')
    docs=root/'docs/FASDD_CV_CLEAN_V1.md'
    if docs.exists():raise FileExistsError(docs)
    docs.write_text(doc,encoding='utf-8')
    print('VERSION_READY '+json.dumps(version,ensure_ascii=False),flush=True)
    return version


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true',help='Build data only; never train')
    p.add_argument('--verify-only',action='store_true',help='Rehash and verify an existing version')
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if not 1<=a.workers<=8:p.error('workers must be 1..8')
    if a.execute and a.verify_only:p.error('choose build or verify, not both')
    if a.verify_only:print(json.dumps(verify(),ensure_ascii=False,indent=2))
    elif a.execute:build(workers=a.workers)
    else:print(json.dumps({'dataset':DATASET,'policy':POLICY,'execute':False,
                          'command':'python scripts/prepare_fasdd_clean_v1.py --execute --workers 4'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
