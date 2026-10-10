# FASDD_CV clean v1

Version: fasdd_cv_clean_v1

| split | images | background | fire boxes | smoke boxes |
|---|---:|---:|---:|---:|
| train | 44113 | 19525 | 33137 | 22869 |
| val | 29804 | 13019 | 20984 | 15474 |
| test | 15488 | 6501 | 8312 | 7733 |

Exclusions: {"cross_dataset_byte_duplicate": 5733, "within_split_byte_duplicate": 90, "invalid_source_label": 3, "cross_split_decoded_duplicate": 11, "cross_dataset_decoded_duplicate": 6, "within_split_decoded_duplicate": 6, "cross_split_byte_duplicate": 60}

All retained and D-Fire reference files were decoded and byte-hashed. There are zero retained byte or decoded-pixel identity collisions within this version and against the canonical D-Fire reference. Official split membership is preserved; byte/pixel duplicate priority is test > val > train.

Perceptual review candidates: 168098; status: pending. Candidate pairs are NOT confirmed duplicates and have not been automatically excluded. Scene independence remains unverified. Existing tests have already influenced development; this version is not a new blind confirmatory holdout.

Data: data/prepared/fasdd_cv_clean_v1/data.yaml

Protocol: configs/protocol-fasdd-cv-clean-v1.yaml

Portable content fingerprint: 56d5ca7e69646edd5b9a3351c86bfa24b190cf37e01af1335721955e8d6a1ca2

Inspect audit/exclusions.jsonl, content_hashes.jsonl, perceptual_candidates.csv and version.json for provenance. Source images are linked, canonical labels are separate files. Original data/configs remain unchanged.

Preview (no training):

```bash
python -m firesmoke --config configs/protocol-fasdd-cv-clean-v1.yaml plan --dataset fasdd_cv_clean_v1 --method baseline --seed 0 --tag fasdd-cv-clean-v1
python scripts/prepare_fasdd_clean_v1.py --verify-only
```

Training entry after reviewing the registered protocol:

```bash
python -m firesmoke --config configs/protocol-fasdd-cv-clean-v1.yaml train --dataset fasdd_cv_clean_v1 --method baseline --seed 0 --tag fasdd-cv-clean-v1 --execute
```

This builder has not launched training. For another workstation, copy the original datasets and rebuild symlinks locally; do not copy absolute symlinks as portable data.
