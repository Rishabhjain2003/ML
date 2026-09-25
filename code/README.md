# Business Entity Resolution — Pipeline

## How to Run (from your Mac Terminal)

```bash
cd ~/MLChallenge
venv/bin/python3 code/pipeline.py
```

Estimated runtime: **45–90 minutes** (dominated by blocking 1.7M test S1 entities).

## What It Does

1. **Loads** training S1 + ground truth (S2/S3 loaded incrementally to save RAM)
2. **Builds** 3-path blocking indices on full S2+S3 (10M records):
   - Name token inverted index (IDF-filtered)
   - Multi-key geographic index (last 1/2/3/4 address tokens)
   - Address numeric token index (catches Hindi/cross-script pairs)
3. **Trains** LightGBM on 30% of S1 (~660K entities, ~18M pairs after neg sampling)
4. **Tunes** classification threshold on held-out validation to maximize F_0.5
5. **Runs inference** on test set (1.73M S1, ~10M S2+S3)
6. **Writes** `output/matching_results.tsv` and `output/candidate_pairs.tsv`
7. **Validates** output format

## Validated Performance (medium-scale test)

- Blocking recall: ~85–92%
- Val F_0.5: **~0.90** (medium sample, 60K S1)
- Classification threshold: ~0.875

## Memory Requirements

Peak ~5–6 GB RAM. The pipeline releases raw S2/S3 dataframes immediately after indexing.

## Files

```
code/
├── pipeline.py          # main end-to-end script
├── requirements.txt     # pinned dependencies
└── src/
    ├── normalize.py     # text normalization (names, addresses, domains, Indic scripts)
    ├── blocking.py      # 3-path candidate generation
    ├── features.py      # 15 feature vector per candidate pair
    └── train.py         # training utilities (also used by pipeline.py)
```

## Reproducing From Scratch

```bash
cd ~/MLChallenge
python3 -m venv venv
venv/bin/pip install -r code/requirements.txt
brew install libomp   # required for LightGBM on macOS
venv/bin/python3 code/pipeline.py
```

Output files will be at `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
Run the validator before submitting:
```bash
venv/bin/python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
