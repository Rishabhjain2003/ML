"""
Inference pipeline: generate matching_results.tsv and candidate_pairs.tsv
for the test set using trained LightGBM model.
"""
import os
import sys
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize_name, normalize_address
from blocking import build_indices, generate_candidates
from features import compute_features

BASE_TEST = "/Users/arvshaa/MLChallenge/dataset/test"
MODELS    = "/Users/arvshaa/MLChallenge/models"
OUTDIR    = "/Users/arvshaa/MLChallenge/output"
os.makedirs(OUTDIR, exist_ok=True)


def run_inference(clf, threshold: float,
                  name_idx, city_idx,
                  test_s1: pd.DataFrame,
                  s2_norm: pd.DataFrame,
                  s3_norm: pd.DataFrame):

    # Fast lookup for S2+S3
    s23_lookup = {}
    for _, row in s2_norm.iterrows():
        s23_lookup[row['entity_id']] = row.to_dict()
    for _, row in s3_norm.iterrows():
        s23_lookup[row['entity_id']] = row.to_dict()

    # Normalize test S1
    test_s1 = test_s1.copy()
    test_s1['name_norm'] = test_s1['business_name'].map(normalize_name)
    test_s1['addr_norm'] = test_s1['business_address'].map(normalize_address)
    s1_lookup = {row['entity_id']: row.to_dict()
                 for _, row in test_s1.iterrows()}

    # Generate candidates
    print("Generating test candidates …")
    candidates = generate_candidates(test_s1, name_idx, geo_idx, numeric_idx)

    # Score each candidate pair
    print("Scoring candidate pairs …")
    matching_results = {}
    BATCH_SIZE = 50_000

    all_s1_ids = test_s1['entity_id'].tolist()

    # Collect all pairs for batch feature computation
    pair_list   = []  # (s1_id, other_id)
    s1_id_order = []

    for s1_id in all_s1_ids:
        cands = candidates.get(s1_id, [])
        for cid in cands:
            pair_list.append((s1_id, cid))
            s1_id_order.append(s1_id)

    print(f"  Total candidate pairs to score: {len(pair_list):,}")

    # Process in batches
    all_probs = []
    for start in tqdm(range(0, len(pair_list), BATCH_SIZE),
                      desc="scoring", mininterval=10):
        batch = pair_list[start:start+BATCH_SIZE]
        X_batch = []
        for s1_id, other_id in batch:
            r1 = s1_lookup.get(s1_id, {})
            r2 = s23_lookup.get(other_id, {})
            f  = compute_features(
                r1.get('name_norm',''), r1.get('addr_norm',''), r1.get('country',''),
                r2.get('name_norm',''), r2.get('addr_norm',''), r2.get('country',''),
            )
            X_batch.append(f)
        probs = clf.predict_proba(np.array(X_batch, dtype=np.float32))[:, 1]
        all_probs.extend(probs.tolist())

    # Build results dict
    s1_probs = defaultdict(list)
    for (s1_id, other_id), prob in zip(pair_list, all_probs):
        s1_probs[s1_id].append((other_id, prob))

    # Apply threshold
    for s1_id in all_s1_ids:
        pairs_with_probs = s1_probs.get(s1_id, [])
        matched = [eid for eid, p in pairs_with_probs if p >= threshold]
        matching_results[s1_id] = matched

    return matching_results, candidates


def write_output(matching_results: dict, candidates: dict,
                 all_s1_ids: list, outdir: str):
    # matching_results.tsv
    mr_path = os.path.join(outdir, 'matching_results.tsv')
    with open(mr_path, 'w') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matched = matching_results.get(s1_id, [])
            # Deduplicate, keep S2/S3 only
            seen = set()
            deduped = []
            for m in matched:
                if m not in seen and (m.startswith('S2-') or m.startswith('S3-')):
                    seen.add(m)
                    deduped.append(m)
            f.write(f"{s1_id}\t{','.join(deduped)}\n")

    # candidate_pairs.tsv
    cp_path = os.path.join(outdir, 'candidate_pairs.tsv')
    with open(cp_path, 'w') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = candidates.get(s1_id, [])
            seen = set()
            deduped = []
            for c in cands:
                if c not in seen and (c.startswith('S2-') or c.startswith('S3-')):
                    seen.add(c)
                    deduped.append(c)
            f.write(f"{s1_id}\t{','.join(deduped)}\n")

    print(f"\nWrote {mr_path}")
    print(f"Wrote {cp_path}")

    # Summary stats
    n_with_matches = sum(1 for v in matching_results.values() if v)
    n_singletons   = sum(1 for v in matching_results.values() if not v)
    avg_matches    = np.mean([len(v) for v in matching_results.values() if v])
    print(f"\nOutput summary:")
    print(f"  Entities with ≥1 match  : {n_with_matches:,}")
    print(f"  Singletons (no match)   : {n_singletons:,}")
    print(f"  Avg matches (non-single): {avg_matches:.2f}")


def main():
    # Load model + threshold
    print("Loading model …")
    with open(f"{MODELS}/lgbm_model.pkl", 'rb') as f:
        clf = pickle.load(f)
    with open(f"{MODELS}/threshold.pkl", 'rb') as f:
        threshold = pickle.load(f)
    print(f"  Model loaded. Threshold = {threshold:.3f}")

    # Load test data
    print("\nLoading test data …")
    test_s1 = pd.read_csv(f"{BASE_TEST}/test_source1.tsv", sep="\t", dtype=str)
    test_s2 = pd.read_csv(f"{BASE_TEST}/test_source2.tsv", sep="\t", dtype=str)
    test_s3 = pd.read_csv(f"{BASE_TEST}/test_source3.tsv", sep="\t", dtype=str)
    print(f"  Test S1: {len(test_s1):,}  S2: {len(test_s2):,}  S3: {len(test_s3):,}")

    # Build fresh indices on TEST S2+S3
    print("\nBuilding blocking indices on test S2+S3 …")
    name_idx, geo_idx, numeric_idx, s2_norm, s3_norm = build_indices(test_s2, test_s3)

    # Run inference
    matching_results, candidates = run_inference(
        clf, threshold, name_idx, city_idx,
        test_s1, s2_norm, s3_norm
    )

    # Write output
    write_output(matching_results, candidates,
                 test_s1['entity_id'].tolist(), OUTDIR)


if __name__ == '__main__':
    main()
