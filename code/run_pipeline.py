"""
Full end-to-end pipeline:
  1. Train on 30% of S1 training data
  2. Run inference on full test set
  3. Write output/matching_results.tsv + output/candidate_pairs.tsv
  4. Validate output

Usage:
    python run_pipeline.py [--train-only] [--infer-only]
"""
import os, sys, pickle, random, argparse
from collections import defaultdict
import numpy as np, pandas as pd
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from normalize import normalize_name, normalize_address
from blocking import build_indices, generate_candidates, evaluate_blocking_recall
from features import build_feature_matrix, compute_features, FEATURE_NAMES
from train import build_ground_truth_map, compute_f05_macro, tune_threshold

SEED = 42; random.seed(SEED); np.random.seed(SEED)

ROOT     = "/Users/arvshaa/MLChallenge"
TRAIN_DIR = f"{ROOT}/dataset/train"
TEST_DIR  = f"{ROOT}/dataset/test"
MODEL_DIR = f"{ROOT}/models"
OUT_DIR   = f"{ROOT}/output"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR,   exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_FRACTION   = 0.30   # fraction of S1 to use for training
NEG_RATIO        = 8      # negatives per positive
VAL_FRACTION     = 0.20   # of training sample for internal validation
BATCH_SIZE       = 50_000 # pairs per feature-computation batch


def run_training():
    print("\n" + "="*60)
    print("PHASE 1: TRAINING")
    print("="*60)

    # Load
    print("Loading training data …")
    s1 = pd.read_csv(f"{TRAIN_DIR}/train_source1.tsv", sep="\t", dtype=str)
    s2 = pd.read_csv(f"{TRAIN_DIR}/train_source2.tsv", sep="\t", dtype=str)
    s3 = pd.read_csv(f"{TRAIN_DIR}/train_source3.tsv", sep="\t", dtype=str)
    gt = pd.read_csv(f"{TRAIN_DIR}/train_ground_truth.tsv", sep="\t", dtype=str)
    s1['name_norm'] = s1['business_name'].map(normalize_name)
    s1['addr_norm'] = s1['business_address'].map(normalize_address)
    print(f"  S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}")

    gt_map = build_ground_truth_map(gt)

    # Build blocking indices on FULL S2+S3
    print("\nBuilding blocking indices on full S2+S3 …")
    name_idx, geo_idx, numeric_idx, s2_norm, s3_norm = build_indices(s2, s3)

    # Save indices for reuse in inference
    print("Saving indices …")
    with open(f"{MODEL_DIR}/name_idx.pkl",    'wb') as f: pickle.dump(name_idx, f)
    with open(f"{MODEL_DIR}/geo_idx.pkl",     'wb') as f: pickle.dump(geo_idx, f)
    with open(f"{MODEL_DIR}/numeric_idx.pkl", 'wb') as f: pickle.dump(numeric_idx, f)

    # Lookups for feature computation
    print("Building lookups …")
    s23_lookup = {}
    for _, row in tqdm(s2_norm.iterrows(), total=len(s2_norm), desc="s2 lookup"):
        s23_lookup[row['entity_id']] = row.to_dict()
    for _, row in tqdm(s3_norm.iterrows(), total=len(s3_norm), desc="s3 lookup"):
        s23_lookup[row['entity_id']] = row.to_dict()
    s1_lookup = {row['entity_id']: row.to_dict() for _, row in s1.iterrows()}

    # Sample S1 for training
    print(f"\nSampling {TRAIN_FRACTION*100:.0f}% of S1 …")
    all_ids = s1['entity_id'].tolist()
    random.shuffle(all_ids)
    n_sample = int(len(all_ids) * TRAIN_FRACTION)
    sampled  = set(all_ids[:n_sample])
    s1_train = s1[s1['entity_id'].isin(sampled)].reset_index(drop=True)
    print(f"  Training on {len(s1_train):,} S1 entities")

    # Generate candidates
    print("\nGenerating training candidates …")
    candidates = generate_candidates(s1_train, name_idx, geo_idx, numeric_idx)
    gt_train   = gt[gt['source1_entity_id'].isin(sampled)]
    evaluate_blocking_recall(candidates, gt_train)

    # Build pairs
    print("\nBuilding training pairs …")
    positives, negatives = [], []
    true_pair_set = set()
    for s1_id, cands in candidates.items():
        tm  = gt_map.get(s1_id, set())
        cset = set(cands)
        for m in tm:
            if m in cset:
                positives.append((s1_id, m, 1))
                true_pair_set.add((s1_id, m))
        for c in cands:
            if (s1_id, c) not in true_pair_set:
                negatives.append((s1_id, c, 0))

    print(f"  Positives={len(positives):,}  Negatives={len(negatives):,}")
    random.shuffle(negatives)
    neg_s   = negatives[:len(positives)*NEG_RATIO]
    all_pairs = positives + neg_s
    print(f"  Total pairs={len(all_pairs):,}")

    # Feature matrix
    print("\nComputing features …")
    X, y = build_feature_matrix(all_pairs, s1_lookup, s23_lookup)
    print(f"  X={X.shape}  positives={int(y.sum()):,}")

    # Train/val split
    groups   = np.array([p[0] for p in all_pairs[:len(X)]])
    gss      = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
    tr_i,vl_i = next(gss.split(X, y, groups=groups))
    X_tr,y_tr = X[tr_i], y[tr_i]
    X_vl,y_vl = X[vl_i], y[vl_i]
    val_s1_ids = list(set(groups[vl_i]))
    print(f"  Train={len(tr_i):,}  Val={len(vl_i):,}  Val-S1={len(val_s1_ids):,}")

    # Train LightGBM
    print("\nTraining LightGBM …")
    clf = lgb.LGBMClassifier(
        n_estimators=700, learning_rate=0.05, num_leaves=127,
        min_child_samples=30, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=5,
        lambda_l1=0.1, lambda_l2=0.1,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    clf.fit(X_tr, y_tr, eval_X=X_vl, eval_y=y_vl,
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(100)])

    # Threshold tuning
    print("\nTuning threshold …")
    val_pairs_list = [all_pairs[i] for i in vl_i]
    probs_raw = clf.predict_proba(X_vl)[:, 1]
    probs_by_s1 = defaultdict(list)
    for (s1_id, other_id, _), p in zip(val_pairs_list, probs_raw):
        probs_by_s1[s1_id].append((other_id, p))
    best_thresh = tune_threshold(dict(probs_by_s1), gt_map, val_s1_ids)

    # Feature importances
    print("\nTop features:")
    for n, i in sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x:-x[1])[:8]:
        print(f"  {n:30s}: {i}")

    # Save model + threshold
    with open(f"{MODEL_DIR}/lgbm_model.pkl",  'wb') as f: pickle.dump(clf, f)
    with open(f"{MODEL_DIR}/threshold.pkl",   'wb') as f: pickle.dump(best_thresh, f)
    print(f"\nModel saved to {MODEL_DIR}")
    return clf, best_thresh


def run_inference():
    print("\n" + "="*60)
    print("PHASE 2: INFERENCE")
    print("="*60)

    # Load model
    print("Loading model …")
    with open(f"{MODEL_DIR}/lgbm_model.pkl", 'rb') as f: clf = pickle.load(f)
    with open(f"{MODEL_DIR}/threshold.pkl",  'rb') as f: threshold = pickle.load(f)
    print(f"  Threshold={threshold:.3f}")

    # Load test data
    print("Loading test data …")
    ts1 = pd.read_csv(f"{TEST_DIR}/test_source1.tsv", sep="\t", dtype=str)
    ts2 = pd.read_csv(f"{TEST_DIR}/test_source2.tsv", sep="\t", dtype=str)
    ts3 = pd.read_csv(f"{TEST_DIR}/test_source3.tsv", sep="\t", dtype=str)
    print(f"  S1={len(ts1):,}  S2={len(ts2):,}  S3={len(ts3):,}")
    print(f"  Country dist: {ts1['country'].value_counts().to_dict()}")

    # Build fresh test indices
    print("\nBuilding blocking indices on test S2+S3 …")
    name_idx, geo_idx, numeric_idx, s2_norm, s3_norm = build_indices(ts2, ts3)

    # Lookups
    print("Building lookups …")
    s23_lookup = {}
    for _, row in tqdm(s2_norm.iterrows(), total=len(s2_norm), desc="s2"):
        s23_lookup[row['entity_id']] = row.to_dict()
    for _, row in tqdm(s3_norm.iterrows(), total=len(s3_norm), desc="s3"):
        s23_lookup[row['entity_id']] = row.to_dict()

    ts1['name_norm'] = ts1['business_name'].map(normalize_name)
    ts1['addr_norm'] = ts1['business_address'].map(normalize_address)
    s1_lookup = {row['entity_id']: row.to_dict() for _, row in ts1.iterrows()}

    # Generate test candidates
    print("\nGenerating test candidates …")
    candidates = generate_candidates(ts1, name_idx, geo_idx, numeric_idx)
    total_cands = sum(len(v) for v in candidates.values())
    print(f"  Total candidate pairs: {total_cands:,}")

    # Score pairs in batches
    print(f"\nScoring pairs in batches of {BATCH_SIZE:,} …")
    all_s1_ids = ts1['entity_id'].tolist()
    pair_list  = []
    for s1_id in all_s1_ids:
        for cid in candidates.get(s1_id, []):
            pair_list.append((s1_id, cid))

    all_probs = []
    for start in tqdm(range(0, len(pair_list), BATCH_SIZE),
                      desc="scoring", mininterval=30):
        batch = pair_list[start:start+BATCH_SIZE]
        X_b = []
        for s1_id, cid in batch:
            r1 = s1_lookup.get(s1_id, {})
            r2 = s23_lookup.get(cid, {})
            X_b.append(compute_features(
                r1.get('name_norm',''), r1.get('addr_norm',''), r1.get('country',''),
                r2.get('name_norm',''), r2.get('addr_norm',''), r2.get('country',''),
            ))
        all_probs.extend(clf.predict_proba(np.array(X_b, dtype=np.float32))[:,1])

    # Apply threshold
    probs_by_s1 = defaultdict(list)
    for (s1_id, cid), p in zip(pair_list, all_probs):
        probs_by_s1[s1_id].append((cid, p))

    matching = {}
    for s1_id in all_s1_ids:
        pairs = probs_by_s1.get(s1_id, [])
        matching[s1_id] = [cid for cid, p in pairs if p >= threshold]

    # Write output
    print(f"\nWriting output to {OUT_DIR} …")
    mr_path = f"{OUT_DIR}/matching_results.tsv"
    cp_path = f"{OUT_DIR}/candidate_pairs.tsv"

    with open(mr_path, 'w') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matched = list(dict.fromkeys(
                m for m in matching.get(s1_id, [])
                if m.startswith('S2-') or m.startswith('S3-')
            ))
            f.write(f"{s1_id}\t{','.join(matched)}\n")

    with open(cp_path, 'w') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = list(dict.fromkeys(
                c for c in candidates.get(s1_id, [])
                if c.startswith('S2-') or c.startswith('S3-')
            ))
            f.write(f"{s1_id}\t{','.join(cands)}\n")

    # Stats
    n_match  = sum(1 for v in matching.values() if v)
    n_single = sum(1 for v in matching.values() if not v)
    avg_m    = np.mean([len(v) for v in matching.values() if v]) if n_match else 0
    print(f"\n  Entities with matches: {n_match:,}")
    print(f"  Singletons:            {n_single:,}")
    print(f"  Avg matches/entity:    {avg_m:.2f}")
    print(f"\nOutput files written:")
    print(f"  {mr_path}")
    print(f"  {cp_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-only', action='store_true')
    parser.add_argument('--infer-only', action='store_true')
    args = parser.parse_args()

    if not args.infer_only:
        run_training()
    if not args.train_only:
        run_inference()

    # Validate
    print("\nRunning validator …")
    os.system(f"/Users/arvshaa/MLChallenge/venv/bin/python3 "
              f"{ROOT}/utils/validate_submission.py "
              f"--matching {OUT_DIR}/matching_results.tsv "
              f"--candidate {OUT_DIR}/candidate_pairs.tsv "
              f"--test-dir {TEST_DIR}")
