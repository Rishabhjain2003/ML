"""
Full training pipeline.
Samples 30% of S1 for training to manage scale (~660K entities × ~80 candidates).
"""
import os, sys, pickle, random
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize_name, normalize_address
from blocking import build_indices, generate_candidates, evaluate_blocking_recall
from features import build_feature_matrix, FEATURE_NAMES

SEED = 42; random.seed(SEED); np.random.seed(SEED)
BASE   = "/Users/arvshaa/MLChallenge/dataset/train"
OUTDIR = "/Users/arvshaa/MLChallenge/models"
os.makedirs(OUTDIR, exist_ok=True)

TRAIN_S1_FRACTION = 0.30
NEG_TO_POS_RATIO  = 8
VAL_FRACTION      = 0.20


def load_data():
    print("Loading …")
    s1 = pd.read_csv(f"{BASE}/train_source1.tsv", sep="\t", dtype=str)
    s2 = pd.read_csv(f"{BASE}/train_source2.tsv", sep="\t", dtype=str)
    s3 = pd.read_csv(f"{BASE}/train_source3.tsv", sep="\t", dtype=str)
    gt = pd.read_csv(f"{BASE}/train_ground_truth.tsv", sep="\t", dtype=str)
    s1['name_norm'] = s1['business_name'].map(normalize_name)
    s1['addr_norm'] = s1['business_address'].map(normalize_address)
    print(f"  S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}")
    return s1, s2, s3, gt


def build_ground_truth_map(gt):
    gt_map = {}
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        if pd.notna(row['matched_entity_ids']) and str(row['matched_entity_ids']).strip():
            gt_map[s1_id] = set(str(row['matched_entity_ids']).split(','))
        else:
            gt_map[s1_id] = set()
    return gt_map


def compute_f05_macro(preds, gt_map, s1_ids):
    scores = []
    for s1_id in s1_ids:
        true_set = gt_map.get(s1_id, set())
        pred_set = set(preds.get(s1_id, []))
        if not true_set and not pred_set:
            scores.append(1.0); continue
        if not true_set:
            scores.append(0.0); continue
        if not pred_set:
            scores.append(0.0); continue
        tp = len(true_set & pred_set)
        p  = tp / len(pred_set)
        r  = tp / len(true_set)
        scores.append((1.25*p*r)/(0.25*p+r) if p+r > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def tune_threshold(val_probs_by_s1, gt_map, val_s1_ids):
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.20, 0.90, 0.025):
        preds = {s1: [eid for eid, p in pairs if p >= t]
                 for s1, pairs in val_probs_by_s1.items()}
        f05 = compute_f05_macro(preds, gt_map, val_s1_ids)
        if f05 > best_f:
            best_f, best_t = f05, t
    print(f"  Best threshold={best_t:.3f}  val F_0.5={best_f:.4f}")
    return best_t


def main():
    s1, s2, s3, gt = load_data()
    gt_map = build_ground_truth_map(gt)

    # Build indices on full S2+S3
    print("\nBuilding blocking indices …")
    name_idx, geo_idx, numeric_idx, s2_norm, s3_norm = build_indices(s2, s3)

    # Lookups
    s23_lookup = {}
    for _, row in s2_norm.iterrows(): s23_lookup[row['entity_id']] = row.to_dict()
    for _, row in s3_norm.iterrows(): s23_lookup[row['entity_id']] = row.to_dict()
    s1_lookup  = {row['entity_id']: row.to_dict() for _, row in s1.iterrows()}

    # Sample S1
    print(f"\nSampling {TRAIN_S1_FRACTION*100:.0f}% of S1 …")
    all_s1_ids = s1['entity_id'].tolist()
    random.shuffle(all_s1_ids)
    n_sample = int(len(all_s1_ids) * TRAIN_S1_FRACTION)
    sampled_ids = set(all_s1_ids[:n_sample])
    s1_sample = s1[s1['entity_id'].isin(sampled_ids)].reset_index(drop=True)
    print(f"  Using {len(s1_sample):,} S1 entities")

    # Generate candidates
    candidates = generate_candidates(s1_sample, name_idx, geo_idx, numeric_idx)
    gt_sample  = gt[gt['source1_entity_id'].isin(sampled_ids)]
    evaluate_blocking_recall(candidates, gt_sample)

    # Build pairs
    print("\nBuilding training pairs …")
    positives, negatives = [], []
    true_pair_set = set()

    for s1_id, cands in candidates.items():
        true_matches = gt_map.get(s1_id, set())
        cand_set = set(cands)
        for m in true_matches:
            if m in cand_set:
                positives.append((s1_id, m, 1))
                true_pair_set.add((s1_id, m))
        for c in cands:
            if (s1_id, c) not in true_pair_set:
                negatives.append((s1_id, c, 0))

    print(f"  Positives: {len(positives):,}  Hard negatives: {len(negatives):,}")
    random.shuffle(negatives)
    neg_sample = negatives[:len(positives) * NEG_TO_POS_RATIO]
    all_pairs  = positives + neg_sample
    print(f"  Total pairs: {len(all_pairs):,}")

    # Feature matrix
    print("\nComputing features …")
    X, y = build_feature_matrix(all_pairs, s1_lookup, s23_lookup)
    print(f"  X={X.shape}  positives={int(y.sum()):,}")

    # Train/val split (by S1 entity group)
    s1_groups = np.array([p[0] for p in all_pairs[:len(X)]])
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
    tr_idx, val_idx = next(gss.split(X, y, groups=s1_groups))
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    val_s1_ids = list(set(s1_groups[val_idx]))
    print(f"  Train={len(tr_idx):,}  Val={len(val_idx):,}  Val-S1={len(val_s1_ids):,}")

    # LightGBM
    print("\nTraining LightGBM …")
    clf = lgb.LGBMClassifier(
        n_estimators=700, learning_rate=0.05, num_leaves=127,
        min_child_samples=30, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=5,
        lambda_l1=0.1, lambda_l2=0.1,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    clf.fit(X_tr, y_tr,
            eval_X=X_val, eval_y=y_val,
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(100)])

    # Threshold tuning
    print("\nTuning threshold …")
    val_pairs = [all_pairs[i] for i in val_idx]
    val_probs_raw = clf.predict_proba(X_val)[:, 1]
    val_probs_by_s1 = defaultdict(list)
    for (s1_id, other_id, _), prob in zip(val_pairs, val_probs_raw):
        val_probs_by_s1[s1_id].append((other_id, prob))
    best_thresh = tune_threshold(val_probs_by_s1, gt_map, val_s1_ids)

    # Feature importances
    print("\nTop feature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, clf.feature_importances_),
                            key=lambda x: -x[1])[:10]:
        print(f"  {name:30s}: {imp}")

    # Save
    print(f"\nSaving to {OUTDIR} …")
    with open(f"{OUTDIR}/lgbm_model.pkl", 'wb') as f: pickle.dump(clf, f)
    with open(f"{OUTDIR}/threshold.pkl",  'wb') as f: pickle.dump(best_thresh, f)
    print("Saved.")
    return clf, best_thresh


if __name__ == '__main__':
    main()
