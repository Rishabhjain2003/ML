"""
Memory-efficient end-to-end pipeline.
5-path blocking: name tokens + char trigrams + geo + numeric + postal codes.
Estimated peak RAM: ~8-10 GB.

Run from Mac Terminal:
    cd ~/MLChallenge
    venv/bin/python3 code/pipeline.py
"""
import os, sys, gc, pickle, random, time
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from normalize import normalize_name, normalize_address
from blocking import (NameInvertedIndex, GeoIndex, NumericIndex,
                      CharTrigramIndex, PostalCodeIndex,
                      generate_candidates, evaluate_blocking_recall,
                      _meaningful_tokens, _geo_tokens, _numeric_tokens,
                      _char_trigrams, _postal_codes, MIN_IDF, MIN_TRIGRAM_IDF)
from features import compute_features, build_feature_matrix, FEATURE_NAMES

import io as _io, math as _math
from collections import Counter as _Counter

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try: s.write(data)
            except: pass
    def flush(self):
        for s in self.streams:
            try: s.flush()
            except: pass
    def isatty(self): return False

_logfile = open('/Users/arvshaa/MLChallenge/logs/pipeline.log', 'a', buffering=1)
sys.stdout = _Tee(sys.__stdout__, _logfile)
sys.stderr = _Tee(sys.__stderr__, _logfile)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = os.path.expanduser("~/MLChallenge")
TRAIN_DIR = f"{ROOT}/dataset/train"
TEST_DIR  = f"{ROOT}/dataset/test"
MODEL_DIR = f"{ROOT}/models"
OUT_DIR   = f"{ROOT}/output"
for d in [MODEL_DIR, OUT_DIR]: os.makedirs(d, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
SEED             = 42
TRAIN_FRACTION   = 0.30
NEG_RATIO        = 8
VAL_FRACTION     = 0.20
SCORE_BATCH      = 100_000

random.seed(SEED)
np.random.seed(SEED)


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_gt_map(gt: pd.DataFrame) -> dict:
    m = {}
    for _, row in gt.iterrows():
        sid = row['source1_entity_id']
        val = row.get('matched_entity_ids', '')
        if pd.notna(val) and str(val).strip():
            m[sid] = set(str(val).split(','))
        else:
            m[sid] = set()
    return m


def f05(preds: dict, gt_map: dict, s1_ids: list) -> float:
    scores = []
    for sid in s1_ids:
        true = gt_map.get(sid, set())
        pred = set(preds.get(sid, []))
        if not true and not pred:  scores.append(1.0); continue
        if not true:               scores.append(0.0); continue
        if not pred:               scores.append(0.0); continue
        tp = len(true & pred)
        p, r = tp / len(pred), tp / len(true)
        scores.append((1.25*p*r)/(0.25*p+r) if p+r else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def build_indices_low_mem(s2_path: str, s3_path: str):
    """
    Build all 5 indices from S2+S3 in two streaming passes (one per source file).
    Releases each dataframe immediately after indexing to conserve RAM.
    Returns: (name_idx, geo_idx, numeric_idx, trigram_idx, postal_idx, compact_lookup)
    """
    name_idx    = NameInvertedIndex()
    geo_idx     = GeoIndex()
    numeric_idx = NumericIndex()
    trigram_idx = CharTrigramIndex()
    postal_idx  = PostalCodeIndex()
    compact     = {}   # entity_id → {name_norm, addr_norm, country}

    # Initialise name/trigram IDF counters (shared across both files)
    name_df_count    = _Counter()
    trigram_df_count = _Counter()
    total_records    = 0

    for path, label in [(s2_path, 'S2'), (s3_path, 'S3')]:
        print(f"  Loading {label} …", flush=True)
        df = pd.read_csv(path, sep='\t', dtype=str)
        df['name_norm'] = df['business_name'].map(normalize_name)
        df['addr_norm'] = df['business_address'].map(normalize_address)

        ids   = df['entity_id'].tolist()
        names = df['name_norm'].tolist()
        addrs = df['addr_norm'].tolist()
        ctys  = df['country'].tolist()
        total_records += len(ids)

        print(f"  Indexing {label} ({len(df):,} records) …", flush=True)
        for eid, name, addr, cty in tqdm(zip(ids, names, addrs, ctys),
                                          total=len(ids), desc=f"  {label}-idx",
                                          mininterval=10):
            # Path A: name word tokens
            toks = set(_meaningful_tokens(name))
            for tok in toks:
                name_idx.index[tok].append(eid)
                name_df_count[tok] += 1

            # Path B: geo tokens
            for tok in _geo_tokens(addr, n_last=4):
                if len(tok) >= 2:
                    geo_idx.index[(str(cty).lower(), tok)].append(eid)

            # Path C: numeric tokens
            for num in _numeric_tokens(addr):
                numeric_idx.index[(str(cty).lower(), num)].append(eid)

            # Path D: character trigrams
            tgrams = set(_char_trigrams(name))
            for tg in tgrams:
                trigram_idx.index[tg].append(eid)
                trigram_df_count[tg] += 1

            # Path E: postal codes
            for code in _postal_codes(addr, cty):
                postal_idx.index[(str(cty).lower(), code)].append(eid)

            compact[eid] = {'name_norm': name, 'addr_norm': addr, 'country': cty}

        del df; gc.collect()

    # Compute and store IDF, prune low-IDF tokens
    N = total_records
    name_idx.n_docs    = N
    trigram_idx.n_docs = N

    name_idx.idf = {tok: _math.log(N / df) for tok, df in name_df_count.items()}
    name_prune   = [t for t, v in name_idx.idf.items() if v < MIN_IDF]
    for t in name_prune:
        del name_idx.index[t]; del name_idx.idf[t]

    trigram_idx.idf = {tg: _math.log(N / df) for tg, df in trigram_df_count.items()}
    tg_prune        = [t for t, v in trigram_idx.idf.items() if v < MIN_TRIGRAM_IDF]
    for t in tg_prune:
        del trigram_idx.index[t]; del trigram_idx.idf[t]

    print(f"  Name index  : {len(name_idx.index):,} tokens   (pruned {len(name_prune):,})")
    print(f"  Trigram idx : {len(trigram_idx.index):,} trigrams (pruned {len(tg_prune):,})")
    print(f"  Geo index   : {len(geo_idx.index):,} buckets")
    print(f"  Numeric idx : {len(numeric_idx.index):,} buckets")
    print(f"  Postal idx  : {len(postal_idx.index):,} buckets")
    print(f"  Lookup      : {len(compact):,} records")
    return name_idx, geo_idx, numeric_idx, trigram_idx, postal_idx, compact


def score_candidates(clf, threshold, candidates, s1_lookup, s23_lookup, all_s1_ids):
    pair_list = [(s1, cid) for s1 in all_s1_ids for cid in candidates.get(s1, [])]
    print(f"  Scoring {len(pair_list):,} candidate pairs …", flush=True)

    all_probs = []
    for start in tqdm(range(0, len(pair_list), SCORE_BATCH),
                      desc="  scoring", mininterval=30):
        batch = pair_list[start:start+SCORE_BATCH]
        X_b = []
        for s1id, cid in batch:
            r1 = s1_lookup.get(s1id, {})
            r2 = s23_lookup.get(cid, {})
            X_b.append(compute_features(
                r1.get('name_norm',''), r1.get('addr_norm',''), r1.get('country',''),
                r2.get('name_norm',''), r2.get('addr_norm',''), r2.get('country',''),
            ))
        all_probs.extend(clf.predict_proba(np.array(X_b, dtype=np.float32))[:,1])

    probs_by_s1 = defaultdict(list)
    for (s1id, cid), p in zip(pair_list, all_probs):
        probs_by_s1[s1id].append((cid, p))

    results = {}
    for s1id in all_s1_ids:
        results[s1id] = [cid for cid, p in probs_by_s1.get(s1id, []) if p >= threshold]
    return results


# ── Index file registry (training + test) ────────────────────────────────────
TRAIN_IDX_FILES = {
    'name':    f"{MODEL_DIR}/name_idx.pkl",
    'geo':     f"{MODEL_DIR}/geo_idx.pkl",
    'numeric': f"{MODEL_DIR}/numeric_idx.pkl",
    'trigram': f"{MODEL_DIR}/trigram_idx.pkl",
    'postal':  f"{MODEL_DIR}/postal_idx.pkl",
    'lookup':  f"{MODEL_DIR}/s23_lookup.pkl",
}
TEST_IDX_FILES = {
    'name':    f"{MODEL_DIR}/test_name_idx.pkl",
    'geo':     f"{MODEL_DIR}/test_geo_idx.pkl",
    'numeric': f"{MODEL_DIR}/test_numeric_idx.pkl",
    'trigram': f"{MODEL_DIR}/test_trigram_idx.pkl",
    'postal':  f"{MODEL_DIR}/test_postal_idx.pkl",
    'lookup':  f"{MODEL_DIR}/test_s23_lookup.pkl",
}

def _save_indices(files, name_idx, geo_idx, numeric_idx, trigram_idx, postal_idx, lookup):
    print("Saving indices …", flush=True)
    with open(files['name'],    'wb') as f: pickle.dump(name_idx,    f, protocol=4)
    with open(files['geo'],     'wb') as f: pickle.dump(geo_idx,     f, protocol=4)
    with open(files['numeric'], 'wb') as f: pickle.dump(numeric_idx, f, protocol=4)
    with open(files['trigram'], 'wb') as f: pickle.dump(trigram_idx, f, protocol=4)
    with open(files['postal'],  'wb') as f: pickle.dump(postal_idx,  f, protocol=4)
    with open(files['lookup'],  'wb') as f: pickle.dump(lookup,      f, protocol=4)

def _load_indices(files):
    print("Resuming: loading saved indices from disk …", flush=True)
    with open(files['name'],    'rb') as f: name_idx    = pickle.load(f)
    with open(files['geo'],     'rb') as f: geo_idx     = pickle.load(f)
    with open(files['numeric'], 'rb') as f: numeric_idx = pickle.load(f)
    with open(files['trigram'], 'rb') as f: trigram_idx = pickle.load(f)
    with open(files['postal'],  'rb') as f: postal_idx  = pickle.load(f)
    with open(files['lookup'],  'rb') as f: lookup      = pickle.load(f)
    print(f"  Loaded. lookup={len(lookup):,} records", flush=True)
    return name_idx, geo_idx, numeric_idx, trigram_idx, postal_idx, lookup


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()

    # ── PHASE 1: TRAIN ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 1: TRAINING")
    print("="*60)

    print("Loading S1 + ground truth …", flush=True)
    s1 = pd.read_csv(f"{TRAIN_DIR}/train_source1.tsv", sep='\t', dtype=str)
    gt = pd.read_csv(f"{TRAIN_DIR}/train_ground_truth.tsv", sep='\t', dtype=str)
    s1['name_norm'] = s1['business_name'].map(normalize_name)
    s1['addr_norm'] = s1['business_address'].map(normalize_address)
    gt_map = build_gt_map(gt)
    print(f"  S1={len(s1):,}  GT={len(gt):,}", flush=True)
    del gt; gc.collect()

    # Resume if all 6 training index files exist
    if all(os.path.exists(f) for f in TRAIN_IDX_FILES.values()):
        name_idx, geo_idx, numeric_idx, trigram_idx, postal_idx, s23_lookup = \
            _load_indices(TRAIN_IDX_FILES)
    else:
        print("\nBuilding blocking indices (low-memory) …", flush=True)
        name_idx, geo_idx, numeric_idx, trigram_idx, postal_idx, s23_lookup = \
            build_indices_low_mem(
                f"{TRAIN_DIR}/train_source2.tsv",
                f"{TRAIN_DIR}/train_source3.tsv"
            )
        gc.collect()
        _save_indices(TRAIN_IDX_FILES, name_idx, geo_idx, numeric_idx,
                      trigram_idx, postal_idx, s23_lookup)

    s1_lookup = {row['entity_id']: row.to_dict() for _, row in s1.iterrows()}

    # Sample S1
    all_ids = s1['entity_id'].tolist()
    random.shuffle(all_ids)
    sampled = set(all_ids[:int(len(all_ids)*TRAIN_FRACTION)])
    s1_train = s1[s1['entity_id'].isin(sampled)].reset_index(drop=True)
    print(f"\nSampled {len(s1_train):,} S1 for training", flush=True)

    # Generate candidates (all 5 paths)
    print("Generating training candidates …", flush=True)
    train_cands = generate_candidates(
        s1_train, name_idx, geo_idx, numeric_idx, trigram_idx, postal_idx)

    # Blocking recall
    gt_reload = pd.read_csv(f"{TRAIN_DIR}/train_ground_truth.tsv", sep='\t', dtype=str)
    gt_train  = gt_reload[gt_reload['source1_entity_id'].isin(sampled)]
    evaluate_blocking_recall(train_cands, gt_train)
    del gt_reload; gc.collect()

    # Build pairs
    print("\nBuilding training pairs …", flush=True)
    positives, negatives, seen = [], [], set()
    for s1id, cands in train_cands.items():
        tm = gt_map.get(s1id, set())
        cs = set(cands)
        for m in tm:
            if m in cs:
                positives.append((s1id, m, 1)); seen.add((s1id, m))
        for c in cands:
            if (s1id, c) not in seen:
                negatives.append((s1id, c, 0))
    random.shuffle(negatives)
    neg_s     = negatives[:len(positives)*NEG_RATIO]
    all_pairs = positives + neg_s
    print(f"  Pos={len(positives):,}  Neg(sampled)={len(neg_s):,}  Total={len(all_pairs):,}", flush=True)
    del train_cands, negatives; gc.collect()

    # Features
    print("Computing features …", flush=True)
    X, y = build_feature_matrix(all_pairs, s1_lookup, s23_lookup)
    print(f"  X={X.shape}  pos={int(y.sum()):,}", flush=True)

    # Split
    groups = np.array([p[0] for p in all_pairs[:len(X)]])
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
    tr_i, vl_i = next(gss.split(X, y, groups=groups))
    X_tr, y_tr = X[tr_i], y[tr_i]
    X_vl, y_vl = X[vl_i], y[vl_i]
    val_ids    = list(set(groups[vl_i]))
    print(f"  Train={len(tr_i):,}  Val={len(vl_i):,}  Val-S1={len(val_ids):,}", flush=True)

    # Train
    print("\nTraining LightGBM …", flush=True)
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
    del X, X_tr, y_tr, X_vl, y_vl; gc.collect()

    # Threshold tuning
    print("\nTuning threshold …", flush=True)
    val_pairs = [all_pairs[i] for i in vl_i]
    vp_raw    = clf.predict_proba(np.array([
        compute_features(
            s1_lookup.get(s,{}).get('name_norm',''), s1_lookup.get(s,{}).get('addr_norm',''),
            s1_lookup.get(s,{}).get('country',''),
            s23_lookup.get(c,{}).get('name_norm',''), s23_lookup.get(c,{}).get('addr_norm',''),
            s23_lookup.get(c,{}).get('country',''))
        for s,c,_ in val_pairs], dtype=np.float32))[:,1]
    probs_vp = defaultdict(list)
    for (sid,cid,_), p in zip(val_pairs, vp_raw):
        probs_vp[sid].append((cid, p))

    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.20, 0.90, 0.025):
        preds = {s:[c for c,p in pairs if p>=t] for s,pairs in probs_vp.items()}
        fv = f05(preds, gt_map, val_ids)
        if fv > best_f: best_f, best_t = fv, t
    print(f"  Best threshold={best_t:.3f}  Val F_0.5={best_f:.4f}", flush=True)

    # Feature importances
    print("\nTop features:")
    for n,i in sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x:-x[1])[:8]:
        print(f"  {n:30s}: {i}")

    with open(f"{MODEL_DIR}/lgbm_model.pkl", 'wb') as f: pickle.dump(clf, f, protocol=4)
    with open(f"{MODEL_DIR}/threshold.pkl",  'wb') as f: pickle.dump(best_t, f)
    del all_pairs, val_pairs, probs_vp, s23_lookup; gc.collect()
    print(f"\nPhase 1 done in {(time.time()-t0)/60:.1f} min", flush=True)

    # ── PHASE 2: INFERENCE ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 2: INFERENCE")
    print("="*60)

    print("Loading test data …", flush=True)
    ts1 = pd.read_csv(f"{TEST_DIR}/test_source1.tsv", sep='\t', dtype=str)
    ts1['name_norm'] = ts1['business_name'].map(normalize_name)
    ts1['addr_norm'] = ts1['business_address'].map(normalize_address)
    print(f"  Test S1={len(ts1):,}  countries: {ts1['country'].value_counts().to_dict()}", flush=True)

    if all(os.path.exists(f) for f in TEST_IDX_FILES.values()):
        t_name_idx, t_geo_idx, t_num_idx, t_tgram_idx, t_postal_idx, t_s23_lookup = \
            _load_indices(TEST_IDX_FILES)
    else:
        print("\nBuilding test blocking indices …", flush=True)
        t_name_idx, t_geo_idx, t_num_idx, t_tgram_idx, t_postal_idx, t_s23_lookup = \
            build_indices_low_mem(
                f"{TEST_DIR}/test_source2.tsv",
                f"{TEST_DIR}/test_source3.tsv"
            )
        gc.collect()
        _save_indices(TEST_IDX_FILES, t_name_idx, t_geo_idx, t_num_idx,
                      t_tgram_idx, t_postal_idx, t_s23_lookup)

    ts1_lookup = {row['entity_id']: row.to_dict() for _, row in ts1.iterrows()}

    print("\nGenerating test candidates …", flush=True)
    test_cands = generate_candidates(
        ts1, t_name_idx, t_geo_idx, t_num_idx, t_tgram_idx, t_postal_idx)
    total_cands = sum(len(v) for v in test_cands.values())
    print(f"  Total={total_cands:,}  Avg={total_cands/max(len(test_cands),1):.1f}/entity", flush=True)

    print("\nScoring …", flush=True)
    all_s1_ids = ts1['entity_id'].tolist()
    results    = score_candidates(clf, best_t, test_cands, ts1_lookup, t_s23_lookup, all_s1_ids)

    # ── Write output ──────────────────────────────────────────────────────────
    print("\nWriting output …", flush=True)
    with open(f"{OUT_DIR}/matching_results.tsv", 'w') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in all_s1_ids:
            matched = list(dict.fromkeys(
                m for m in results.get(sid, [])
                if m.startswith('S2-') or m.startswith('S3-')
            ))
            f.write(f"{sid}\t{','.join(matched)}\n")

    with open(f"{OUT_DIR}/candidate_pairs.tsv", 'w') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in all_s1_ids:
            cands = list(dict.fromkeys(
                c for c in test_cands.get(sid, [])
                if c.startswith('S2-') or c.startswith('S3-')
            ))
            f.write(f"{sid}\t{','.join(cands)}\n")

    n_match = sum(1 for v in results.values() if v)
    avg_m   = np.mean([len(v) for v in results.values() if v]) if n_match else 0
    print(f"\n  Entities with matches : {n_match:,}")
    print(f"  Singletons            : {len(results)-n_match:,}")
    print(f"  Avg matches/entity    : {avg_m:.2f}")
    print(f"\nOutput: {OUT_DIR}/matching_results.tsv")
    print(f"        {OUT_DIR}/candidate_pairs.tsv")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min", flush=True)

    # ── Validate ──────────────────────────────────────────────────────────────
    print("\nValidating …")
    os.system(
        f"'{sys.executable}' '{ROOT}/utils/validate_submission.py' "
        f"--matching '{OUT_DIR}/matching_results.tsv' "
        f"--candidate '{OUT_DIR}/candidate_pairs.tsv' "
        f"--test-dir '{TEST_DIR}'"
    )


if __name__ == '__main__':
    main()
