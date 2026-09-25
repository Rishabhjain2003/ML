# Entity Resolution Pipeline — Design Document

> **Status:** v1 complete (running), v2 implemented (pending next run)  
> **Metric:** Macro F_0.5 (precision-weighted, β = 0.5)  
> **Scale:** ~24M total records across 6 TSV files

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset](#2-dataset)
3. [High Level Design (HLD)](#3-high-level-design)
4. [Current Approach — v1](#4-current-approach--v1-3-path-blocking)
   - [Stage 1: Blocking](#41-stage-1-blocking)
   - [Stage 2: Classification](#42-stage-2-classification)
   - [Results](#43-results)
5. [Proposed Approach — v2](#5-proposed-approach--v2-5-path-blocking)
   - [New Paths](#51-new-blocking-paths)
   - [Parameter Changes](#52-parameter-changes)
6. [Low Level Design (LLD)](#6-low-level-design)
   - [Module Map](#61-module-map)
   - [Data Flow](#62-data-flow)
   - [Key Classes](#63-key-classes)
   - [Feature Vector](#64-feature-vector-15-dimensions)
   - [Index Persistence & Resume](#65-index-persistence--resume)
7. [Known Limitations & v3 Ideas](#7-known-limitations--v3-ideas)
8. [Running the Pipeline](#8-running-the-pipeline)

---

## 1. Problem Statement

Three independent business registries (S1, S2, S3) list the same real-world
companies, but with inconsistent names, addresses, scripts, and formats. The
goal is:

> **For every business entity in S1, identify all matching entities in S2 and S3.**

This is the **entity resolution** problem (also known as record linkage or
deduplication). Key challenges:

- **Scale** — S1 × (S2 + S3) = ~2.2M × 10M = ~22 trillion candidate pairs. Brute
  force comparison is impossible; a fast candidate-generation (blocking) step is
  mandatory.
- **Name variation** — "Apple Inc", "Apple Incorporated", "APPLE INC.", "एप्पल
  इंक" all refer to the same entity.
- **Script mixing** — A single business can appear in Devanagari (Hindi), Latin
  (English), or French in different sources.
- **Address inconsistency** — One source has a full street address, another has
  only a city and PIN code.
- **Multilingual data** — Dataset spans India (majority), France (~260K test
  entities), and the US.

**Evaluation metric: macro F_0.5**

F_0.5 is a precision-weighted F-score (β = 0.5). It weights precision 4× more
than recall, meaning a wrong prediction hurts significantly more than a missed
one. Score is averaged per S1 entity, then macro-averaged.

```
F_0.5 = (1 + 0.25) × (P × R) / (0.25 × P + R)
```

---

## 2. Dataset

| File | Split | Rows | Description |
|---|---|---|---|
| `train_source1.tsv` | Train | 2,206,821 | Query entities (S1) |
| `train_source2.tsv` | Train | 5,034,616 | Target entities (S2) |
| `train_source3.tsv` | Train | 5,285,603 | Target entities (S3) |
| `test_source1.tsv` | Test | 1,732,544 | Query entities (S1) |
| `test_source2.tsv` | Test | 4,887,273 | Target entities (S2) |
| `test_source3.tsv` | Test | 5,082,316 | Target entities (S3) |

**Schema (all files):**

```
entity_id   business_name   business_address   country
```

**Ground truth (train only):**

```
source1_entity_id   matched_entity_ids
S1-965667           S2-681193310,S2-743505751,S3-775321672
```

**Countries in test:** India (majority), France (~259K S1 entities), US (smaller
fraction).

---

## 3. High Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                              │
│   S1 (query)       S2 (target)       S3 (target)               │
│   2.2M entities    5M entities       5.3M entities             │
└────────┬────────────────┬──────────────────┬───────────────────┘
         │                └──────────────────┘
         │                         │
         │                   ┌─────▼──────────┐
         │                   │  NORMALIZATION  │
         │                   │ • lowercase     │
         │                   │ • strip suffix  │
         │                   │ • unicode NFC   │
         │                   │ • addr abbrevs  │
         │                   └─────────────────┘
         │                         │
         │                   ┌─────▼──────────────────────────────┐
         │                   │     INDEX BUILD  (S2 + S3 only)    │
         │                   │  A. Name inverted index (word tok) │
         │                   │  B. Geo index  (addr tokens)       │
         │                   │  C. Numeric index (addr numbers)   │
         │                   │  D. Trigram index (char 3-grams) ★ │
         │                   │  E. Postal code index            ★ │
         │                   └─────────────────────────────────────┘
         │                         │  (saved as .pkl)
         │                   ┌─────▼──────────────────────────────┐
         │                   │      BLOCKING (per S1 entity)      │
S1 ──────┼──────────────────►│  Query all 5 indices, union result │
         │                   │  ~100-150 candidates per entity     │
         │                   └─────────────────────────────────────┘
         │                         │
         │                   ┌─────▼──────────────────────────────┐
         │                   │    FEATURE EXTRACTION (per pair)   │
         │                   │  15 string-similarity features     │
         │                   │  (name + address + structural)     │
         │                   └─────────────────────────────────────┘
         │                         │
         │                   ┌─────▼──────────────────────────────┐
         │                   │    LightGBM CLASSIFIER             │
         │                   │  Binary: match / no-match          │
         │                   │  Threshold tuned for F_0.5         │
         │                   └─────────────────────────────────────┘
         │                         │
         └─────────────────────────►
                               OUTPUT
                   matching_results.tsv + candidate_pairs.tsv
```

★ = added in v2

**Two-phase execution:**

| Phase | Input | Output |
|---|---|---|
| **Train** | S1 + S2 + S3 + ground truth | Trained LightGBM model + threshold |
| **Inference** | Test S1 + S2 + S3 | `matching_results.tsv` |

Training uses a 30% sample of S1 (662K entities) to keep blocking tractable
while still seeing diverse examples.

---

## 4. Current Approach — v1 (3-path blocking)

### 4.1 Stage 1: Blocking

Blocking is the **recall gate** of the entire pipeline. Any true match not
retrieved here can never be recovered downstream. The goal is high recall (≥80%
ideally) with manageable candidate count (<200 per entity).

**v1 uses three retrieval paths on the S2+S3 combined index:**

#### Path A — Name Token Inverted Index

1. Normalize business name (see §6.3 normalization).
2. Tokenize into word tokens; remove a curated stop-list (`pvt`, `ltd`, `inc`,
   `co`, `corp`, `llc`, `sa`, `sarl`, `and`, `the`, etc.).
3. Build an inverted index: `token → [entity_id, ...]`.
4. Compute IDF for every token: `IDF(t) = log(N / df(t))`.
5. Prune tokens with `IDF < 1.5` (appear in > 22% of records — too common to
   discriminate).
6. At query time: sum IDF scores for shared tokens, return **top-30** by score.

**Strength:** Very fast. Directly targets the name field.  
**Weakness:** Requires exact word-token overlap. No tolerance for typos,
abbreviations, or different scripts.

#### Path B — Geographic Multi-key Index

1. Extract the last 4 non-numeric tokens from the normalized address.
2. Index each token independently under key `(country, token)`.
3. At query time: collect candidates from all matching `(country, token)` buckets,
   cap large buckets (>500 entries) at 50 to avoid noise, return up to 20 candidates.

**Strength:** Catches address-level locality matches across granularity levels
(city vs. district vs. state).  
**Weakness:** City names like "delhi", "mumbai", "paris" create enormous buckets
that get capped — true match may be cut off.

#### Path C — Numeric Address Token Index

1. Extract 2-6 digit sequences and alphanumeric codes (e.g. "a502") from address.
2. Index under `(country, number)`.
3. At query time: return up to 20 per numeric key; hard-cap large buckets at 10.

**Strength:** Language-independent. "Plot No 502" (English) and "प्लॉट नं 502"
(Hindi) both produce the token `502`.  
**Weakness:** Common numbers (100, 123, 500) create high-noise buckets.

**Union:** All three paths are unioned per S1 entity. Name-path results come
first (higher priority for downstream scoring). Result is a deduplicated list of
candidate entity IDs.

### 4.2 Stage 2: Classification

#### Feature Extraction (15 features per pair)

Every `(S1 entity, candidate)` pair is represented as a 15-dimensional vector
of string similarity scores over pre-normalized text:

| # | Feature | Method |
|---|---|---|
| 1 | `name_token_set_ratio` | rapidfuzz token set ratio ÷ 100 |
| 2 | `name_partial_ratio` | rapidfuzz partial ratio ÷ 100 |
| 3 | `name_jaro_winkler` | jellyfish Jaro-Winkler (capped at 100 chars) |
| 4 | `name_char3gram_jaccard` | Jaccard of character trigram sets |
| 5 | `name_char4gram_jaccard` | Jaccard of character 4-gram sets |
| 6 | `name_len_ratio` | min(len) / max(len) |
| 7 | `addr_token_set_ratio` | rapidfuzz token set ratio on address |
| 8 | `addr_partial_ratio` | rapidfuzz partial ratio on address |
| 9 | `addr_char3gram_jaccard` | Jaccard of address character trigrams |
| 10 | `addr_token_overlap` | Token Jaccard on address words |
| 11 | `city_exact_match` | 1 if last non-numeric address token matches |
| 12 | `numeric_token_match` | Jaccard of numeric tokens in address |
| 13 | `both_non_ascii` | 1 if both names have >20% non-ASCII characters |
| 14 | `country_match` | 1 if country fields match exactly |
| 15 | `name_city_product` | `name_token_set_ratio × city_exact_match` |

#### Model

**LightGBM binary classifier** with the following config:

```python
n_estimators     = 700
learning_rate    = 0.05
num_leaves       = 127
min_child_samples = 30
feature_fraction  = 0.8
bagging_fraction  = 0.8
bagging_freq      = 5
lambda_l1 = lambda_l2 = 0.1
early_stopping    = 50 rounds (on validation set)
```

**Training data construction:**

- Positives: (S1, S2/S3) pairs where S2/S3 appears in ground truth AND was
  retrieved by blocking. Label = 1.
- Negatives: All other blocking candidates (not in ground truth). Label = 0.
  Sampled at 8:1 negative-to-positive ratio.
- Group-aware train/val split (20% val) using `GroupShuffleSplit` on S1 entity
  ID — prevents data leakage from the same S1 entity appearing in both splits.

**Threshold tuning:**

After training, sweep threshold from 0.20 to 0.90 in steps of 0.025. Pick the
threshold maximising F_0.5 on the validation set.

### 4.3 Results (v1)

| Metric | Value |
|---|---|
| Blocking recall (train sample) | 61.7% |
| Avg candidates / S1 entity | 90.7 |
| Best threshold | 0.600 |
| Val F_0.5 | **0.7343** |

The 61.7% blocking recall is the hard ceiling. ~38% of true matches are
irretrievably lost before the classifier even runs.

---

## 5. Proposed Approach — v2 (5-path blocking)

v2 keeps Stage 2 (features + LightGBM) identical and adds two new retrieval
paths to the blocking stage, plus tunes two constants.

### 5.1 New Blocking Paths

#### Path D — Character Trigram Index

**Motivation:** Word-token overlap requires exact match. "Agarwal Traders" and
"Agrawal Traders" share zero word tokens after normalization (different
spellings), but share many character trigrams.

**Implementation:**

1. Collapse all whitespace in the normalized name: `"apple inc" → "appleinc"`.
2. Extract all character trigrams: `{app, ppl, ple, lei, ein, inc}`.
3. Build inverted index: `trigram → [entity_id, ...]` with IDF scoring.
4. Prune trigrams with `IDF < 2.0` (stricter than word tokens because the trigram
   space is much larger and common trigrams like `"the"`, `"inc"`, `"pvt"` are
   very high frequency).
5. At query time: score and return **top-20** by IDF-weighted trigram overlap.

**Catches:**
- Spelling variations: "Sharma" / "Sarma" / "Shrama"
- Abbreviations: "Intl" / "International" (share "int")
- Partial name matches: "Tata Steel" / "TATA STEEL LIMITED"

#### Path E — Postal/PIN Code Index

**Motivation:** A postal code is a near-perfect geographic anchor. Two businesses
sharing a 6-digit Indian PIN code (covering ~1-2 sq km) are almost certainly in
the same locality, regardless of how differently the rest of the address is written.

**Implementation:**

```
Country     Pattern               Example
---------   -------------------   --------
India       6-digit, first 1-9    400001 (Mumbai South)
France      5-digit, 01xxx-95xxx  75001 (Paris 1st arr.)
US          5-digit ZIP (± +4)    90210
Generic     4-6 digit sequences   —
```

Index under key `(country, postal_code)`. Postal buckets are inherently small
(at most a few thousand businesses per PIN code) so we allow up to **top-40**
candidates per bucket.

**Catches:**
- Addresses written at different granularity where the PIN code is the only
  shared structured field.
- French businesses where arrondissement or department is common but street
  address differs.

### 5.2 Parameter Changes

| Parameter | v1 | v2 | Reason |
|---|---|---|---|
| `TOP_K_NAME` | 30 | 60 | True match may rank 31st+ in name index |
| `MIN_IDF` | 1.5 | 1.0 | Keep tokens appearing in up to 37% of records (was 22%) |
| `MIN_TRIGRAM_IDF` | — | 2.0 | New; stricter because trigram space is larger |
| `TOP_K_TRIGRAM` | — | 20 | New path |
| `TOP_K_POSTAL` | — | 40 | New path; tight buckets allow higher cap |

**Expected impact:**

| Change | Expected recall lift |
|---|---|
| Char trigram index (Path D) | +5 – 10 pp |
| Postal code index (Path E) | +3 – 6 pp |
| TOP_K_NAME 30 → 60 | +2 – 4 pp |
| MIN_IDF 1.5 → 1.0 | +1 – 2 pp |
| **Combined (v2 estimate)** | **+8 – 15 pp → ~70 – 77% recall** |

Higher blocking recall → more true pairs reach the classifier → higher F_0.5.
Expected v2 Val F_0.5: **~0.80 – 0.85**.

---

## 6. Low Level Design

### 6.1 Module Map

```
MLChallenge/
├── code/
│   ├── pipeline.py          # Orchestrator: train phase + inference phase
│   └── src/
│       ├── normalize.py     # Text normalization (names + addresses)
│       ├── blocking.py      # All index classes + candidate generation
│       └── features.py      # 15-feature vector extraction
├── dataset/
│   ├── train/               # 3 TSV files + ground truth
│   └── test/                # 3 TSV files (no ground truth)
├── models/                  # Saved .pkl files (indices + model + threshold)
├── output/                  # matching_results.tsv, candidate_pairs.tsv
├── logs/                    # pipeline.log (append mode, tqdm updates)
└── utils/
    └── validate_submission.py
```

### 6.2 Data Flow

```
pipeline.py  main()
│
├── PHASE 1: TRAINING
│   │
│   ├── Load S1 (2.2M rows) + ground truth → normalize in-place
│   │
│   ├── build_indices_low_mem(S2, S3)
│   │   ├── Stream S2 row-by-row: build name/geo/numeric/trigram/postal indices
│   │   ├── del S2 dataframe → gc.collect()
│   │   ├── Stream S3 row-by-row: same
│   │   ├── del S3 dataframe → gc.collect()
│   │   ├── Compute IDF + prune low-IDF tokens (name + trigram)
│   │   └── Return 5 index objects + compact s23_lookup dict
│   │
│   ├── Save all 6 index files to models/ (name/geo/numeric/trigram/postal/lookup)
│   │
│   ├── Sample 30% of S1 (662K entities) for training blocking
│   │
│   ├── generate_candidates(s1_sample, all 5 indices)
│   │   └── Per S1 entity: query A+B+C+D+E, union, deduplicate
│   │
│   ├── evaluate_blocking_recall(candidates, ground_truth)  → logs recall
│   │
│   ├── Build (positive, negative) training pairs
│   │   ├── Positives: blocking retrieved the true match
│   │   └── Negatives: blocking retrieved a non-match (8:1 ratio)
│   │
│   ├── build_feature_matrix(pairs, s1_lookup, s23_lookup)
│   │   └── compute_features() × N pairs → (N × 15) float32 matrix
│   │
│   ├── GroupShuffleSplit → train 80% / val 20%
│   │
│   ├── LGBMClassifier.fit() with early stopping
│   │
│   ├── Threshold sweep [0.20 … 0.90 step 0.025] → best F_0.5 on val
│   │
│   └── Save lgbm_model.pkl + threshold.pkl
│
└── PHASE 2: INFERENCE
    │
    ├── Load test S1 (1.7M rows) → normalize
    │
    ├── build_indices_low_mem(test_S2, test_S3)  [or load from disk]
    │
    ├── generate_candidates(test_S1, all 5 test indices)
    │
    ├── score_candidates(clf, threshold, candidates, ...)
    │   ├── Flatten to list of (s1_id, cand_id) pairs
    │   ├── Batch-score in chunks of 100K pairs
    │   └── Filter pairs where P(match) ≥ threshold
    │
    └── Write matching_results.tsv + candidate_pairs.tsv
```

### 6.3 Key Classes

#### `normalize.py`

```python
normalize_name(text: str) -> str
```
Steps in order:
1. Strip DBA / f/k/a / AKA aliases (before any processing, while slashes intact)
2. Handle `.com` domain names — strip TLD
3. Lowercase
4. Unicode NFKD normalization — strip combining diacritics from Latin chars but
   **preserve** Indic scripts (Devanagari U+0900–U+097F, Bengali, Gujarati, etc.)
5. Normalize `&` → `and`
6. Legal suffix normalization via regex map:
   `incorporated→inc`, `corporation→corp`, `limited→ltd`, `société anonyme→sa`, etc.
7. Remove remaining punctuation (keep alphanumeric, spaces, hyphens, Indic)
8. Collapse whitespace

```python
normalize_address(text: str) -> str
```
Same pipeline with address-specific abbreviations:
`street→st`, `road→rd`, `avenue→ave`, `boulevard→blvd`, `nagar→ngr`, etc.

---

#### `blocking.py` — Index Classes

All five index classes share the same interface pattern:

```python
class SomeIndex:
    def build(self, entity_ids, ...) -> None   # build from S2+S3
    def query(self, ...) -> List[str]          # return candidate entity_ids
```

**NameInvertedIndex**
```
index: Dict[token_str, List[entity_id]]
idf:   Dict[token_str, float]
query: IDF-weighted counter → most_common(TOP_K_NAME=60)
```

**GeoIndex**
```
index: Dict[(country, geo_token), List[entity_id]]
query: collect from each _geo_tokens() result, cap large buckets
```

**NumericIndex**
```
index: Dict[(country, num_str), List[entity_id]]
query: collect from each _numeric_tokens() result, hard-cap >200-entry buckets
```

**CharTrigramIndex** *(v2)*
```
index: Dict[trigram_str, List[entity_id]]
idf:   Dict[trigram_str, float]
query: IDF-weighted counter → most_common(TOP_K_TRIGRAM=20)
prune: MIN_TRIGRAM_IDF = 2.0 (stricter than word tokens)
```

**PostalCodeIndex** *(v2)*
```
index: Dict[(country, postal_code), List[entity_id]]
query: _postal_codes(addr, country) → collect from matching buckets
country-aware patterns: India 6-digit, France 5-digit, US 5-digit, generic 4-6
```

---

#### `generate_candidates()` execution per S1 entity

```python
# Pseudocode per entity
name_cands  = name_idx.query(name_norm, top_k=60)       # Path A
tgram_cands = trigram_idx.query(name_norm, top_k=20)    # Path D
geo_cands   = geo_idx.query(addr_norm, country)         # Path B
num_cands   = numeric_idx.query(addr_norm, country)     # Path C
postal_cands = postal_idx.query(addr_norm, country)     # Path E

# Union with deduplication; name-path first (priority for scoring)
merged = deduplicate(name_cands + tgram_cands + geo_cands
                     + num_cands + postal_cands)
```

### 6.4 Feature Vector (15 dimensions)

```
Dims 0–5   Name features
  0  name_token_set_ratio   — rapidfuzz.token_set_ratio / 100
  1  name_partial_ratio     — rapidfuzz.partial_ratio / 100
  2  name_jaro_winkler      — jellyfish.jaro_winkler_similarity (first 100 chars)
  3  name_char3gram_jaccard — |trigrams(n1) ∩ trigrams(n2)| / |union|
  4  name_char4gram_jaccard — same for 4-grams
  5  name_len_ratio         — min(len(n1), len(n2)) / max(len(n1), len(n2))

Dims 6–9   Address features
  6  addr_token_set_ratio   — rapidfuzz.token_set_ratio on normalized address
  7  addr_partial_ratio     — rapidfuzz.partial_ratio on address
  8  addr_char3gram_jaccard — character trigram Jaccard on address
  9  addr_token_overlap     — token Jaccard: |T(a1) ∩ T(a2)| / |T(a1) ∪ T(a2)|

Dims 10–13 Structural features
  10 city_exact_match       — 1.0 if last non-numeric addr token matches exactly
  11 numeric_token_match    — Jaccard of address numeric tokens; 0.5 if both empty
  12 both_non_ascii         — 1.0 if both names >20% non-ASCII (cross-script signal)
  13 country_match          — 1.0 if country field matches exactly

Dim 14     Interaction feature
  14 name_city_product      — name_token_set_ratio × city_exact_match
```

**Why these features:**
- `token_set_ratio` handles word-order differences ("Tata Steel" vs "Steel Tata")
- `partial_ratio` handles one name being a substring of the other
- `jaro_winkler` gives prefix-weighted edit distance (good for company names that
  often share prefixes)
- `char ngrams` on both name and address catch spelling variation not handled by
  word-level metrics
- `numeric_token_match` is language-independent — building numbers survive script
  changes
- `name_city_product` is a high-precision interaction: only high if BOTH the name
  AND city match, which is a very strong match signal

### 6.5 Index Persistence & Resume

All index objects are serialized to disk with `pickle.dump(protocol=4)` after
build. On the next run, the pipeline checks for the existence of all 6 files:

**v1 (3-path):**
```
models/name_idx.pkl, geo_idx.pkl, numeric_idx.pkl, s23_lookup.pkl
```

**v2 (5-path) — file registry:**
```
Training:   models/{name,geo,numeric,trigram,postal}_idx.pkl  +  s23_lookup.pkl
Test:  models/test_{name,geo,numeric,trigram,postal}_idx.pkl  +  test_s23_lookup.pkl
```

If all files exist → load from disk (resume, skip the 2-3 hour build step).
If any file is missing → rebuild all from scratch and save.

> **Note for v2:** Since `trigram_idx.pkl` and `postal_idx.pkl` are new filenames,
> the v2 run will unconditionally rebuild all indices even if the v1 files are
> present. The old files will be overwritten.

---

## 7. Known Limitations & v3 Ideas

### Hard limit: Transliteration gap

The largest unresolved miss category: a business name appears in Devanagari in S1
and in Latin script in S2/S3 (or vice versa). None of the five blocking paths
can bridge this — word tokens differ, trigrams differ, geo and numeric may both
be absent.

**v3 option A — Transliteration library:**
Use `indic-transliteration` or `aksharamukha` to generate a Latin equivalent of
every non-ASCII normalized name, index both the original and the transliterated
form. Estimated recall lift: +3–5 pp for Indian data.

**v3 option B — Multilingual dense embeddings:**
Use LaBSE (Language-agnostic BERT Sentence Embeddings) or a similar model to
embed all names into a shared vector space. Use FAISS approximate nearest-
neighbour search for blocking instead of inverted index. This would catch script-
bridging pairs by semantic similarity. Significantly more compute-intensive
(GPU recommended) but potentially largest single recall improvement.

### Other gaps

| Gap | Notes |
|---|---|
| Address-only matches | Some S1 entities have NULL/generic names; only address blocking finds them |
| France-specific names | "Société" → "sa" normalization exists; French stopwords not fully covered |
| One-to-many matches | Some S1 entities match 5+ S2/S3 records; classifier threshold is global |
| Feature completeness | No phonetic features (Soundex/Metaphone); limited French name handling |

---

## 8. Running the Pipeline

**Requirements:** Python 3.10+, venv at `~/MLChallenge/venv`

```bash
# First time setup (venv already created)
cd ~/MLChallenge
venv/bin/pip install lightgbm rapidfuzz jellyfish pandas numpy scikit-learn tqdm

# Run (screen + caffeinate keeps it alive if Mac sleeps or terminal closes)
screen -dmS mlpipeline caffeinate -i \
  venv/bin/python3 -u code/pipeline.py

# Monitor
tail -f logs/pipeline.log           # live log
screen -r mlpipeline                # attach to tqdm progress bar
# Ctrl-A D to detach from screen

# Resume (indices already built — skips the 2-3 hour build step)
# Just rerun the same command; pipeline detects saved .pkl files and loads them.

# Validate submission
venv/bin/python3 utils/validate_submission.py \
  --matching  output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir  dataset/test
```

**Estimated runtimes (Apple M-series, 8 cores):**

| Step | v1 | v2 (estimate) |
|---|---|---|
| Index build (train) | ~2–3 h | ~3–4 h (+trigram/postal) |
| Training blocking (662K S1) | ~2.5 h | ~3–4 h (more candidates) |
| LightGBM training | ~20 min | ~25 min |
| Index build (test) | ~2–3 h | ~3–4 h |
| Test blocking (1.7M S1) | ~10.5 h | ~13–16 h |
| Scoring | ~1.5 h | ~2–3 h |
| **Total** | **~20 h** | **~25–32 h** |
