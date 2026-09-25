"""
Candidate generation (blocking) — 5 parallel retrieval paths:

  A) Word-token inverted index on normalized name (IDF-filtered, top-60)
  B) Multi-key geographic blocking: last 4 non-numeric address tokens
  C) Address numeric token blocking: plot/flat/unit numbers
  D) Character trigram index on normalized name (NEW — handles typos/abbreviations)
  E) Postal/PIN code index on address (NEW — high-precision geographic anchor)

Union all five paths per S1 entity.
"""
import re
import math
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

from normalize import (normalize_name, normalize_address,
                       extract_city_tokens, extract_first_city_token,
                       is_non_ascii_name)

# ── Constants ─────────────────────────────────────────────────────────────────
TOP_K_NAME     = 60   # raised from 30 — more candidates = better recall
TOP_K_GEO      = 20
TOP_K_NUMERIC  = 20
TOP_K_TRIGRAM  = 20   # top candidates from trigram index
TOP_K_POSTAL   = 40   # postal buckets are tight → allow more
MIN_IDF        = 1.0  # relaxed from 1.5 — keep more moderately-common tokens
MIN_TRIGRAM_IDF = 2.0  # stricter for trigrams (vast space, prune only very common)
MIN_TOKEN_LEN  = 2

_STOP = {
    'pvt', 'ltd', 'inc', 'co', 'corp', 'llc', 'llp', 'pvtltd', 'pllc', 'lp',
    'sa', 'sas', 'sarl', 'and', 'the', 'of', 'a', 'an', 'by', 'at', 'in',
    'private', 'limited', 'incorporated', 'company', 'associates',
}


def _meaningful_tokens(name_norm: str) -> List[str]:
    return [t for t in name_norm.split()
            if len(t) >= MIN_TOKEN_LEN and t not in _STOP]


def _char_trigrams(name_norm: str) -> List[str]:
    """
    Character trigrams of normalized name with spaces collapsed.
    "apple inc" → {app, ppl, ple, lei, ein, inc}
    Robust to typos, abbreviations, and partial name matches.
    """
    s = re.sub(r'\s+', '', name_norm)   # collapse all whitespace
    if len(s) < 3:
        return []
    return [s[i:i+3] for i in range(len(s) - 2)]


def _geo_tokens(addr_norm: str, n_last: int = 3) -> List[str]:
    if not addr_norm:
        return []
    tokens = [t for t in addr_norm.split()
              if not t.isdigit() and len(t) > 1]
    geo = tokens[-n_last:] if len(tokens) >= n_last else tokens
    return [t for t in geo if len(t) > 2]


def _numeric_tokens(addr_norm: str) -> List[str]:
    if not addr_norm:
        return []
    addr_clean = re.sub(r'[/\\]', '', addr_norm)
    nums = re.findall(r'\b[a-z]?\d{2,6}\b', addr_clean)
    return [n for n in nums if len(n) >= 2]


def _postal_codes(addr_norm: str, country: str) -> List[str]:
    """
    Extract postal/PIN codes from normalized address by country.
    These are high-precision geographic anchors — same postal code almost
    guarantees same locality, regardless of how the rest of address is written.
    """
    if not addr_norm:
        return []
    c = str(country).lower()
    if 'india' in c or c == 'in':
        # Indian PIN codes: 6 digits, first digit 1-9
        codes = re.findall(r'\b[1-9]\d{5}\b', addr_norm)
    elif 'france' in c or c == 'fr':
        # French postal: 5 digits, first two are department (01-95 + DOM-TOM)
        codes = re.findall(r'\b(?:0[1-9]|[1-9]\d)\d{3}\b', addr_norm)
    elif 'us' in c or 'united states' in c or 'usa' in c:
        # US ZIP: 5 digits (avoid matching any random 5-digit number with context)
        codes = re.findall(r'\b\d{5}(?:-\d{4})?\b', addr_norm)
    else:
        # Generic: 4-6 digit sequences that could be postal codes
        codes = re.findall(r'\b[1-9]\d{3,5}\b', addr_norm)
    # De-duplicate and exclude obviously non-postal numbers
    return list(dict.fromkeys(c for c in codes if len(c) >= 4))


# ── Name inverted index ───────────────────────────────────────────────────────
class NameInvertedIndex:
    def __init__(self):
        self.index:  Dict[str, List[str]] = defaultdict(list)
        self.idf:    Dict[str, float]     = {}
        self.n_docs: int = 0

    def build(self, entity_ids: List[str], name_norms: List[str]) -> None:
        print(f"  Building name index on {len(entity_ids):,} records …")
        self.n_docs = len(entity_ids)
        df_count: Counter = Counter()

        for eid, name in tqdm(zip(entity_ids, name_norms), total=len(entity_ids),
                              desc="  name-idx", mininterval=5):
            tokens = set(_meaningful_tokens(name))
            for t in tokens:
                self.index[t].append(eid)
                df_count[t] += 1

        N = self.n_docs
        self.idf = {t: math.log(N / df) for t, df in df_count.items()}
        prune = [t for t, v in self.idf.items() if v < MIN_IDF]
        for t in prune:
            del self.index[t]
            del self.idf[t]
        print(f"  Name index: {len(self.index):,} tokens kept  "
              f"(pruned {len(prune):,} low-IDF)")

    def query(self, name_norm: str, top_k: int = TOP_K_NAME) -> List[Tuple[str, float]]:
        tokens = _meaningful_tokens(name_norm)
        scores: Counter = Counter()
        for t in tokens:
            if t in self.index:
                w = self.idf[t]
                for eid in self.index[t]:
                    scores[eid] += w
        return scores.most_common(top_k)


# ── Character trigram index ───────────────────────────────────────────────────
class CharTrigramIndex:
    """
    Inverted index on character trigrams of normalized names.
    Catches: typos, abbreviations, partial matches, and near-duplicate names
    that share no full word tokens but overlap in character sequences.
    e.g. "Agarwal" vs "Agrawal" → share {agr, wal} etc.
    """
    def __init__(self):
        self.index:  Dict[str, List[str]] = defaultdict(list)
        self.idf:    Dict[str, float]     = {}
        self.n_docs: int = 0

    def build(self, entity_ids: List[str], name_norms: List[str]) -> None:
        print(f"  Building trigram index on {len(entity_ids):,} records …")
        self.n_docs = len(entity_ids)
        df_count: Counter = Counter()

        for eid, name in tqdm(zip(entity_ids, name_norms), total=len(entity_ids),
                              desc="  tri-idx", mininterval=5):
            tgrams = set(_char_trigrams(name))
            for t in tgrams:
                self.index[t].append(eid)
                df_count[t] += 1

        N = self.n_docs
        self.idf = {t: math.log(N / df) for t, df in df_count.items()}
        prune = [t for t, v in self.idf.items() if v < MIN_TRIGRAM_IDF]
        for t in prune:
            del self.index[t]
            del self.idf[t]
        print(f"  Trigram index: {len(self.index):,} trigrams kept  "
              f"(pruned {len(prune):,} low-IDF)")

    def query(self, name_norm: str,
              top_k: int = TOP_K_TRIGRAM) -> List[Tuple[str, float]]:
        tgrams = _char_trigrams(name_norm)
        scores: Counter = Counter()
        for t in tgrams:
            if t in self.index:
                w = self.idf[t]
                for eid in self.index[t]:
                    scores[eid] += w
        return scores.most_common(top_k)


# ── Geographic (multi-key) index ──────────────────────────────────────────────
class GeoIndex:
    def __init__(self):
        self.index: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    def build(self, entity_ids: List[str],
              addr_norms: List[str],
              countries: List[str]) -> None:
        print(f"  Building geo index on {len(entity_ids):,} records …")
        for eid, addr, cty in zip(entity_ids, addr_norms, countries):
            for tok in _geo_tokens(addr, n_last=4):
                if len(tok) >= 2:
                    key = (str(cty).lower(), tok)
                    self.index[key].append(eid)
        print(f"  Geo index: {len(self.index):,} unique (country, geo-token) buckets")

    def query(self, addr_norm: str, country: str,
              max_k: int = TOP_K_GEO) -> List[str]:
        cty = str(country).lower()
        results: Set[str] = set()
        for tok in _geo_tokens(addr_norm, n_last=4):
            if len(tok) >= 2:
                key = (cty, tok)
                bucket = self.index.get(key, [])
                cap = min(max_k, 50) if len(bucket) > 500 else max_k
                results.update(bucket[:cap])
        return list(results)


# ── Numeric address token index ───────────────────────────────────────────────
class NumericIndex:
    def __init__(self):
        self.index: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    def build(self, entity_ids: List[str],
              addr_norms: List[str],
              countries: List[str]) -> None:
        print(f"  Building numeric index on {len(entity_ids):,} records …")
        for eid, addr, cty in zip(entity_ids, addr_norms, countries):
            for num in _numeric_tokens(addr):
                key = (str(cty).lower(), num)
                self.index[key].append(eid)
        print(f"  Numeric index: {len(self.index):,} unique (country, num) buckets")

    def query(self, addr_norm: str, country: str,
              max_k: int = TOP_K_NUMERIC) -> List[str]:
        if not addr_norm:
            return []
        cty = str(country).lower()
        results: Set[str] = set()
        for num in _numeric_tokens(addr_norm):
            key = (cty, num)
            bucket = self.index.get(key, [])
            cap = min(max_k, 10) if len(bucket) > 200 else max_k
            results.update(bucket[:cap])
        return list(results)


# ── Postal code index ─────────────────────────────────────────────────────────
class PostalCodeIndex:
    """
    Index S2/S3 by postal/PIN code. Very tight, high-precision buckets.
    Two businesses in the same 6-digit Indian PIN or 5-digit French postal code
    are almost certainly in the same locality — a strong blocking signal.
    """
    def __init__(self):
        self.index: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    def build(self, entity_ids: List[str],
              addr_norms: List[str],
              countries: List[str]) -> None:
        print(f"  Building postal index on {len(entity_ids):,} records …")
        found = 0
        for eid, addr, cty in zip(entity_ids, addr_norms, countries):
            for code in _postal_codes(addr, cty):
                key = (str(cty).lower(), code)
                self.index[key].append(eid)
                found += 1
        print(f"  Postal index: {len(self.index):,} unique (country, postal) buckets "
              f"({found:,} entries)")

    def query(self, addr_norm: str, country: str,
              max_k: int = TOP_K_POSTAL) -> List[str]:
        cty = str(country).lower()
        results: Set[str] = set()
        for code in _postal_codes(addr_norm, cty):
            key = (cty, code)
            bucket = self.index.get(key, [])
            # Postal buckets are inherently small; only cap truly giant ones
            cap = min(max_k, 100) if len(bucket) > 1000 else max_k
            results.update(bucket[:cap])
        return list(results)


# ── Build all indices ─────────────────────────────────────────────────────────
def build_indices(s2: pd.DataFrame, s3: pd.DataFrame):
    s23 = pd.concat([s2, s3], ignore_index=True)
    s23['name_norm'] = s23['business_name'].map(normalize_name)
    s23['addr_norm'] = s23['business_address'].map(normalize_address)
    print(f"Preprocessed {len(s23):,} S2+S3 records")

    name_idx    = NameInvertedIndex()
    geo_idx     = GeoIndex()
    numeric_idx = NumericIndex()
    trigram_idx = CharTrigramIndex()
    postal_idx  = PostalCodeIndex()

    ids   = s23['entity_id'].tolist()
    names = s23['name_norm'].tolist()
    addrs = s23['addr_norm'].tolist()
    ctys  = s23['country'].tolist()

    name_idx.build(ids, names)
    geo_idx.build(ids, addrs, ctys)
    numeric_idx.build(ids, addrs, ctys)
    trigram_idx.build(ids, names)
    postal_idx.build(ids, addrs, ctys)

    s2_norm = s2.copy()
    s2_norm['name_norm'] = s2_norm['business_name'].map(normalize_name)
    s2_norm['addr_norm'] = s2_norm['business_address'].map(normalize_address)
    s3_norm = s3.copy()
    s3_norm['name_norm'] = s3_norm['business_name'].map(normalize_name)
    s3_norm['addr_norm'] = s3_norm['business_address'].map(normalize_address)

    return name_idx, geo_idx, numeric_idx, trigram_idx, postal_idx, s2_norm, s3_norm


# ── Generate candidates ───────────────────────────────────────────────────────
def generate_candidates(s1: pd.DataFrame,
                        name_idx: NameInvertedIndex,
                        geo_idx: GeoIndex,
                        numeric_idx: NumericIndex,
                        trigram_idx: 'CharTrigramIndex | None' = None,
                        postal_idx:  'PostalCodeIndex | None'  = None,
                        top_k_name: int = TOP_K_NAME) -> Dict[str, List[str]]:

    s1 = s1.copy()
    s1['name_norm'] = s1['business_name'].map(normalize_name)
    s1['addr_norm'] = s1['business_address'].map(normalize_address)

    candidates: Dict[str, List[str]] = {}
    print(f"Generating candidates for {len(s1):,} S1 entities …")

    for _, row in tqdm(s1.iterrows(), total=len(s1), desc="blocking", mininterval=10):
        s1_id = row['entity_id']

        # Path A: word-token name (IDF-scored, top-60)
        name_cands = [eid for eid, _ in
                      name_idx.query(row['name_norm'], top_k=top_k_name)]

        # Path B: geographic tokens (multi-level)
        geo_cands = geo_idx.query(row['addr_norm'], row['country'])

        # Path C: address numeric tokens
        num_cands = numeric_idx.query(row['addr_norm'], row['country'])

        # Path D: character trigrams (typo/abbreviation tolerance)
        tgram_cands = ([eid for eid, _ in trigram_idx.query(row['name_norm'])]
                       if trigram_idx is not None else [])

        # Path E: postal/PIN code (high-precision geo anchor)
        postal_cands = (postal_idx.query(row['addr_norm'], row['country'])
                        if postal_idx is not None else [])

        # Union, deduplicate (name-based first = higher priority scoring)
        seen: Set[str] = set()
        merged = []
        for eid in name_cands + tgram_cands + geo_cands + num_cands + postal_cands:
            if eid not in seen:
                seen.add(eid)
                merged.append(eid)

        candidates[s1_id] = merged

    return candidates


# ── Recall evaluation (training only) ────────────────────────────────────────
def evaluate_blocking_recall(candidates: Dict[str, List[str]],
                              ground_truth: pd.DataFrame) -> None:
    total_true, total_found = 0, 0
    missed_examples = []

    gt_map = {}
    for _, row in ground_truth.iterrows():
        s1_id = row['source1_entity_id']
        if pd.notna(row['matched_entity_ids']) and str(row['matched_entity_ids']).strip():
            gt_map[s1_id] = set(str(row['matched_entity_ids']).split(','))

    for s1_id, true_matches in gt_map.items():
        cands = set(candidates.get(s1_id, []))
        found = true_matches & cands
        total_true  += len(true_matches)
        total_found += len(found)
        if len(found) < len(true_matches) and len(missed_examples) < 5:
            missed_examples.append((s1_id, true_matches - cands))

    recall    = total_found / max(total_true, 1)
    avg_cands = sum(len(v) for v in candidates.values()) / max(len(candidates), 1)

    print(f"\n=== Blocking Recall ===")
    print(f"  Covered : {total_found:,} / {total_true:,}  (recall={recall:.4f})")
    print(f"  Avg candidates/S1 entity: {avg_cands:.1f}")
    if missed_examples:
        print("  Missed examples (sample):")
        for s1_id, missed in missed_examples:
            print(f"    {s1_id}  missed={list(missed)[:3]}")
