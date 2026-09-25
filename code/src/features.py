"""
Feature extraction for candidate pairs.
All features are string-based (fast, no dense embeddings required at scale).
15 features per pair.
"""
import re
import math
from typing import List, Dict, Tuple

import numpy as np
from rapidfuzz import fuzz
import jellyfish

from normalize import extract_numeric_tokens, is_non_ascii_name, DOMAIN_RE

FEATURE_NAMES = [
    # Name features (6)
    'name_token_set_ratio',
    'name_partial_ratio',
    'name_jaro_winkler',
    'name_char3gram_jaccard',
    'name_char4gram_jaccard',
    'name_len_ratio',
    # Address features (4)
    'addr_token_set_ratio',
    'addr_partial_ratio',
    'addr_char3gram_jaccard',
    'addr_token_overlap',
    # Structural (4)
    'city_exact_match',
    'numeric_token_match',
    'both_non_ascii',
    'country_match',
    # Interaction (1)
    'name_city_product',
]


def _char_ngrams(s: str, n: int) -> set:
    return set(s[i:i+n] for i in range(max(len(s)-n+1, 0)))


def _jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _token_overlap(a_norm: str, b_norm: str) -> float:
    ta = set(a_norm.split())
    tb = set(b_norm.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _city_token(addr_norm: str) -> str:
    tokens = [t for t in addr_norm.split() if not t.isdigit() and len(t) > 1]
    return tokens[-1] if tokens else ''


def compute_features(s1_name: str, s1_addr: str, s1_country: str,
                     s2_name: str, s2_addr: str, s2_country: str) -> List[float]:
    """Compute feature vector for one candidate pair (both inputs already normalized)."""

    # ── Name features ──────────────────────────────────────────────────────
    # Handle empty names
    n1, n2 = s1_name or '', s2_name or ''

    if n1 and n2:
        name_tsr   = fuzz.token_set_ratio(n1, n2)   / 100.0
        name_pr    = fuzz.partial_ratio(n1, n2)      / 100.0
        name_jw    = jellyfish.jaro_winkler_similarity(n1[:100], n2[:100])
        name_j3    = _jaccard(_char_ngrams(n1, 3), _char_ngrams(n2, 3))
        name_j4    = _jaccard(_char_ngrams(n1, 4), _char_ngrams(n2, 4))
        name_lr    = min(len(n1), len(n2)) / max(len(n1), len(n2), 1)
    else:
        name_tsr = name_pr = name_jw = name_j3 = name_j4 = name_lr = 0.0

    # ── Address features ───────────────────────────────────────────────────
    a1, a2 = s1_addr or '', s2_addr or ''

    if a1 and a2:
        addr_tsr  = fuzz.token_set_ratio(a1, a2) / 100.0
        addr_pr   = fuzz.partial_ratio(a1, a2)   / 100.0
        addr_j3   = _jaccard(_char_ngrams(a1, 3), _char_ngrams(a2, 3))
        addr_to   = _token_overlap(a1, a2)
    else:
        addr_tsr = addr_pr = addr_j3 = addr_to = 0.0

    # ── Structural features ────────────────────────────────────────────────
    # City token exact match
    c1 = _city_token(a1)
    c2 = _city_token(a2)
    city_match = 1.0 if (c1 and c2 and c1 == c2) else 0.0

    # Numeric tokens in address (building numbers)
    nums1 = extract_numeric_tokens(a1)
    nums2 = extract_numeric_tokens(a2)
    if nums1 or nums2:
        num_match = len(nums1 & nums2) / max(len(nums1 | nums2), 1)
    else:
        num_match = 0.5  # neither has numbers: neutral

    # Both names are non-ASCII (cross-script pairs — usually both Hindi or both French)
    both_non_ascii = 1.0 if (is_non_ascii_name(n1) and is_non_ascii_name(n2)) else 0.0

    # Country match
    country_match = 1.0 if (str(s1_country).lower() == str(s2_country).lower()) else 0.0

    # ── Interaction feature ────────────────────────────────────────────────
    name_city_product = name_tsr * city_match

    return [
        name_tsr, name_pr, name_jw, name_j3, name_j4, name_lr,
        addr_tsr, addr_pr, addr_j3, addr_to,
        city_match, num_match, both_non_ascii, country_match,
        name_city_product,
    ]


def build_feature_matrix(pairs: List[Tuple],
                          s1_lookup: Dict,
                          s23_lookup: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    pairs: list of (s1_id, other_id, label)
    Returns X (n_pairs × 15), y (n_pairs,)
    """
    X, y = [], []
    for s1_id, other_id, label in pairs:
        r1 = s1_lookup.get(s1_id)
        r2 = s23_lookup.get(other_id)
        if r1 is None or r2 is None:
            continue
        feats = compute_features(
            r1.get('name_norm',''), r1.get('addr_norm',''), r1.get('country',''),
            r2.get('name_norm',''), r2.get('addr_norm',''), r2.get('country',''),
        )
        X.append(feats)
        y.append(float(label))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


if __name__ == '__main__':
    # Quick sanity check
    from normalize import normalize_name, normalize_address

    pairs = [
        # True match: name variation + address match
        ("Continental Resources Pvt Ltd",
         "Plot No C-26 G/F, Shiv Vihar, Delhi", "India",
         "CONTINENTAL RESOURCES PVT PVT",
         "PLOT NO C-26 G/F KH 11/10, SHIV VIHAR MATIYALA, DELHI", "India",
         1),
        # True match: .com domain
        ("General Electronics Partners",
         "1681 Big Branch Road, Clyde, NC", "US",
         "generalelectronicspartners.com",
         "1681D BIG BRANCH RD, CYDE, NC", "US",
         1),
        # False: different business, same address area
        ("Acme Bakery LLC",
         "500 Market St, San Jose, CA", "US",
         "Acme Robotics Inc",
         "500 Market Street, San Jose CA", "US",
         0),
        # True match: Hindi transliteration
        ("One Smart Producer Private Limited",
         "Office No A/502, Navi Mumbai", "India",
         "वन स्मार्ट प्रोड्यूसर प्राइवेट लिमिटेड",
         "OFFNCE NO A/502, THANE, Maharashtra", "India",
         1),
    ]
    print("=== Feature sanity check ===\n")
    for p in pairs:
        n1,a1,c1,n2,a2,c2,label = p
        nn1,na1 = normalize_name(n1), normalize_address(a1)
        nn2,na2 = normalize_name(n2), normalize_address(a2)
        f = compute_features(nn1,na1,c1,nn2,na2,c2)
        print(f"  label={label}  name_tsr={f[0]:.2f}  jaro={f[2]:.2f}  "
              f"addr_tsr={f[6]:.2f}  city={f[10]:.0f}  num={f[11]:.2f}  "
              f"non_ascii={f[12]:.0f}  score_proxy={f[0]*0.6+f[6]*0.2+f[10]*0.2:.2f}")
        print(f"    {nn1!r}  vs  {nn2!r}")
        print()
