"""
Text normalization for business names and addresses.
Handles: ALL CAPS, punctuation, legal suffixes, domain names, NULL/nan strings,
         Devanagari/Tamil/Gujarati scripts (preserved for multilingual embedding),
         f/k/a, DBA, .com domain names.
"""
import re
import unicodedata

# ── Legal suffix normalization ────────────────────────────────────────────────
SUFFIX_MAP = [
    (r'\bincorporated\b', 'inc'),
    (r'\bcorporation\b',  'corp'),
    (r'\blimited\b',      'ltd'),
    (r'\bprivate limited\b', 'pvtltd'),
    (r'\bprivate\b',      'pvt'),
    (r'\bcompany\b',      'co'),
    (r'\bllimited\b',     'ltd'),
    (r'\bltd\b',          'ltd'),
    (r'\binc\b',          'inc'),
    (r'\bcorp\b',         'corp'),
    (r'\bpvt\b',          'pvt'),
    (r'\bpvtltd\b',       'pvtltd'),
    (r'\bsoci.t. anonyme\b',   'sa'),
    (r'\bsoci.t. par actions simplifi.e\b', 'sas'),
    (r'\bsoci.t. . responsabilit. limit.e\b', 'sarl'),
]

ADDR_MAP = [
    (r'\bstreet\b',    'st'),
    (r'\broad\b',      'rd'),
    (r'\bavenue\b',    'ave'),
    (r'\bboulevard\b', 'blvd'),
    (r'\blane\b',      'ln'),
    (r'\bdrive\b',     'dr'),
    (r'\bnagar\b',     'ngr'),
    (r'\bsector\b',    'sec'),
    (r'\bnear\b',      'nr'),
    (r'\bopposite\b',  'opp'),
    (r'\bbuilding\b',  'bldg'),
    (r'\bhighway\b',   'hwy'),
    (r'\broute\b',     'rte'),
    (r'\bnorth\b',     'n'),
    (r'\bsouth\b',     's'),
    (r'\beast\b',      'e'),
    (r'\bwest\b',      'w'),
]

NULL_STRINGS = {'null', 'nan', 'none', 'n/a', 'na', '', '#', '##', '###',
                'not available', 'not applicable', 'unknown'}

# Domain TLD pattern
DOMAIN_RE = re.compile(r'^(.+?)\.(com|net|org|in|co|biz|info|io|us|fr|uk|de|gov|edu)$',
                       re.IGNORECASE)


def _apply_map(text, mapping):
    for pat, repl in mapping:
        text = re.sub(pat, repl, text)
    return text


def normalize_name(text) -> str:
    if not isinstance(text, str):
        return ''
    t = text.strip()
    if not t or t.lower() in NULL_STRINGS:
        return ''

    # ① Strip DBA / f/k/a aliases BEFORE any other processing (while slashes intact)
    t = re.sub(r'\s+f/k/a\s+.*$', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\s+fka\s+.*$',   '', t, flags=re.IGNORECASE)
    t = re.sub(r'\s+aka\s+.*$',   '', t, flags=re.IGNORECASE)
    t = re.sub(r'\s+dba\s+.*$',   '', t, flags=re.IGNORECASE)
    t = re.sub(r'\s+d/b/a\s+.*$', '', t, flags=re.IGNORECASE)

    # ② Handle .com domain names (e.g. "generalelectronicspartners.com")
    if ' ' not in t:
        m = DOMAIN_RE.match(t)
        if m:
            t = m.group(1)  # strip TLD; char n-grams will still match

    # ③ Lowercase
    t = t.lower()

    # ④ Normalize Unicode: strip combining accents from Latin chars,
    #    but PRESERVE Indic scripts (Devanagari U+0900-U+097F, etc.)
    t = unicodedata.normalize('NFKD', t)
    cleaned = []
    for c in t:
        cp = ord(c)
        # Keep: Devanagari, Bengali, Gujarati, Tamil, Telugu, Kannada, Malayalam
        if 0x0900 <= cp <= 0x0DFF:
            cleaned.append(c)
        elif unicodedata.combining(c):
            pass  # strip combining diacritics from Latin
        else:
            cleaned.append(c)
    t = ''.join(cleaned)

    # ⑤ Normalize & → and
    t = re.sub(r'\s*&\s*', ' and ', t)

    # ⑥ Legal suffix normalization
    t = _apply_map(t, SUFFIX_MAP)

    # ⑦ Remove remaining punctuation (keep alphanumeric, spaces, hyphens, Indic)
    t = re.sub(r'[^\w\s\u0900-\u0DFF-]', ' ', t)

    # ⑧ Collapse whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def normalize_address(text) -> str:
    if not isinstance(text, str):
        return ''
    t = text.strip()
    if not t or t.lower() in NULL_STRINGS:
        return ''

    t = t.lower()

    # Remove literal "null" / "nan" / "n/a" tokens
    t = re.sub(r'\bnull\b|\bnan\b|\bn/a\b', ' ', t)

    # Remove leading ## artifacts common in S2/S3
    t = re.sub(r'#+\s*', '', t)

    # Keep content inside parentheses, remove parens
    t = re.sub(r'\(([^)]+)\)', r' \1 ', t)

    # Address abbreviations
    t = _apply_map(t, ADDR_MAP)

    # Normalize Unicode (same as names)
    t = unicodedata.normalize('NFKD', t)
    cleaned = []
    for c in t:
        cp = ord(c)
        if 0x0900 <= cp <= 0x0DFF:
            cleaned.append(c)
        elif unicodedata.combining(c):
            pass
        else:
            cleaned.append(c)
    t = ''.join(cleaned)

    # Remove remaining punctuation
    t = re.sub(r'[^\w\s\u0900-\u0DFF-]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def extract_city_tokens(addr_norm: str) -> str:
    """Last 2 non-numeric tokens from normalized address → city/state proxy."""
    if not addr_norm:
        return ''
    tokens = [t for t in addr_norm.split()
              if not t.isdigit() and len(t) > 1 and t.lower() not in NULL_STRINGS]
    return ' '.join(tokens[-2:]) if len(tokens) >= 2 else ' '.join(tokens)


def extract_first_city_token(addr_norm: str) -> str:
    """Single last non-numeric token → just city name."""
    if not addr_norm:
        return ''
    tokens = [t for t in addr_norm.split()
              if not t.isdigit() and len(t) > 1 and t.lower() not in NULL_STRINGS]
    return tokens[-1] if tokens else ''


def extract_numeric_tokens(addr_norm: str) -> set:
    """Building/unit numbers from address (2-6 digit sequences)."""
    return set(re.findall(r'\b\d{2,6}\b', addr_norm))


def is_non_ascii_name(name: str) -> bool:
    """True if name contains significant non-ASCII (Hindi/Tamil/etc.) characters."""
    if not name:
        return False
    non_ascii = sum(1 for c in name if ord(c) > 127)
    return non_ascii / max(len(name), 1) > 0.2


if __name__ == '__main__':
    tests = [
        ("CONTINENTAL RESOURCES PVT PVT",
         "PLOT NO C-26 G/F KH 11/10, 11/11, SHIV VIHAR MATIYALA, DELHI, Delhi"),
        ("generalelectronicspartners.com", "1681D BIG BRANCH ROAD, CYDE, NC"),
        ("वन स्मार्ट प्रोड्यूसर प्राइवेट लिमिटेड",
         "NO #464 OFFNCE NO A/502, THANE, Maharashtra"),
        ("Lyraviovera f/k/a Continental Public School",
         "No. 252, No. 2Nd & 3Rd Floor, 8Th Main, Amarajyothi Layout"),
        ("KEYST0NE LOGISTICS-INTERNATIONAL LTD", "215 E DOLPHIN WAY, BEAUFORT, NC"),
        ("Société Anonyme de Boulangerie Française", "12, Rue de la Paix, 75001 Paris"),
        ("Green Impex Private Limited", "nan"),
        (None, None),
    ]
    print("=== Normalization tests ===\n")
    for name, addr in tests:
        nn = normalize_name(name)
        na = normalize_address(addr)
        ct = extract_city_tokens(na)
        print(f"  IN : {name!r}")
        print(f"  name: {nn!r}")
        print(f"  addr: {na!r}  city: {ct!r}")
        print()
