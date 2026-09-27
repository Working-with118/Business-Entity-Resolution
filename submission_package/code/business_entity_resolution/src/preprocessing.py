"""
Normalization utilities for business_name and business_address fields.

Design notes
------------
Entity resolution here hinges on collapsing superficial noise (abbreviations,
punctuation, legal-suffix variants, transliteration spellings) into a common
surface form *without* destroying the information that actually distinguishes
two different businesses. Everything here is deterministic, dictionary-based
and language-agnostic on purpose: no external lookups are used (the challenge
rules forbid external data/APIs), so all mappings below are hand-curated from
the noise patterns called out in the problem statement.
"""

import re
import unicodedata

try:
    from anyascii import anyascii as _anyascii
except ImportError:  # pragma: no cover - keeps the pipeline usable if the
    # optional dependency isn't installed yet; falls back to accent-stripping
    # only (won't transliterate non-Latin scripts, but won't crash either).
    _anyascii = None

# ---------------------------------------------------------------------------
# Legal-suffix / abbreviation normalization for business names
# ---------------------------------------------------------------------------
# Map many surface variants to one canonical token. Order matters: longer
# phrases are matched before shorter ones via word-boundary regex.
NAME_SUFFIX_MAP = {
    r"\bprivate limited\b": "pvt ltd",
    r"\bpvt\.?\b": "pvt",
    r"\bprivate\b": "pvt",
    r"\blimited\b": "ltd",
    r"\bltd\.?\b": "ltd",
    r"\bcorporation\b": "corp",
    r"\bcorp\.?\b": "corp",
    r"\bincorporated\b": "inc",
    r"\binc\.?\b": "inc",
    r"\bcompany\b": "co",
    r"\bco\.?\b": "co",
    r"\bllp\.?\b": "llp",
    r"\bllc\.?\b": "llc",
    r"\b&\b": "and",
    r"\bdba\b": "dba",
}

# Common address abbreviations -> canonical long form (canonicalizing to the
# *longer* form tends to help TF-IDF / token-overlap features because it is
# less ambiguous than a 2-letter abbreviation).
ADDRESS_ABBR_MAP = {
    r"\brd\.?\b": "road",
    r"\bst\.?\b": "street",
    r"\bave\.?\b": "avenue",
    r"\bblvd\.?\b": "boulevard",
    r"\bdr\.?\b": "drive",
    r"\bln\.?\b": "lane",
    r"\bapt\.?\b": "apartment",
    r"\bfl\.?\b": "floor",
    r"\bbldg\.?\b": "building",
    r"\bno\.?\b": "number",
    r"\bnr\.?\b": "near",
    r"\bopp\.?\b": "opposite",
    r"\bsec\.?\b": "sector",
    r"\bcolony\b": "colony",
    r"\bpo\b": "post office",
    r"\bps\b": "police station",
}

LANDMARK_PATTERN = re.compile(
    r"\b(near|opposite|opp\.?|behind|next to|adjacent to)\b\s+[^,]+", re.IGNORECASE
)

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _strip_accents(text: str) -> str:
    """Fold accented and non-Latin-script characters to plain ASCII.

    ~13.9% of true matches in the training ground truth pair a Latin-script
    Source-1 name against a Source-2/3 name written in native script
    (Devanagari, Tamil, etc.) — Source 1 is always Latin-script (0% non-ASCII
    in training), but S2/S3 are not. Every downstream string feature
    (Levenshtein, token Jaccard) sees these as completely disjoint strings
    unless they're first brought into a common alphabet, so this is not
    optional accent-stripping — it's what makes ~14% of true matches visible
    to the feature set at all.

    Uses `anyascii`, a static, offline character-transliteration table (not a
    lookup service), which keeps this compliant with the "no external data
    lookup" rule the same way a hand-written abbreviation map is. Falls back
    to NFKD accent-folding (handles accented Latin, e.g. French test-set
    names, but not other scripts) if the dependency is missing.
    """
    if _anyascii is not None:
        return _anyascii(text)
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _apply_map(text: str, mapping: dict) -> str:
    for pattern, repl in mapping.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


def basic_clean(text) -> str:
    if text is None:
        return ""
    text = str(text)
    if text.strip().lower() in ("nan", "none", ""):
        return ""
    text = _strip_accents(text)
    text = text.lower()
    text = text.strip()
    return text


def normalize_name(text) -> str:
    """Canonical form of a business name for matching / blocking."""
    text = basic_clean(text)
    if not text:
        return ""
    text = _apply_map(text, NAME_SUFFIX_MAP)
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_name_core(text) -> str:
    """Name with legal-entity tokens removed entirely — useful as a second,
    coarser blocking / comparison key (e.g. 'acme pvt ltd' -> 'acme')."""
    norm = normalize_name(text)
    if not norm:
        return ""
    stop = {"pvt", "ltd", "corp", "inc", "co", "llp", "llc", "and", "dba", "the"}
    tokens = [t for t in norm.split() if t not in stop]
    return " ".join(tokens)


def extract_landmark(text) -> str:
    text = basic_clean(text)
    m = LANDMARK_PATTERN.search(text)
    return m.group(0) if m else ""


def normalize_address(text) -> str:
    """Canonical form of an address string for matching."""
    text = basic_clean(text)
    if not text:
        return ""
    # Drop landmark clauses (they help humans, but are noisy for token overlap
    # since two branches of the same chain can share a landmark phrase).
    text = LANDMARK_PATTERN.sub(" ", text)
    text = _apply_map(text, ADDRESS_ABBR_MAP)
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_country(text) -> str:
    text = basic_clean(text)
    aliases = {
        "usa": "us", "u.s.a": "us", "u.s.": "us", "united states": "us",
        "united states of america": "us", "america": "us",
        "india": "in", "bharat": "in",
        "france": "fr",
    }
    return aliases.get(text, text)


def tokens(text: str) -> set:
    if not text:
        return set()
    return set(text.split())


def blocking_key(name_core: str, n: int = 4) -> str:
    """Sorted-neighborhood style key: first n chars of each token, sorted,
    joined. Two names with the same key are cheap-to-find near neighbours
    even when the token order or a middle word differs."""
    if not name_core:
        return ""
    toks = sorted(t[:n] for t in name_core.split() if t)
    return "".join(toks)


_NUM_RE = re.compile(r"\d+")


def address_block_key(address_norm: str, country_norm: str) -> str:
    """Country + the (up to two) numeric tokens in a normalized address --
    house/unit/PIN numbers. Catches matches that name-based keys miss
    entirely (cross-script names, heavy typos, transliteration) as long as
    the address carries a shared numeric identifier, which is common for
    branches of the same business or partial-address noise."""
    nums = _NUM_RE.findall(address_norm or "")
    if not nums:
        return ""
    return f"{country_norm}:{'-'.join(sorted(nums)[:2])}"


def loose_name_key(name_core: str, n: int = 3) -> str:
    """Looser companion to blocking_key: only the first two (sorted) tokens'
    first n chars, not every token. Requiring agreement on every token is
    brittle when one source drops or adds a middle word (e.g. a DBA name or
    a dropped legal suffix) -- this key still finds those pairs."""
    if not name_core:
        return ""
    toks = sorted(name_core.split())[:2]
    return "".join(t[:n] for t in toks if t)