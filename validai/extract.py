"""Rule-based claim extraction.

Turns free text into structured ``Claim`` objects. No model calls: every
number in a claim is parsed directly from the input text. The optional LLM
extractor in ``llm.py`` produces the same ``Claim`` objects and is validated
against the same number parser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .catalog import Catalog, Indicator, Match
from .models import Claim, ClaimKind

SCALES = {
    "trillion": 1e12,
    "tn": 1e12,
    "billion": 1e9,
    "bn": 1e9,
    "million": 1e6,
    "mn": 1e6,
    "crore": 1e7,
    "lakh": 1e5,
    "thousand": 1e3,
}

_NUM_RE = re.compile(
    r"""
    (?<![\w.,\-−–])
    (?P<cur>US\$|\$|USD\s?|Rs\.?\s?|₹\s?)?
    (?P<sign>[-−–](?=\d))?
    (?P<num>\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    (?![\d])
    (?:\s*(?P<scale>trillion|billion|million|crore|lakh|thousand|tn|bn|mn)\b)?
    (?:\s*(?P<unit>%
        |percentage[\s-]+points?\b
        |percent(?:age)?\b
        |per[\s-]?cent\b
        |pp\b
        |points?\b
        |basis[\s-]+points?\b
        |bps\b
        |years?\b
        |per\s+100,?000(?:\s+live\s+births|\s+people)?
        |per\s+100\b(?:\s+people)?
        |per\s+1,?000(?:\s+live\s+births|\s+people)?
        |per\s+thousand\b
        |dollars\b
        |USD\b))?
    """,
    re.X | re.I,
)

_YEAR_RE = re.compile(r"(?<![\w.,$\-])((?:19|20)\d{2})(?:\s?[-–/]\s?(\d{2}))?(?![\w%])(?!\.\d)(?!,\d)")
# A 4-digit number followed by these is a quantity, not a year.
_NOT_YEAR_AFTER = re.compile(
    r"\s*(%|percent|per\s?cent|trillion|billion|million|crore|lakh|thousand|people|"
    r"dollars|points|years|per\s+1)",
    re.I,
)

_UP_WORDS = (
    r"rose|risen|rises|rising|rise|increased|increases|increasing|increase|grew|grown|"
    r"grows|growing|grow|climbed|climbs|climbing|jumped|jumps|surged|surges|soared|"
    r"went\s+up|gone\s+up|gained|expanded|expands|up"
)
_DOWN_WORDS = (
    r"fell|fallen|falls|falling|fall|declined|declines|declining|decline|dropped|drops|"
    r"dropping|drop|decreased|decreases|decreasing|decrease|shrank|shrunk|shrinks|shrink|"
    r"contracted|contracts|contraction|went\s+down|gone\s+down|plunged|plummeted|slumped|"
    r"eased|slowed|down"
)
_UP_RE = re.compile(rf"(?<![\w-])(?:{_UP_WORDS})(?![\w-])", re.I)
_DOWN_RE = re.compile(rf"(?<![\w-])(?:{_DOWN_WORDS})(?![\w-])", re.I)
_CONTRACT_RE = re.compile(r"(?<![\w-])(?:contracted|contracts|contraction|shrank|shrunk|shrinks|shrink)(?![\w-])", re.I)
_MULT_RE = re.compile(r"(?<![\w-])(doubled|tripled|trebled|halved)(?![\w-])", re.I)
_MULT_VALUE = {"doubled": 100.0, "tripled": 200.0, "trebled": 200.0, "halved": -50.0}

_QUALIFIERS = (
    (r"about|around|roughly|approximately|approx\.?|some|close\s+to|an\s+estimated", "approx"),
    (r"nearly|almost|just\s+under|just\s+below", "nearly"),
    (r"over|more\s+than|above|exceeding|at\s+least|just\s+over|upwards\s+of", "over"),
    (r"under|less\s+than|below|fewer\s+than", "under"),
)
_QUAL_RES = [(re.compile(rf"(?:{p})\s*$", re.I), q) for p, q in _QUALIFIERS]
_QUAL_ANY = "|".join(p for p, _ in _QUALIFIERS)
_PREP_RE = re.compile(rf"(?<![\w])(by|to|from|at|of)\s+(?:(?:{_QUAL_ANY})\s+)?$", re.I)

_PRONOUN_START = re.compile(r"^\W*(it|this|that|the\s+rate|the\s+figure|the\s+number)\b", re.I)

# National series cover everyone. "Youth unemployment is 15%" is a different
# statistic, so we refuse rather than compare it with the overall rate.
_SUBGROUP_RE = re.compile(
    r"(?<![\w])(youth|young\s+people|teenagers?|women|woman|female|men|male|urban|rural|"
    r"graduates?|aged\s+\d+|children|elderly|minorities|households?\s+headed)(?![\w'])",
    re.I,
)

_SHARE_OF_GDP = re.compile(r"(?:%|per\s?cent|percent)\s+of\s+(?:its\s+|the\s+|our\s+)?(?:gdp|gross\s+domestic\s+product|the\s+economy|economic\s+output)", re.I)

_NUMBER_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20,
}
_LOOKBACK_RE = re.compile(
    r"(?:over|in|during|for)\s+the\s+(?:past|last|previous)\s+(?:(decade)|(\d+|\w+)\s+years)", re.I
)


def _lookback(text: str) -> int | None:
    m = _LOOKBACK_RE.search(text)
    if not m:
        return None
    if m.group(1):
        return 10
    word = m.group(2).lower()
    return int(word) if word.isdigit() else _NUMBER_WORDS.get(word)


@dataclass
class NumTok:
    start: int
    end: int
    mantissa: float
    decimals: int
    scale: float
    unit: str | None  # pct, pp, years, per1000, usd, inr, None
    raw: str

    @property
    def value(self) -> float:
        return self.mantissa * self.scale


@dataclass
class YearTok:
    start: int
    end: int
    year: int
    role: str  # "start", "end", "plain"


def parse_numbers(text: str, skip: list[tuple[int, int]] | None = None) -> list[NumTok]:
    """Find quantities in ``text``. Spans listed in ``skip`` (years) are ignored."""
    skip = skip or []
    out: list[NumTok] = []
    for m in _NUM_RE.finditer(text):
        s = m.start("num")
        if any(a <= s < b for a, b in skip):
            continue
        if re.search(r"(?<![\w])per\s+$", text[max(0, m.start() - 6) : m.start()], re.I):
            continue  # the "1,000" in "beds per 1,000 people" is part of a unit
        num_txt = m.group("num")
        digits = num_txt.replace(",", "")
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        mantissa = float(digits)
        if m.group("sign"):
            mantissa = -mantissa
        scale_word = (m.group("scale") or "").lower()
        scale = SCALES.get(scale_word, 1.0)
        ut = re.sub(r"[\s-]+", " ", (m.group("unit") or "").lower()).strip()
        cur = (m.group("cur") or "").strip().lower()
        unit: str | None = None
        if ut in ("%", "percent", "percentage", "per cent", "percent"):
            unit = "pct"
        elif "basis" in ut or ut == "bps":
            unit = "pp"
            mantissa /= 100.0
            decimals += 2
        elif "point" in ut or ut == "pp":
            unit = "pp"
        elif ut.startswith("year"):
            unit = "years"
        elif re.match(r"per 100,?000", ut):
            unit = "per100k"
        elif re.match(r"per 100\b(?!,)", ut):
            unit = "per100"
        elif ut.startswith("per ") and ("1" in ut or "thousand" in ut):
            unit = "per1000"
        if unit is None and not cur:
            # "0.6 hospital beds per 1,000 people": unit a few words later.
            later = re.match(r"[A-Za-z\s'-]{1,40}?\bper\s+(100,?000|1,?000|100)(?![\d,])", text[m.end() :])
            if later:
                unit = {"100000": "per100k", "1000": "per1000", "100": "per100"}[later.group(1).replace(",", "")]
        if cur in ("$", "us$", "usd") or ut in ("dollars", "usd"):
            unit = "usd"
        elif cur.startswith("rs") or cur == "₹":
            unit = "inr"
        out.append(
            NumTok(
                start=m.start(),
                end=m.end(),
                mantissa=mantissa,
                decimals=decimals,
                scale=scale,
                unit=unit,
                raw=m.group(0).strip(),
            )
        )
    return out


def parse_years(text: str, ref_year: int) -> list[YearTok]:
    out: list[YearTok] = []
    for m in _YEAR_RE.finditer(text):
        if _NOT_YEAR_AFTER.match(text, m.end()):
            continue
        year = int(m.group(1))
        before = text[max(0, m.start() - 12) : m.start()].lower()
        if re.search(r"(since|from|between|starting)\s+(the\s+)?$", before):
            role = "start"
        elif re.search(r"(to|until|till|through|and)\s+$", before):
            role = "end"
        else:
            role = "plain"
        out.append(YearTok(m.start(), m.end(), year, role))
    for m in re.finditer(r"(?<![\w])(last|this|previous)\s+year(?![\w])", text, re.I):
        word = m.group(1).lower()
        out.append(YearTok(m.start(), m.end(), ref_year if word == "this" else ref_year - 1, "plain"))
    out.sort(key=lambda y: y.start)
    return out


# ---------------------------------------------------------------- splitting

_SENT_END = re.compile(r"[.!?]+[\"”’')\]]*\s+|\n\s*\n|\n")
_ABBREV_END = re.compile(r"(?:\b[A-Za-z]\.){2,3}$|\b(?:Mr|Mrs|Ms|Dr|vs|etc|e\.g|i\.e|Rs|No|approx|Govt|St)\.$")


def split_sentences(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENT_END.finditer(text):
        head = text[start : m.start() + 1].rstrip()
        nxt = text[m.end() : m.end() + 1]
        if "\n" not in m.group(0):
            if _ABBREV_END.search(head) and not nxt.isupper():
                continue
            if nxt and not (nxt.isupper() or nxt.isdigit() or nxt in "\"“'‘("):
                continue
        end = m.start() + len(m.group(0).rstrip())
        if text[start:end].strip():
            spans.append((start, end))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


_CLAUSE_SEP = re.compile(
    r";\s*"
    r"|,?\s+(?:while|whereas|but|although|even\s+as|before)\s+"
    r"|,?\s+(?:and\s+)?then\s+"
    r"|,\s+and\s+"
    r"|,?\s+(?:compared\s+(?:with|to)|versus|vs\.?)\s+"
    # "higher than Sylvania's 6.6%" but not "more than 99%"
    r"|(?:(?<=higher)|(?<=lower)|(?<=bigger)|(?<=smaller)|(?<=faster)|(?<=slower)|(?<=better)|(?<=worse))\s+than\s+",
    re.I,
)
_AND_SEP = re.compile(r"\s+and\s+", re.I)


def _has_quantity(text: str, ref_year: int) -> bool:
    years = parse_years(text, ref_year)
    nums = parse_numbers(text, [(y.start, y.end) for y in years])
    return bool(nums) or bool(_MULT_RE.search(text))


def split_clauses(text: str, start: int, end: int, ref_year: int) -> list[tuple[int, int]]:
    pieces: list[tuple[int, int]] = []
    cur = start
    for m in _CLAUSE_SEP.finditer(text, start, end):
        pieces.append((cur, m.start()))
        cur = m.end()
    pieces.append((cur, end))

    out: list[tuple[int, int]] = []
    for s, e in pieces:
        sub_start = s
        for m in _AND_SEP.finditer(text, s, e):
            left, right = text[sub_start : m.start()], text[m.end() : e]
            if _has_quantity(left, ref_year) and _has_quantity(right, ref_year):
                out.append((sub_start, m.start()))
                sub_start = m.end()
        out.append((sub_start, e))
    return [(s, e) for s, e in out if text[s:e].strip()]


# ---------------------------------------------------------------- extraction


def _compatible(ind: Indicator, tok: NumTok) -> bool:
    u = ind.unit
    if tok.unit == "inr":
        return False
    if u == "percent":
        return tok.unit in ("pct", "pp")
    if u == "count":
        return tok.unit in (None, "pct") and (tok.unit == "pct" or tok.value >= 1000 or tok.scale > 1)
    if u == "usd":
        # Require an explicit currency so "added 2 million jobs" isn't read as GDP.
        return tok.unit in ("usd", "pct")
    if u == "years":
        return tok.unit == "years" or (tok.unit is None and 15 <= tok.value <= 100)
    if u == "per1000":
        return tok.unit in ("per1000", "pct") or (tok.unit is None and 0 < tok.value < 1000)
    if u in ("per100", "per100k"):
        # "85% of households" is not "per 100 people".
        return tok.unit in (None, u)
    if u == "number":
        return tok.unit is None
    return False


def _preposition(text: str, pos: int) -> str | None:
    m = _PREP_RE.search(text[max(0, pos - 30) : pos])
    return m.group(1).lower() if m else None


def _qualifier(text: str, pos: int) -> str | None:
    before = text[max(0, pos - 25) : pos]
    for pat, q in _QUAL_RES:
        if pat.search(before):
            return q
    return None


def _direction(text: str) -> str | None:
    up, down = _UP_RE.search(text), _DOWN_RE.search(text)
    if up and down:
        return "up" if up.start() < down.start() else "down"
    if up:
        return "up"
    if down:
        return "down"
    return None


def _pick_indicator(matches: list[Match], first_num_pos: int, tok: NumTok | None) -> str | None:
    strong = [m for m in matches if not m.weak]
    if strong:
        before = [m for m in strong if m.start < first_num_pos]
        return (before[-1] if before else strong[0]).key
    weak = [m for m in matches if m.weak]
    if weak and tok is not None and tok.unit is None and tok.value >= 1e6:
        return weak[0].key
    return None


def _nearest_year(years: list[YearTok], pos: int) -> YearTok | None:
    if not years:
        return None
    after = [y for y in years if y.start >= pos]
    return after[0] if after else years[-1]


def _unit_label(ind: Indicator, tok: NumTok, kind: ClaimKind) -> str:
    if kind is ClaimKind.CHANGE:
        if ind.unit == "percent":
            return "pp" if tok.unit == "pp" else "pct_or_pp"
        return "rel%" if tok.unit == "pct" else ind.unit
    return ind.unit


def extract_claims(
    text: str,
    catalog: Catalog,
    default_country: str | None = None,
    ref_year: int = 2026,
) -> list[Claim]:
    claims: list[Claim] = []
    doc_country = default_country
    prev_sentence_ind: str | None = None

    for s_start, s_end in split_sentences(text):
        sentence = text[s_start:s_end]
        sent_ind: str | None = None
        sent_countries = catalog.find_countries(sentence)
        sent_country = sent_countries[0].key if sent_countries else None

        for c_start, c_end in split_clauses(text, s_start, s_end, ref_year):
            clause = text[c_start:c_end]
            claim = _clause_claim(
                clause,
                c_start,
                catalog,
                inherited_ind=sent_ind
                or (prev_sentence_ind if _PRONOUN_START.match(sentence) else None),
                country=None,
                ref_year=ref_year,
            )
            cty = catalog.find_countries(clause)
            if cty:
                sent_country = cty[0].key
            if claim is None:
                continue
            claim.country = (cty[0].key if cty else None) or sent_country or doc_country
            if claim.indicator:
                sent_ind = claim.indicator
            claims.append(claim)

        if sent_country:
            doc_country = sent_country
        prev_sentence_ind = sent_ind or prev_sentence_ind
    return claims


def _clause_claim(
    clause: str,
    offset: int,
    catalog: Catalog,
    inherited_ind: str | None,
    country: str | None,
    ref_year: int,
    forced_ind: str | None = None,
) -> Claim | None:
    years = parse_years(clause, ref_year)
    nums = parse_numbers(clause, [(y.start, y.end) for y in years])
    mult = _MULT_RE.search(clause)
    if not nums and not mult:
        return None

    lead = len(clause) - len(clause.lstrip())
    trail = len(clause.rstrip())
    base = dict(
        text=clause.strip(),
        start=offset + lead,
        end=offset + trail,
        country=country,
    )

    ind_matches = catalog.find_indicators(clause)
    first_pos = nums[0].start if nums else (mult.start() if mult else 0)
    code = forced_ind or _pick_indicator(ind_matches, first_pos, nums[0] if nums else None) or inherited_ind

    if code is None:
        # A quantity we can't tie to any series. Report it only if it looks
        # like a statistic (a percentage, money, or a big scaled number).
        if any(t.unit in ("pct", "pp", "usd", "inr", "per1000", "per100", "per100k") or t.scale > 1 for t in nums):
            tok = nums[0]
            return Claim(kind=ClaimKind.LEVEL, value=tok.value, decimals=tok.decimals, scale=tok.scale, **base)
        return None

    ind = catalog.indicators[code]
    if not forced_ind and code in ("NY.GDP.MKTP.CD", "NY.GDP.MKTP.KD.ZG") and _SHARE_OF_GDP.search(clause):
        # "Defence spending is 2.4% of GDP" is about defence spending, not GDP.
        tok = nums[0] if nums else None
        if tok is not None:
            return Claim(kind=ClaimKind.LEVEL, value=tok.value, decimals=tok.decimals, scale=tok.scale, **base)
        return None
    direction = _direction(clause)

    masked = clause
    for m in ind_matches:
        masked = masked[: m.start] + " " * (m.end - m.start) + masked[m.end :]
    if nums and re.search(r"(?<![\w])(averag\w*|on\s+average|median|cumulative\w*)(?![\w])", masked, re.I):
        tok = nums[0]
        return Claim(kind=ClaimKind.LEVEL, indicator=ind.code, value=tok.value, unit="aggregate", decimals=tok.decimals, scale=tok.scale, **base)
    sub = None if forced_ind else _SUBGROUP_RE.search(masked)
    if sub and nums:
        tok = nums[0]
        return Claim(kind=ClaimKind.LEVEL, indicator=ind.code, value=tok.value, unit=f"subgroup:{sub.group(1).lower()}", decimals=tok.decimals, scale=tok.scale, **base)
    lookback = _lookback(clause)

    # "GDP grew 7%" is a statement about GDP growth, not the GDP level.
    # "Since 2015, GDP has grown 80%" stays a relative change of the level.
    redirected = False
    if ind.percent_variant and nums and nums[0].unit == "pct" and not forced_ind:
        if not re.search(r"\bsince\b|\bfrom\b|\bbetween\b", clause, re.I):
            ind = catalog.indicators[ind.percent_variant]
            redirected = True

    toks = [t for t in nums if _compatible(ind, t)]

    if mult and not toks:
        start_y = next((y for y in years if y.role == "start"), None)
        end_y = next((y for y in years if y.role in ("end", "plain")), None)
        return Claim(
            kind=ClaimKind.CHANGE,
            indicator=ind.code,
            value=abs(_MULT_VALUE[mult.group(1).lower()]),
            unit="rel%",
            direction="down" if mult.group(1).lower() == "halved" else "up",
            year=end_y.year if end_y else None,
            start_year=start_y.year if start_y else None,
            since=bool(start_y and not end_y),
            lookback=lookback,
            decimals=0,
            # "doubled" is loose speech; "nearly doubled" is looser in one direction.
            qualifier=_qualifier(clause, mult.start()) or "approx",
            **base,
        )

    if not toks:
        if nums:
            # Number present but in the wrong unit for this series.
            tok = nums[0]
            return Claim(kind=ClaimKind.LEVEL, indicator=ind.code, value=tok.value, unit=f"mismatch:{tok.unit}", decimals=tok.decimals, scale=tok.scale, **base)
        return None

    # from A to B
    if len(toks) >= 2 and _preposition(clause, toks[0].start) == "from" and _preposition(clause, toks[1].start) == "to":
        a, b = toks[0], toks[1]
        start_year = end_year = None
        for y in years:
            if y.start < a.start and y.role == "start":
                start_year = y.year
            elif a.end <= y.start < b.start:
                start_year = y.year
            elif y.start >= b.end or y.role == "end":
                end_year = y.year
        if start_year is None and len(years) >= 2:
            start_year, end_year = years[0].year, years[-1].year
        return Claim(
            kind=ClaimKind.CHANGE,
            indicator=ind.code,
            start_value=a.value,
            value=b.value,
            unit=ind.unit,
            direction=direction or ("up" if b.value > a.value else "down"),
            year=end_year,
            start_year=start_year,
            since=start_year is not None and end_year is None,
            decimals=min(a.decimals, b.decimals),
            scale=b.scale,
            qualifier=_qualifier(clause, b.start),
            **base,
        )

    tok = toks[0]
    prep = _preposition(clause, tok.start)
    qualifier = _qualifier(clause, tok.start)
    start_tok = next((y for y in years if y.role == "start"), None)
    others = [y for y in years if y is not start_tok]

    kind = ClaimKind.LEVEL
    value = tok.value
    level_direction = None

    if direction:
        if redirected:
            # "GDP grew (by) 7%" / "the economy shrank 5.8%": a growth-rate level.
            if direction == "down":
                value = -abs(value)
        elif prep == "by" or tok.unit == "pp":
            kind = ClaimKind.CHANGE
        elif prep == "to":
            level_direction = direction
        elif ind.rate_of_change:
            if direction == "down" and (redirected or _CONTRACT_RE.search(clause)):
                value = -abs(value)
            elif direction == "down":
                kind = ClaimKind.CHANGE
        elif not prep or prep not in ("at", "of"):
            kind = ClaimKind.CHANGE

    if kind is ClaimKind.CHANGE:
        end_tok = next((y for y in others if y.role == "end"), None) or (others[-1] if others else None)
        start_year = start_tok.year if start_tok else None
        if start_year is None and len(others) >= 2:
            start_year = others[0].year
            end_tok = others[-1]
        return Claim(
            kind=ClaimKind.CHANGE,
            indicator=ind.code,
            value=abs(value),
            unit=_unit_label(ind, tok, kind),
            direction=direction,
            year=end_tok.year if end_tok else None,
            start_year=start_year,
            since=start_tok is not None and end_tok is None,
            lookback=lookback,
            decimals=tok.decimals,
            scale=tok.scale,
            qualifier=qualifier,
            **base,
        )

    year_tok = _nearest_year(others, tok.end)
    return Claim(
        kind=ClaimKind.LEVEL,
        indicator=ind.code,
        value=value,
        unit=ind.unit,
        direction=level_direction,
        year=year_tok.year if year_tok else None,
        start_year=start_tok.year if start_tok else None,
        decimals=tok.decimals,
        scale=tok.scale,
        qualifier=qualifier,
        **base,
    )
