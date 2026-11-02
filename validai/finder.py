"""Series finder: which official series is this claim about?

Used when a claim's statistic isn't in the built-in catalog ("hospital
beds", "youth unemployment"). Two steps:

1. Keyword search (BM25) over World Bank indicator names narrows the list
   to a few candidates whose unit fits the claimed number.
2. The local LLM picks one candidate, or says none fits. It can only pick
   from the list, so it can't invent a series code.

Without an LLM, the top search hit is used if it matches strongly enough.
Either way, the number itself is then fetched from the data source and
checked in code. The model never supplies a value.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .catalog import Indicator

SEED = Path(__file__).parent / "data" / "indicators_seed.json"
DEFAULT_FULL = Path("data") / "indicators.json"

_STOP = set(
    """a an the of in on at to for from by and or is are was were be been being has have had
    its it this that these those with as than about around over under nearly almost more less
    rose fell rise fall grew grow increased decreased declined dropped reached hit stood now
    currently today year years since between per cent percent total share rate level number
    country nation national our their we government official data show shows said says
    new record high low lowest highest just only also still""".split()
)
_SYNONYMS = {
    "labour": "labor", "enrolment": "enrollment", "enrolments": "enrollment",
    "jobless": "unemployment", "joblessness": "unemployment", "unemployed": "unemployment",
    "doctors": "physician", "doctor": "physician", "physicians": "physician",
    "defence": "defense", "kids": "children", "child": "children",
    "young": "youth", "youngsters": "youth",
    "women": "female", "woman": "female", "girls": "female",
    "vaccination": "immunization", "vaccinations": "immunization", "immunisation": "immunization",
    "urbanisation": "urbanization", "mps": "parliament", "lawmakers": "parliament",
    "cellphone": "mobile", "phones": "mobile", "phone": "mobile",
    "using": "use", "users": "use", "user": "use", "uses": "use",
}


def tokenize(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z][a-z&'-]*", text.lower()):
        w = w.strip("'-").removesuffix("'s")
        w = _SYNONYMS.get(w, w)
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        w = _SYNONYMS.get(w, w)
        if w and w not in _STOP and len(w) > 1:
            out.append(w)
    return out


def unit_from_name(name: str) -> tuple[str, bool]:
    """Infer (unit, is_rate_of_change) from a World Bank series name."""
    n = name.lower()
    if "annual %" in n:
        return "percent", True
    if "per 100,000" in n:
        return "per100k", False
    if "per 1,000" in n:
        return "per1000", False
    if "per 100 " in n or n.endswith("per 100 people)"):
        return "per100", False
    if "%" in n:
        return "percent", False
    if "us$" in n and "per us$" not in n:
        return "usd", False
    if "(years)" in n:
        return "years", False
    if n.startswith("population"):
        return "count", False
    return "number", False


@dataclass
class Candidate:
    code: str
    name: str
    unit: str
    rate_of_change: bool
    score: float
    matched: list[str]

    def indicator(self) -> Indicator:
        return Indicator(self.code, self.name, self.unit, (), rate_of_change=self.rate_of_change)

    def to_dict(self) -> dict:
        return {"code": self.code, "name": self.name, "score": round(self.score, 2)}


@dataclass
class SeriesMatch:
    candidate: Candidate
    method: str  # "search" or "local AI: <model>"
    candidates: list[Candidate] = field(default_factory=list)
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "code": self.candidate.code,
            "name": self.candidate.name,
            "method": self.method,
            "reason": self.reason,
            "candidates": [c.to_dict() for c in self.candidates],
        }


class SeriesIndex:
    """BM25 over indicator names plus a few keywords."""

    def __init__(self, entries: list[dict]) -> None:
        self.entries = [e for e in entries if e.get("code") and e.get("name")]
        self.docs = [tokenize(f"{e['name']} {e.get('keywords', '')}") for e in self.entries]
        self.name_terms = [set(tokenize(e["name"])) for e in self.entries]
        self.df: Counter = Counter()
        for d in self.docs:
            self.df.update(set(d))
        self.avgdl = sum(map(len, self.docs)) / max(len(self.docs), 1)

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "SeriesIndex":
        """Full downloaded list if present, else the bundled starter list."""
        candidates = [path, os.environ.get("VALIDAI_INDICATORS"), DEFAULT_FULL, SEED]
        for p in candidates:
            if p and Path(p).exists():
                data = json.loads(Path(p).read_text(encoding="utf-8"))
                return cls(data["indicators"])
        return cls([])

    def __len__(self) -> int:
        return len(self.entries)

    def search(self, query: str, k: int = 8) -> list[Candidate]:
        q = list(dict.fromkeys(tokenize(query)))
        if not q or not self.docs:
            return []
        n = len(self.docs)
        scored = []
        for i, doc in enumerate(self.docs):
            tf = Counter(doc)
            s = 0.0
            matched = []
            for t in q:
                if t not in tf:
                    continue
                matched.append(t)
                idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                f = tf[t]
                s += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * len(doc) / self.avgdl))
            if s > 0:
                scored.append((s, i, matched))
        scored.sort(key=lambda x: -x[0])
        out = []
        for s, i, matched in scored[:k]:
            e = self.entries[i]
            unit, roc = unit_from_name(e["name"])
            out.append(Candidate(e["code"], e["name"], unit, roc, s, matched))
        return out

    def strong(self, c: Candidate) -> bool:
        """Is a search hit good enough to use without an LLM?"""
        i = next(i for i, e in enumerate(self.entries) if e["code"] == c.code)
        name_terms = self.name_terms[i]
        hits_in_name = name_terms.intersection(c.matched)
        if len(c.matched) >= 2 and hits_in_name:
            return True
        if name_terms and name_terms <= set(c.matched):
            return True
        # One very specific word in the series name ("remittances", "gini") is enough.
        return any(self.df[t] == 1 for t in hits_in_name)


PICK_SYSTEM = """You match a statistical claim to the official data series that measures exactly the same thing.
Choose only from the numbered candidates. If none measures the same thing (same population group,
same concept, compatible unit), answer null. A national total is NOT the same as a subgroup
(youth, women, rural...). Answer JSON: {"code": "<candidate code or null>", "reason": "<one short sentence>"}"""


class SeriesFinder:
    def __init__(self, index: SeriesIndex | None = None, llm=None) -> None:
        self.index = index or SeriesIndex.load()
        self.llm = llm

    def find(self, claim_text: str, fits: Callable[[Indicator], bool], exclude: set[str] | None = None) -> SeriesMatch | None:
        exclude = exclude or set()
        cands = [c for c in self.index.search(claim_text, k=20) if c.code not in exclude and fits(c.indicator())][:6]
        if not cands:
            return None

        if self.llm is not None and self.llm.available:
            from .llm import LLMError

            listing = "\n".join(f"{i + 1}. {c.code}: {c.name}" for i, c in enumerate(cands))
            try:
                data = self.llm.chat_json(PICK_SYSTEM, f"Claim: {claim_text}\n\nCandidates:\n{listing}")
            except LLMError:
                data = None
            if data is not None:
                code = data.get("code")
                reason = data.get("reason") if isinstance(data.get("reason"), str) else None
                chosen = next((c for c in cands if c.code == code), None)
                if chosen is None:
                    return None  # the model said none fit (or answered off-list)
                return SeriesMatch(chosen, f"local AI ({self.llm.model})" if self.llm.is_local else f"AI ({self.llm.model})", cands, reason)

        top = cands[0]
        if self.index.strong(top):
            return SeriesMatch(top, "keyword search", cands, f"matched: {', '.join(top.matched)}")
        return None
