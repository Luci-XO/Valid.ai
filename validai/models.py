from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class Verdict(str, Enum):
    SUPPORTED = "supported"  # matches official data within rounding
    INACCURATE = "inaccurate"  # close, but outside rounding (minor error)
    CONTRADICTED = "contradicted"  # clearly wrong value or wrong direction
    MISLEADING = "misleading"  # number is real but framed wrongly (stale year, flat trend)
    UNVERIFIABLE = "unverifiable"  # no matching series, country, or year


class ClaimKind(str, Enum):
    LEVEL = "level"  # "unemployment was 4.2% in 2023"
    CHANGE = "change"  # "unemployment fell by 1.1 points since 2020"


@dataclass
class Claim:
    text: str  # the clause the claim was read from
    start: int  # character offsets into the original input
    end: int
    kind: ClaimKind
    indicator: str | None = None  # catalog code, e.g. SL.UEM.TOTL.ZS
    country: str | None = None  # ISO3 code
    year: int | None = None  # end year; None means "latest"
    start_year: int | None = None  # for change claims
    value: float | None = None  # level value, or change magnitude
    start_value: float | None = None  # for "from A to B"
    unit: str | None = None  # "%", "pp", "count", "usd", "years"
    direction: str | None = None  # "up" / "down"
    qualifier: str | None = None  # "approx", "over", "under", "nearly"
    decimals: int = 0  # precision of the claimed number (mantissa)
    scale: float = 1.0  # multiplier applied to the mantissa (e.g. 1e9)
    since: bool = False  # "since 2019": compare start_year to latest
    lookback: int | None = None  # "over the past five years": start = end - 5

    def to_dict(self) -> dict:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


@dataclass
class Evidence:
    year: int
    value: float


@dataclass
class ClaimResult:
    claim: Claim
    verdict: Verdict
    explanation: str
    indicator_name: str | None = None
    country_name: str | None = None
    evidence: list[Evidence] = field(default_factory=list)
    source_url: str | None = None
    source_name: str | None = None
    series_match: dict | None = None  # set when the series finder picked the series

    def to_dict(self) -> dict:
        return {
            "series_match": self.series_match,
            "claim": self.claim.to_dict(),
            "verdict": self.verdict.value,
            "explanation": self.explanation,
            "indicator_name": self.indicator_name,
            "country_name": self.country_name,
            "evidence": [asdict(e) for e in self.evidence],
            "source_url": self.source_url,
            "source_name": self.source_name,
        }
