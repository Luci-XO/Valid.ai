"""Indicators and countries the checker understands.

Each indicator maps plain-language aliases to a World Bank WDI series code.
Add rows here to cover more statistics; add a new data source (see
``sources/``) to cover national series that the World Bank doesn't publish.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Indicator:
    code: str
    name: str
    # "percent", "count", "usd", "years", "per1000", "per100", "per100k", "number"
    unit: str
    aliases: tuple[str, ...]
    # For level indicators like GDP: a claim such as "GDP grew 7%" is really
    # about the growth-rate series, so we redirect to it.
    percent_variant: str | None = None
    # Weak aliases (like "people") only count when nothing better matched.
    weak_aliases: tuple[str, ...] = field(default_factory=tuple)
    # True for series that are themselves a rate of change (inflation, GDP
    # growth). "Inflation rose 6.7%" states the level of the series, so the
    # direction word describes prices, not the series.
    rate_of_change: bool = False


INDICATORS: tuple[Indicator, ...] = (
    Indicator(
        "SL.UEM.TOTL.ZS",
        "Unemployment rate (% of labour force, ILO modelled estimate)",
        "percent",
        ("unemployment rate", "jobless rate", "unemployment", "joblessness"),
    ),
    Indicator(
        "FP.CPI.TOTL.ZG",
        "Inflation, consumer prices (annual %)",
        "percent",
        (
            "consumer price inflation",
            "retail inflation",
            "cpi inflation",
            "inflation rate",
            "inflation",
            "consumer prices",
        ),
        rate_of_change=True,
    ),
    Indicator(
        "NY.GDP.MKTP.KD.ZG",
        "GDP growth (annual %)",
        "percent",
        ("gdp growth rate", "gdp growth", "economic growth", "growth rate of the economy"),
        rate_of_change=True,
    ),
    Indicator(
        "NY.GDP.PCAP.CD",
        "GDP per capita (current US$)",
        "usd",
        ("gdp per capita", "per capita gdp", "per capita income", "income per person"),
    ),
    Indicator(
        "NY.GDP.MKTP.CD",
        "GDP (current US$)",
        "usd",
        ("gross domestic product", "gdp", "the economy", "economy"),
        percent_variant="NY.GDP.MKTP.KD.ZG",
    ),
    Indicator(
        "SP.POP.TOTL",
        "Population, total",
        "count",
        ("population",),
        weak_aliases=("people", "citizens", "residents", "inhabitants"),
    ),
    Indicator(
        "SP.DYN.LE00.IN",
        "Life expectancy at birth, total (years)",
        "years",
        ("life expectancy",),
    ),
    Indicator(
        "SE.ADT.LITR.ZS",
        "Literacy rate, adult total (% of people ages 15 and above)",
        "percent",
        ("adult literacy rate", "literacy rate", "literacy"),
    ),
    Indicator(
        "SI.POV.DDAY",
        "Poverty headcount ratio at the international poverty line (% of population)",
        "percent",
        ("extreme poverty rate", "extreme poverty", "poverty rate", "poverty"),
    ),
    Indicator(
        "SP.DYN.IMRT.IN",
        "Mortality rate, infant (per 1,000 live births)",
        "per1000",
        ("infant mortality rate", "infant mortality"),
    ),
    Indicator(
        "SL.TLF.CACT.FE.ZS",
        "Labour force participation rate, female (% of female population ages 15+)",
        "percent",
        (
            "female labour force participation",
            "female labor force participation",
            "women's labour force participation",
            "women's labor force participation",
            "female participation rate",
        ),
    ),
    Indicator(
        "EG.ELC.ACCS.ZS",
        "Access to electricity (% of population)",
        "percent",
        ("access to electricity", "electrification rate", "electricity access"),
    ),
)

INDICATORS_BY_CODE = {i.code: i for i in INDICATORS}


# ISO3 code -> (display name, aliases). Aliases are matched case-insensitively
# on word boundaries. Extend freely.
COUNTRIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "IND": ("India", ("india", "indian", "bharat")),
    "USA": ("United States", ("united states", "u.s.", "us economy", "america", "american", "usa")),
    "GBR": ("United Kingdom", ("united kingdom", "uk", "u.k.", "britain", "british")),
    "CHN": ("China", ("china", "chinese")),
    "PAK": ("Pakistan", ("pakistan", "pakistani")),
    "BGD": ("Bangladesh", ("bangladesh", "bangladeshi")),
    "LKA": ("Sri Lanka", ("sri lanka", "sri lankan")),
    "NPL": ("Nepal", ("nepal", "nepali", "nepalese")),
    "IDN": ("Indonesia", ("indonesia", "indonesian")),
    "BRA": ("Brazil", ("brazil", "brazilian")),
    "NGA": ("Nigeria", ("nigeria", "nigerian")),
    "KEN": ("Kenya", ("kenya", "kenyan")),
    "ZAF": ("South Africa", ("south africa", "south african")),
    "DEU": ("Germany", ("germany", "german")),
    "FRA": ("France", ("france", "french")),
    "JPN": ("Japan", ("japan", "japanese")),
    "CAN": ("Canada", ("canada", "canadian")),
    "AUS": ("Australia", ("australia", "australian")),
    "MEX": ("Mexico", ("mexico", "mexican")),
    "PHL": ("Philippines", ("philippines", "filipino")),
    "VNM": ("Vietnam", ("vietnam", "viet nam", "vietnamese")),
    "EGY": ("Egypt", ("egypt", "egyptian")),
    "TUR": ("Turkiye", ("turkey", "turkiye", "turkish")),
    "WLD": ("World", ("world", "global", "worldwide")),
}


@dataclass(frozen=True)
class Match:
    key: str
    start: int
    end: int
    weak: bool = False


def _alias_pattern(alias: str) -> re.Pattern[str]:
    # \b doesn't work next to "." or "'" so use explicit look-arounds.
    return re.compile(r"(?<![\w])" + re.escape(alias) + r"(?![\w])", re.IGNORECASE)


class Catalog:
    """Alias lookup for indicators and countries."""

    def __init__(
        self,
        indicators: tuple[Indicator, ...] = INDICATORS,
        countries: dict[str, tuple[str, tuple[str, ...]]] | None = None,
    ) -> None:
        self.indicators = {i.code: i for i in indicators}
        self.countries = dict(COUNTRIES if countries is None else countries)
        self._ind_patterns: list[tuple[str, re.Pattern[str], bool]] = []
        for ind in indicators:
            for a in ind.aliases:
                self._ind_patterns.append((ind.code, _alias_pattern(a), False))
            for a in ind.weak_aliases:
                self._ind_patterns.append((ind.code, _alias_pattern(a), True))
        self._rebuild_countries()

    def _rebuild_countries(self) -> None:
        self._cty_patterns = [
            (code, _alias_pattern(a))
            for code, (_, aliases) in self.countries.items()
            for a in aliases
        ]

    def add_indicator(self, ind: Indicator) -> None:
        """Register a series found at runtime (by the series finder). No aliases are added."""
        self.indicators.setdefault(ind.code, ind)

    def add_country(self, code: str, name: str, aliases: tuple[str, ...]) -> None:
        self.countries[code] = (name, aliases)
        self._rebuild_countries()

    def country_name(self, code: str) -> str:
        return self.countries.get(code, (code, ()))[0]

    def find_indicators(self, text: str) -> list[Match]:
        """All indicator mentions, longest match wins where they overlap."""
        found = [
            Match(code, m.start(), m.end(), weak)
            for code, pat, weak in self._ind_patterns
            for m in pat.finditer(text)
        ]
        return _drop_overlaps(found)

    def find_countries(self, text: str) -> list[Match]:
        found = [
            Match(code, m.start(), m.end())
            for code, pat in self._cty_patterns
            for m in pat.finditer(text)
        ]
        return _drop_overlaps(found)


def _drop_overlaps(matches: list[Match]) -> list[Match]:
    matches.sort(key=lambda m: (m.start, -(m.end - m.start)))
    kept: list[Match] = []
    for m in matches:
        if kept and m.start < kept[-1].end:
            continue
        kept.append(m)
    return kept
