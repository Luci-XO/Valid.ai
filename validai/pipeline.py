from __future__ import annotations

import datetime as _dt
import time

from .catalog import Catalog
from .extract import _clause_claim, _compatible, extract_claims, parse_numbers, parse_years
from .models import Claim, Verdict
from .sources.base import DataSource
from .verify import verify_claim

def catalog_for(source: DataSource) -> Catalog:
    """The default catalog plus any countries the source adds.

    Cached on the source object itself. (Caching by id() was a bug: ids are
    reused after garbage collection, so a new source could get an old
    source's catalog.)
    """
    cat = getattr(source, "_validai_catalog", None)
    if cat is None:
        cat = Catalog()
        for code, (name, aliases) in source.extra_countries().items():
            cat.add_country(code, name, aliases)
        source._validai_catalog = cat
    return cat


def _needs_series(c: Claim) -> bool:
    return c.indicator is None or (c.unit or "").startswith("subgroup:")


def find_series(text: str, claims: list[Claim], catalog: Catalog, finder, ref_year: int) -> dict[int, dict]:
    """Ask the series finder about claims the catalog couldn't place.

    Replaces those claims in place and returns {claim index: match info}.
    """
    found: dict[int, dict] = {}
    for i, c in enumerate(claims):
        if not _needs_series(c):
            continue
        clause = text[c.start : c.end]
        years = parse_years(clause, ref_year)
        nums = parse_numbers(clause, [(y.start, y.end) for y in years])
        if not nums:
            continue
        tok = nums[0]
        match = finder.find(
            clause,
            fits=lambda ind, tok=tok: _compatible(ind, tok),
            exclude={c.indicator} if c.indicator else None,
        )
        if match is None:
            continue
        ind = match.candidate.indicator()
        catalog.add_indicator(ind)
        new = _clause_claim(clause, c.start, catalog, inherited_ind=None, country=None, ref_year=ref_year, forced_ind=ind.code)
        if new is None:
            continue
        new.country = c.country
        claims[i] = new
        found[i] = match.to_dict()
    return found


def check_text(
    text: str,
    source: DataSource,
    *,
    extractor: str = "rules",
    default_country: str | None = None,
    ref_year: int | None = None,
    llm_client=None,
    finder=None,
) -> dict:
    """Extract every numeric claim in ``text`` and verify each one.

    ``finder`` (a ``SeriesFinder``) looks up series for claims the built-in
    catalog doesn't cover. Pass None to use the catalog only.
    """
    ref_year = ref_year or _dt.date.today().year
    catalog = catalog_for(source)

    t0 = time.perf_counter()
    used = "rules"
    if extractor == "llm":
        from .llm import LLMError, extract_claims_llm

        try:
            claims = extract_claims_llm(text, catalog, llm_client, default_country, ref_year)
            used = "llm"
        except LLMError as e:
            claims = extract_claims(text, catalog, default_country, ref_year)
            used = f"rules (LLM failed: {e})"
    else:
        claims = extract_claims(text, catalog, default_country, ref_year)
    t1 = time.perf_counter()

    matches = find_series(text, claims, catalog, finder, ref_year) if finder is not None else {}
    t2 = time.perf_counter()

    results = []
    for i, c in enumerate(claims):
        r = verify_claim(c, source, catalog, ref_year)
        r.series_match = matches.get(i)
        results.append(r)
    t3 = time.perf_counter()

    summary = {v.value: 0 for v in Verdict}
    for r in results:
        summary[r.verdict.value] += 1
    return {
        "claims": [r.to_dict() for r in results],
        "summary": summary,
        "extractor": used,
        "source": source.name,
        "timing_ms": {
            "extract": round((t1 - t0) * 1000, 1),
            "find_series": round((t2 - t1) * 1000, 1),
            "verify": round((t3 - t2) * 1000, 1),
        },
    }
