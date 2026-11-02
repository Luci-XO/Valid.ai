"""Deterministic verdicts. No model calls in this file.

Tolerances follow how people round: a claim of "4.2%" is supported by any
official value in [4.15, 4.25]. A claim of "4%" is supported by [3.5, 4.5].
"""

from __future__ import annotations

from .catalog import Catalog, Indicator
from .models import Claim, ClaimKind, ClaimResult, Evidence, Verdict
from .sources.base import DataSource, SourceError

SEVERITY = {
    Verdict.SUPPORTED: 0,
    Verdict.INACCURATE: 1,
    Verdict.MISLEADING: 2,
    Verdict.CONTRADICTED: 3,
    Verdict.UNVERIFIABLE: 4,
}

# Below the rounding tolerance, a claim is "supported". Up to
# max(2 x tolerance, MINOR_REL x actual) it is "inaccurate". Beyond, "contradicted".
MINOR_REL = 0.05
APPROX_REL = 0.05  # extra slack for "about", "around", "roughly"
NEARLY_REL = 0.10  # "nearly 5%" accepts 4.5% to 5%


def worst(*verdicts: Verdict) -> Verdict:
    return max(verdicts, key=lambda v: SEVERITY[v])


def rounding_tolerance(decimals: int, scale: float) -> float:
    return 0.5 * (10 ** -decimals) * scale


def grade(
    claimed: float,
    actual: float,
    decimals: int,
    scale: float = 1.0,
    qualifier: str | None = None,
) -> Verdict:
    tol = rounding_tolerance(decimals, scale) + 1e-9 * max(abs(actual), 1.0)
    minor = MINOR_REL
    if qualifier == "approx":
        tol = max(tol, APPROX_REL * abs(actual))
        minor = MINOR_REL + APPROX_REL
    elif qualifier == "over" and actual >= claimed - tol:
        return Verdict.SUPPORTED
    elif qualifier == "under" and actual <= claimed + tol:
        return Verdict.SUPPORTED
    elif qualifier == "nearly":
        lo = claimed - max(NEARLY_REL * abs(claimed), tol)
        if lo <= actual <= claimed + tol:
            return Verdict.SUPPORTED
    err = abs(actual - claimed)
    if err <= tol:
        return Verdict.SUPPORTED
    if err <= max(2 * tol, minor * abs(actual)):
        return Verdict.INACCURATE
    return Verdict.CONTRADICTED


# ------------------------------------------------------------------ formatting


def _big(v: float) -> str:
    a = abs(v)
    for size, word in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if a >= size:
            return f"{v / size:,.2f} {word}"
    return f"{v:,.0f}"


def fmt(v: float, unit: str, decimals: int | None = 1) -> str:
    """Format a value. ``decimals=None`` shows up to 2 decimals without trailing zeros."""
    if decimals is None:
        if unit in ("usd", "count"):
            return fmt(v, unit, 0)
        s = f"{v:.2f}".rstrip("0")
        s = s + "0" if s.endswith(".") else s
        return fmt(float(s), unit, len(s.split(".")[1]))
    d = max(decimals, 1)
    if unit == "percent":
        return f"{v:.{d}f}%"
    if unit == "pp":
        return f"{v:+.{d}f} percentage points"
    if unit == "rel%":
        return f"{v:+.{d}f}%"
    if unit == "usd":
        return ("-" if v < 0 else "") + "$" + _big(abs(v)) if abs(v) >= 1e6 else f"${v:,.0f}"
    if unit == "count":
        return _big(v)
    if unit == "years":
        return f"{v:.{d}f} years"
    if unit == "per1000":
        return f"{v:.{d}f} per 1,000"
    if unit == "per100":
        return f"{v:.{d}f} per 100 people"
    if unit == "per100k":
        return f"{v:.{d}f} per 100,000"
    if unit == "number":
        return f"{v:,.{d}f}"
    return f"{v:g}"


def _fmt_delta(v: float, unit: str, decimals: int) -> str:
    if unit == "percent":
        return fmt(v, "pp", decimals)
    if unit in ("usd", "count"):
        return ("+" if v >= 0 else "-") + fmt(abs(v), unit, decimals)
    return f"{v:+.{max(decimals, 1)}f}"


# ------------------------------------------------------------------ verification


def _prev_year(series: dict[int, float], year: int) -> int | None:
    earlier = [y for y in series if y < year]
    return max(earlier) if earlier else None


def _evidence(series: dict[int, float], *years: int | None) -> list[Evidence]:
    if not series:
        return []
    marks = [y for y in years if y is not None] or [max(series)]
    hi = max(max(marks), max(series)) if max(marks) >= max(series) - 3 else max(marks) + 3
    lo = min(min(marks), hi - 14)
    return [Evidence(y, series[y]) for y in sorted(series) if lo <= y <= hi]


def verify_claim(claim: Claim, source: DataSource, catalog: Catalog, ref_year: int = 2026) -> ClaimResult:
    def result(verdict: Verdict, explanation: str, **kw) -> ClaimResult:
        return ClaimResult(claim=claim, verdict=verdict, explanation=explanation, **kw)

    if claim.indicator is None:
        return result(Verdict.UNVERIFIABLE, "No official series found for this number (checked the catalog and the series list).")
    ind = catalog.indicators.get(claim.indicator)
    if ind is None:
        return result(Verdict.UNVERIFIABLE, f"Unknown series {claim.indicator}.")
    meta = dict(indicator_name=ind.name)
    if claim.unit and claim.unit.startswith("subgroup:"):
        group = claim.unit.split(":", 1)[1]
        return result(
            Verdict.UNVERIFIABLE,
            f"This claim is about a subgroup ({group}). The catalog's {ind.name} series covers the whole population, so it can't check it.",
            **meta,
        )
    if claim.unit == "aggregate":
        return result(
            Verdict.UNVERIFIABLE,
            f"This is an average or cumulative figure over several years. The checker compares single-year values of {ind.name}, so it doesn't verify these yet.",
            **meta,
        )
    if claim.unit and claim.unit.startswith("mismatch:"):
        return result(
            Verdict.UNVERIFIABLE,
            f"Found {ind.name}, but the number's unit doesn't fit that series.",
            **meta,
        )
    if claim.country is None:
        return result(Verdict.UNVERIFIABLE, "Couldn't tell which country this is about.", **meta)

    meta.update(
        country_name=catalog.country_name(claim.country),
        source_url=source.source_url(ind.code, claim.country) or None,
        source_name=source.name,
    )
    try:
        series = source.get_series(ind.code, claim.country)
    except SourceError as e:
        return result(Verdict.UNVERIFIABLE, f"Data source unavailable: {e}", **meta)
    if not series:
        return result(Verdict.UNVERIFIABLE, f"No {ind.name} data for {meta['country_name']}.", **meta)

    if claim.kind is ClaimKind.LEVEL:
        verdict, why, years = _verify_level(claim, ind, series, ref_year)
    else:
        verdict, why, years = _verify_change(claim, ind, series)
    return result(verdict, why, evidence=_evidence(series, *years), **meta)


def _verify_level(claim: Claim, ind: Indicator, series: dict[int, float], ref_year: int):
    latest = max(series)
    year = claim.year or latest
    d = claim.decimals
    claimed_txt = fmt(claim.value, ind.unit, d)
    if year not in series:
        if year > latest:
            return Verdict.UNVERIFIABLE, f"No official figure for {year} yet. The latest is {fmt(series[latest], ind.unit, None)} ({latest}).", (latest,)
        return Verdict.UNVERIFIABLE, f"No official figure for {year}.", (year,)

    actual = series[year]
    verdict = grade(claim.value, actual, d, claim.scale, claim.qualifier)
    actual_txt = fmt(actual, ind.unit, None)
    label = f"in {year}" if claim.year else f"for {year} (latest available)"

    if verdict is Verdict.SUPPORTED:
        why = f"Official figure {label}: {actual_txt}. Claimed {claimed_txt}."
    elif verdict is Verdict.INACCURATE:
        why = f"Close but off: official figure {label} is {actual_txt}, not {claimed_txt}."
    else:
        why = f"Official figure {label} is {actual_txt}, not {claimed_txt}."

    if claim.year is None and verdict is not Verdict.SUPPORTED:
        matches = [y for y in sorted(series, reverse=True) if y != year and grade(claim.value, series[y], d, claim.scale, claim.qualifier) is Verdict.SUPPORTED]
        if matches:
            y = matches[0]
            return (
                Verdict.MISLEADING,
                f"{claimed_txt} matches the {y} figure ({fmt(series[y], ind.unit, None)}), "
                f"but the latest ({latest}) is {actual_txt}. Presented as current, it's out of date.",
                (y, latest),
            )

    if claim.year is None and latest < ref_year - 2:
        why += f" Note: the newest official data is from {latest}."

    if claim.direction and verdict in (Verdict.SUPPORTED, Verdict.INACCURATE):
        base_year = claim.start_year or _prev_year(series, year)
        if base_year is not None and base_year in series:
            delta = actual - series[base_year]
            tol = rounding_tolerance(d, claim.scale)
            wrong = (claim.direction == "up" and delta < -tol) or (claim.direction == "down" and delta > tol)
            flat = abs(delta) <= tol / 5
            if wrong or flat:
                moved = "was unchanged" if flat else ("rose" if delta > 0 else "fell")
                return (
                    Verdict.MISLEADING,
                    f"The {year} figure ({actual_txt}) is about right, but it {moved} "
                    f"from {fmt(series[base_year], ind.unit, None)} in {base_year}, "
                    f"so the claimed {'rise' if claim.direction == 'up' else 'fall'} is wrong.",
                    (base_year, year),
                )
            return verdict, why + f" It {'rose' if delta > 0 else 'fell'} from {fmt(series[base_year], ind.unit, None)} in {base_year}.", (base_year, year)
    return verdict, why, (year,)


def _verify_change(claim: Claim, ind: Indicator, series: dict[int, float]):
    latest = max(series)
    end_year = claim.year or latest
    if end_year not in series:
        return Verdict.UNVERIFIABLE, f"No official figure for {end_year}.", (latest,)
    if claim.start_year:
        start_year = claim.start_year
    elif claim.lookback:
        start_year = end_year - claim.lookback
    else:
        start_year = _prev_year(series, end_year)
    if start_year is None or start_year not in series:
        return Verdict.UNVERIFIABLE, f"No official figure for {start_year or 'the start year'}.", (end_year,)
    if start_year >= end_year:
        return Verdict.UNVERIFIABLE, "Couldn't work out the time period of this change.", (end_year,)

    s, e = series[start_year], series[end_year]
    d = claim.decimals
    delta = e - s
    period = f"{start_year} to {end_year}"
    moved = "rose" if delta > 0 else "fell" if delta < 0 else "was unchanged"
    actual_txt = f"{fmt(s, ind.unit, None)} in {start_year} to {fmt(e, ind.unit, None)} in {end_year}"

    # Direction is checked before magnitude: "fell" when it rose is wrong
    # regardless of the number.
    noise = rounding_tolerance(d, claim.scale) / 5
    if claim.direction == "up" and delta < -noise or claim.direction == "down" and delta > noise:
        word = "rise" if claim.direction == "up" else "fall"
        return Verdict.CONTRADICTED, f"Claimed a {word}, but it {moved} ({actual_txt}).", (start_year, end_year)

    # "from A to B": check both endpoints.
    if claim.start_value is not None:
        v1 = grade(claim.start_value, s, d, claim.scale, None)
        v2 = grade(claim.value, e, d, claim.scale, claim.qualifier)
        v = worst(v1, v2)
        claimed_txt = f"{fmt(claim.start_value, ind.unit, d)} to {fmt(claim.value, ind.unit, d)}"
        if v is Verdict.SUPPORTED:
            return v, f"Official figures: {actual_txt}. Claimed {claimed_txt}.", (start_year, end_year)
        return v, f"Official figures are {actual_txt}; claimed {claimed_txt}.", (start_year, end_year)

    # "by X": compare the size of the change, under each plausible reading.
    readings: list[tuple[str, float, str]] = []
    if claim.unit in ("pp", "pct_or_pp"):
        readings.append(("percentage points", delta, "pp"))
    if claim.unit in ("rel%", "pct_or_pp") and s != 0:
        readings.append(("percent", delta / s * 100.0, "rel%"))
    if claim.unit in ("count", "usd", "years", "per1000", "per100", "per100k", "number"):
        readings.append(("absolute", delta, claim.unit))
    if not readings:
        return Verdict.UNVERIFIABLE, "Couldn't interpret the size of this change.", (start_year, end_year)

    graded = [
        (grade(claim.value, abs(val), d, claim.scale, claim.qualifier), name, val, u)
        for name, val, u in readings
    ]
    graded.sort(key=lambda g: SEVERITY[g[0]])
    v, name, val, u = graded[0]
    if u == "pp":
        shown = f"{abs(val):.{max(d + 1, 1)}f} percentage points"
    elif u == "rel%":
        shown = f"{abs(val):.{max(d + 1, 1)}f}%"
    else:
        shown = _fmt_delta(abs(val), ind.unit, d).lstrip("+")
    if v is Verdict.SUPPORTED:
        why = f"It {moved} by {shown} from {period} ({actual_txt})."
    else:
        why = f"It {moved} by {shown} from {period} ({actual_txt}), not by {claim.value:g}{'' if u not in ('pp', 'rel%') else (' points' if u == 'pp' else '%')}."
    if len(readings) > 1:
        why += f" Read as {name}."
    return v, why, (start_year, end_year)
