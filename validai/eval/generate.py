"""Generate a labeled test set from a data snapshot.

    python -m validai.eval.generate --source snapshot:data/snapshot.json --out data/eval.jsonl

Each item is one sentence with one claim and the verdict it *should* get.
Labels come from how the sentence was built (true value, value off by 35%,
an old year's value presented as current...), not from running the
verifier, so the test isn't circular.

What this measures: whether extraction + verification work across many
phrasings, units, and series. What it doesn't: how well the system does on
real-world text. For that, label real claims (see README) and run the same
harness on them.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ..catalog import Catalog, Indicator
from ..pipeline import catalog_for
from ..sources import from_spec

REF_YEAR = 2026


def _fmt(v: float, ind: Indicator, decimals: int = 1) -> str:
    if ind.unit == "percent":
        return f"{v:.{decimals}f}%"
    if ind.unit == "years":
        return f"{v:.{decimals}f} years"
    if ind.unit == "per1000":
        return f"{v:.{decimals}f} per 1,000 live births"
    if ind.unit in ("usd", "count"):
        pre = "$" if ind.unit == "usd" else ""
        a = abs(v)
        for size, word in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
            if a >= size:
                return f"{pre}{v / size:.2f} {word}"
        return f"{pre}{v:,.0f}"
    raise ValueError(ind.unit)


def _alias(ind: Indicator, rng: random.Random) -> str:
    good = [a for a in ind.aliases if a not in ("economy", "the economy", "poverty", "literacy")]
    return rng.choice(good[:3] or list(ind.aliases))


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


LEVEL_TEMPLATES = (
    "{C}'s {a} was {v} in {y}.",
    "In {y}, the {a} in {C} stood at {v}.",
    "Official data show {C}'s {a} at {v} in {y}.",
    "The {a} of {C} reached {v} in {y}, the government said.",
)
CURRENT_TEMPLATES = (
    "{C}'s {a} is currently {v}.",
    "Today the {a} in {C} is {v}.",
    "Right now, {C} has a {a} of {v}.",
)
NEGATIVES = (
    "The minister visited 14 districts in 2023.",
    "The scheme was launched in 2015 with great fanfare.",
    "Over 300 delegates attended the summit.",
    "We will build a better future for every family.",
    "The committee met on 12 March to review 45 proposals.",
    "Chapter 3 covers the period from 1990 to 2005.",
)
UNKNOWN_SERIES = (
    "Metro ridership rose 12% in {y}.",
    "Broadband subscriptions reached 85% of households in {y}.",
    "Crop insurance payouts grew to $4.2 billion in {y}.",
)


def generate(source_spec: str, n_per_series: int = 6, seed: int = 7) -> list[dict]:
    source = from_spec(source_spec)
    catalog: Catalog = catalog_for(source)
    rng = random.Random(seed)
    items: list[dict] = []

    def add(text: str, country: str, verdict: str | None, indicator: str | None, tag: str) -> None:
        expected = [] if verdict is None else [{"verdict": verdict, "indicator": indicator}]
        items.append({"id": f"{tag}-{len(items)}", "text": text, "country": country, "expected": expected, "tag": tag})

    countries = sorted({c for ind in catalog.indicators for c in _countries_with(source, ind)})
    # Only built-in series: ones the finder added at runtime have no aliases to write sentences with.
    for ind in (i for i in catalog.indicators.values() if i.aliases):
        for country in countries:
            series = source.get_series(ind.code, country)
            if len(series) < 3:
                continue
            cname = catalog.country_name(country)
            years = sorted(series)
            latest = years[-1]
            recent = [y for y in years if y >= latest - 15]
            for _ in range(n_per_series):
                y = rng.choice(recent)
                a = series[y]
                alias = _alias(ind, rng)
                tpl = rng.choice(LEVEL_TEMPLATES)
                dec = 2 if ind.unit in ("usd", "count") else 1

                # supported: the true value, rounded the way people round
                add(_cap(tpl.format(C=cname, a=alias, v=_fmt(a, ind, dec), y=y)), country, "supported", ind.code, "level-true")

                # contradicted: off by at least 35% (and at least 1 unit for small rates)
                if ind.unit in ("usd", "count"):
                    wrong = a * rng.choice((0.6, 1.4))
                else:
                    off = max(0.35 * abs(a), 1.0)
                    wrong = a + rng.choice((-1, 1)) * off
                    if wrong < 0 and not ind.rate_of_change:
                        wrong = a + off  # rates like unemployment can't be negative
                    if ind.unit == "percent" and not ind.rate_of_change and wrong > 100:
                        wrong = a - off  # nor can a share go above 100%
                add(_cap(tpl.format(C=cname, a=alias, v=_fmt(wrong, ind, dec), y=y)), country, "contradicted", ind.code, "level-wrong")

                # inaccurate: rounded to a whole number, then off by one
                if ind.unit in ("percent", "years", "per1000") and abs(a) >= 30:
                    r = round(a)
                    near = r + (1 if r >= a else -1)
                    v = f"{near}%" if ind.unit == "percent" else _fmt(near, ind, 0).replace(".0 ", " ")
                    add(_cap(tpl.format(C=cname, a=alias, v=v, y=y)), country, "inaccurate", ind.code, "level-close")

            # misleading: an old figure presented as current
            cur = series[latest]
            stale = [y for y in recent if y < latest and abs(series[y] - cur) > max(0.35 * abs(cur), 1.0)]
            if stale and ind.unit != "count":
                y = rng.choice(stale)
                dec = 2 if ind.unit == "usd" else 1
                add(_cap(rng.choice(CURRENT_TEMPLATES).format(C=cname, a=_alias(ind, rng), v=_fmt(series[y], ind, dec))), country, "misleading", ind.code, "stale")

            # changes between two years
            pairs = [(y1, y2) for y1 in recent for y2 in recent if y2 - y1 >= 3]
            rng.shuffle(pairs)
            for y1, y2 in pairs[:2]:
                s, e = series[y1], series[y2]
                delta = e - s
                alias = _alias(ind, rng)
                if ind.unit == "percent" and abs(delta) >= 0.5:
                    verb = "rose" if delta > 0 else "fell"
                    wrong_verb = "fell" if delta > 0 else "rose"
                    add(_cap(f"{cname}'s {alias} {verb} by {abs(delta):.1f} percentage points between {y1} and {y2}."), country, "supported", ind.code, "change-true")
                    add(_cap(f"{cname}'s {alias} {wrong_verb} by {abs(delta):.1f} percentage points between {y1} and {y2}."), country, "contradicted", ind.code, "change-direction")
                    add(_cap(f"The {alias} in {cname} went from {s:.1f}% in {y1} to {e:.1f}% in {y2}."), country, "supported", ind.code, "from-to")
                elif ind.unit != "percent" and s > 0 and abs(delta / s) >= 0.05:
                    rel = delta / s * 100
                    verb = "rose" if rel > 0 else "fell"
                    wrong_verb = "fell" if rel > 0 else "rose"
                    add(_cap(f"Between {y1} and {y2}, {cname}'s {alias} {verb} by {abs(rel):.0f}%."), country, "supported", ind.code, "change-true")
                    add(_cap(f"Between {y1} and {y2}, {cname}'s {alias} {wrong_verb} by {abs(rel):.0f}%."), country, "contradicted", ind.code, "change-direction")

            # unverifiable: years with no data
            alias = _alias(ind, rng)
            add(_cap(f"{cname}'s {alias} will be {_fmt(series[latest], ind, 1 if ind.unit not in ('usd', 'count') else 2)} in {REF_YEAR + 5}."), country, "unverifiable", ind.code, "future-year")

    some_country = countries[0] if countries else None
    for tpl in UNKNOWN_SERIES:
        add(tpl.format(y=REF_YEAR - 3), some_country, "unverifiable", None, "unknown-series")
    for text in NEGATIVES:
        add(text, some_country, None, None, "no-claim")
    rng.shuffle(items)
    return items


def _countries_with(source, code: str) -> list[str]:
    if hasattr(source, "countries"):
        return [c for c in source.countries() if source.get_series(code, c)]
    return []


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="snapshot:<path> (generation reads every series, so use a snapshot)")
    p.add_argument("--out", default="data/eval.jsonl")
    p.add_argument("--per-series", type=int, default=6)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)
    items = generate(args.source, args.per_series, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    tags: dict[str, int] = {}
    for it in items:
        tags[it["tag"]] = tags.get(it["tag"], 0) + 1
    print(f"Wrote {len(items)} items to {out}: " + ", ".join(f"{k}={v}" for k, v in sorted(tags.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
