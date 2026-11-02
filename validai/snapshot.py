"""Download real World Bank data into an offline snapshot file.

    python -m validai.snapshot --countries IND,USA,WLD --out data/snapshot.json

Run this once with internet access, then demo with
``VALIDAI_SOURCE=snapshot:data/snapshot.json`` so venue Wi-Fi can't break it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from .catalog import COUNTRIES, INDICATORS
from .sources.base import SourceError
from .sources.worldbank import ISO3_TO_ISO2, WorldBankSource


def build_snapshot(countries: list[str], source: WorldBankSource | None = None, extra_codes: list[str] | None = None) -> dict:
    src = source or WorldBankSource()
    series: dict[str, dict[str, dict[str, float]]] = {}
    failures: list[str] = []
    codes = list(dict.fromkeys([i.code for i in INDICATORS] + list(extra_codes or [])))
    for code in codes:
        for c in countries:
            try:
                data = src.get_series(code, c)
            except SourceError as e:
                failures.append(f"{code}/{c}: {e}")
                continue
            if data:
                series.setdefault(code, {})[c] = {str(y): v for y, v in sorted(data.items())}
    return {
        "source_name": src.name,
        "source_url_template": "https://data.worldbank.org/indicator/{indicator}?locations={iso2}",
        "retrieved": dt.date.today().isoformat(),
        "countries": {
            c: {"name": COUNTRIES[c][0], "iso2": ISO3_TO_ISO2.get(c, c)} for c in countries if c in COUNTRIES
        },
        "series": series,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--countries", default="IND", help="comma-separated ISO3 codes (default: IND)")
    p.add_argument("--out", default="data/snapshot.json")
    p.add_argument("--with-finder-series", action="store_true",
                   help="also download every series the series finder knows, so found series work offline")
    args = p.parse_args(argv)

    countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]
    extra = None
    if args.with_finder_series:
        from .finder import SeriesIndex
        extra = [e["code"] for e in SeriesIndex.load().entries]
    snap = build_snapshot(countries, extra_codes=extra)
    n = sum(len(v) for v in snap["series"].values())
    if n == 0:
        print("No data downloaded. Check your internet connection.", file=sys.stderr)
        for f in snap["failures"][:5]:
            print("  " + f, file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snap, indent=1))
    print(f"Wrote {n} series to {out} ({len(snap['failures'])} failed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
