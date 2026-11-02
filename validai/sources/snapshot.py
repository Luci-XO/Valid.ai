"""Offline snapshot of series, saved as JSON.

Use ``python -m validai.snapshot`` to download real World Bank data once,
then demo without internet. The file format also works for national
statistics you export yourself:

{
  "source_name": "World Bank, World Development Indicators",
  "source_url_template": "https://data.worldbank.org/indicator/{indicator}?locations={iso2}",
  "retrieved": "2026-11-01",
  "countries": {"FRD": {"name": "Freedonia", "aliases": ["freedonia"], "iso2": "FD"}},
  "series": {"SL.UEM.TOTL.ZS": {"IND": {"2022": 4.8, "2023": 4.2}}}
}
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .base import SourceError
from .worldbank import ISO3_TO_ISO2


class SnapshotSource:
    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise SourceError(f"could not read snapshot {self.path}: {e}") from e
        self.name = data.get("source_name", "snapshot")
        self.retrieved = data.get("retrieved")
        self._url_template = data.get("source_url_template")
        self._countries = data.get("countries", {})
        self._series = {
            ind: {cty: {int(y): float(v) for y, v in years.items() if v is not None} for cty, years in by_cty.items()}
            for ind, by_cty in data.get("series", {}).items()
        }

    def extra_countries(self) -> dict[str, tuple[str, tuple[str, ...]]]:
        return {
            code: (c.get("name", code), tuple(a.lower() for a in c.get("aliases", [])))
            for code, c in self._countries.items()
            if c.get("aliases")
        }

    def indicators(self) -> list[str]:
        return list(self._series)

    def countries(self) -> list[str]:
        return sorted({c for by_cty in self._series.values() for c in by_cty})

    def get_series(self, indicator: str, country: str) -> dict[int, float]:
        return dict(self._series.get(indicator, {}).get(country, {}))

    def source_url(self, indicator: str, country: str) -> str:
        if not self._url_template:
            return ""
        iso2 = self._countries.get(country, {}).get("iso2") or ISO3_TO_ISO2.get(country, country)
        return self._url_template.format(indicator=indicator, country=country, iso2=iso2)
