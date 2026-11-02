"""World Bank World Development Indicators API (no key required).

Docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .base import SourceError

API = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"

# The data.worldbank.org website takes ISO2 codes in ?locations=.
ISO3_TO_ISO2 = {
    "IND": "IN", "USA": "US", "GBR": "GB", "CHN": "CN", "PAK": "PK", "BGD": "BD",
    "LKA": "LK", "NPL": "NP", "IDN": "ID", "BRA": "BR", "NGA": "NG", "KEN": "KE",
    "ZAF": "ZA", "DEU": "DE", "FRA": "FR", "JPN": "JP", "CAN": "CA", "AUS": "AU",
    "MEX": "MX", "PHL": "PH", "VNM": "VN", "EGY": "EG", "TUR": "TR", "WLD": "1W",
}


class WorldBankSource:
    name = "World Bank, World Development Indicators"

    def __init__(
        self,
        cache_dir: str | os.PathLike | None = None,
        ttl_seconds: int = 7 * 24 * 3600,
        timeout: float = 15.0,
    ) -> None:
        default = Path(os.environ.get("VALIDAI_CACHE", Path.home() / ".cache" / "validai"))
        self.cache_dir = Path(cache_dir) if cache_dir else default
        self.ttl = ttl_seconds
        self.timeout = timeout
        self._mem: dict[tuple[str, str], dict[int, float]] = {}

    def extra_countries(self) -> dict[str, tuple[str, tuple[str, ...]]]:
        return {}

    def source_url(self, indicator: str, country: str) -> str:
        iso2 = ISO3_TO_ISO2.get(country, country)
        return f"https://data.worldbank.org/indicator/{indicator}?locations={urllib.parse.quote(iso2)}"

    def get_series(self, indicator: str, country: str) -> dict[int, float]:
        key = (indicator, country)
        if key in self._mem:
            return self._mem[key]
        cache_file = self.cache_dir / f"{country}_{indicator}.json"
        if cache_file.exists() and time.time() - cache_file.stat().st_mtime < self.ttl:
            series = {int(k): v for k, v in json.loads(cache_file.read_text()).items()}
        else:
            series = self._fetch(indicator, country)
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(series))
            except OSError:
                pass  # caching is best-effort
        self._mem[key] = series
        return series

    def _fetch(self, indicator: str, country: str) -> dict[int, float]:
        url = API.format(
            country=urllib.parse.quote(country), indicator=urllib.parse.quote(indicator)
        ) + "?format=json&per_page=200"
        req = urllib.request.Request(url, headers={"User-Agent": "validai/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            raise SourceError(f"World Bank API request failed: {e}") from e
        return parse_response(payload)


def parse_response(payload: object) -> dict[int, float]:
    """Parse the ``[meta, rows]`` JSON the API returns."""
    if not isinstance(payload, list) or not payload:
        raise SourceError("unexpected World Bank response")
    if isinstance(payload[0], dict) and "message" in payload[0]:
        msg = payload[0]["message"]
        raise SourceError(f"World Bank API error: {msg}")
    if len(payload) < 2 or payload[1] is None:
        return {}
    out: dict[int, float] = {}
    for row in payload[1]:
        try:
            year = int(row["date"])
        except (KeyError, TypeError, ValueError):
            continue
        if row.get("value") is not None:
            out[year] = float(row["value"])
    return out
