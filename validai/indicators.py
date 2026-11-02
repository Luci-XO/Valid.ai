"""Download the full list of World Bank WDI series for the series finder.

    python -m validai.indicators --out data/indicators.json

The bundled starter list has about 50 common series. The full list has
about 1,500, so the finder can place many more claims. The finder loads
data/indicators.json automatically when it exists.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://api.worldbank.org/v2/indicator?format=json&per_page=20000&source=2"  # source 2 = WDI


def parse(payload: object) -> list[dict]:
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        raise ValueError("unexpected World Bank response")
    out = []
    for row in payload[1]:
        code, name = row.get("id"), row.get("name")
        if not code or not name:
            continue
        topics = " ".join(t.get("value", "") for t in row.get("topics") or [] if isinstance(t, dict))
        out.append({"code": code, "name": name.strip(), "keywords": topics.strip()})
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/indicators.json")
    args = p.parse_args(argv)
    req = urllib.request.Request(URL, headers={"User-Agent": "validai/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            entries = parse(json.loads(r.read().decode("utf-8")))
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": URL, "indicators": entries}, indent=0))
    print(f"Wrote {len(entries)} series to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
