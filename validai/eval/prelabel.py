"""Turn real sentences into a labeling file, pre-filled with Valid.ai's guesses.

    python -m validai.eval.prelabel --input data/real_sentences.txt --country IND \
        --source snapshot:data/snapshot.json --out data/real_claims.jsonl

Input: one sentence (or short passage) per line, copied from speeches, news,
or posts. Lines starting with # are skipped. Add the source after " | " to
keep a record of where each one came from:

    India's unemployment rate fell to 3.2% in 2023. | PIB press release, 2024-05-02

Output: the JSONL format used by ``validai.eval.run``, with each "expected"
verdict copied from Valid.ai's current answer and "reviewed": false.

IMPORTANT: these pre-filled labels are guesses. Before you score anything,
open each item, check the number against the official source yourself, fix
"expected", and set "reviewed": true. ``run.py --reviewed-only`` skips
anything you haven't checked, so the system's guesses can't grade themselves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..finder import SeriesFinder
from ..pipeline import check_text
from ..sources import from_spec


def prelabel(lines: list[str], source_spec: str, country: str, use_finder: bool = True) -> list[dict]:
    source = from_spec(source_spec)
    finder = SeriesFinder(llm=None) if use_finder else None
    items = []
    for n, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        text, _, origin = line.partition(" | ")
        r = check_text(text.strip(), source, default_country=country, finder=finder)
        items.append({
            "id": f"real-{n:04d}",
            "text": text.strip(),
            "country": country,
            "origin": origin.strip() or None,
            "expected": [{"verdict": c["verdict"], "indicator": c["claim"]["indicator"]} for c in r["claims"]],
            "system_notes": [c["explanation"] for c in r["claims"]],
            "reviewed": False,
        })
    return items


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="data/real_claims.jsonl")
    p.add_argument("--country", default="IND")
    p.add_argument("--source", default="worldbank")
    args = p.parse_args(argv)

    items = prelabel(Path(args.input).read_text(encoding="utf-8").splitlines(), args.source, args.country)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"Wrote {len(items)} items to {out}. Review each one and set \"reviewed\": true before scoring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
