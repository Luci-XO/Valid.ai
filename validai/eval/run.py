"""Score systems on a labeled JSONL set.

    python -m validai.eval.run --data data/eval.jsonl --source snapshot:data/snapshot.json \
        --systems rules,search,ai,llm_judge --out eval_results

Systems
  rules      rule extractor + built-in catalog + code verification
  search     rules + keyword series finder
  ai         local LLM extractor + LLM series finder + code verification (Valid.ai full)
  llm_judge  baseline: the same LLM answers from its own memory, no data lookup

JSONL format, one item per line:
  {"id": "x1", "text": "...", "country": "IND",
   "expected": [{"verdict": "contradicted", "indicator": "SL.UEM.TOTL.ZS"}]}
``expected`` is empty for text with no checkable claim. ``indicator`` may be
null or omitted when you only labeled the verdict.

Coarse scores collapse verdicts to true / false / unverifiable, which is how
most public fact-check datasets are labeled.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from ..finder import SeriesFinder
from ..llm import LLMClient, LLMError
from ..models import Verdict
from ..pipeline import check_text
from ..sources import from_spec

VERDICTS = [v.value for v in Verdict]
MISSED = "missed"  # expected a claim, system found none
COARSE = {
    "supported": "true",
    "inaccurate": "false",
    "contradicted": "false",
    "misleading": "false",
    "unverifiable": "unverifiable",
    MISSED: "unverifiable",
}

JUDGE_SYSTEM = """You are a fact-checker. For the statement, decide whether its numeric claim matches official
World Bank World Development Indicators data. Answer JSON: {"verdict": V, "indicator": CODE or null}
V is one of: supported (matches within rounding), inaccurate (close but off), contradicted (clearly wrong
or wrong direction), misleading (real number but outdated or framed wrongly), unverifiable (no data / not a statistic)."""


def load(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{n}: invalid JSON ({e})") from None
    return items


_FINDERS: dict = {}


def predict(system: str, item: dict, source, llm: LLMClient) -> list[dict]:
    """Return [{"verdict", "indicator"}] for one item."""
    if system in ("rules", "search", "ai"):
        finder = None
        if system in ("search", "ai"):
            key = (system, id(llm))
            if key not in _FINDERS:
                _FINDERS[key] = SeriesFinder(llm=llm if system == "ai" else None)
            finder = _FINDERS[key]
        extractor = "llm" if system == "ai" else "rules"
        r = check_text(item["text"], source, extractor=extractor, default_country=item.get("country"), llm_client=llm, finder=finder)
        if system == "ai" and r["extractor"] != "llm":
            raise LLMError(r["extractor"])
        return [{"verdict": c["verdict"], "indicator": c["claim"]["indicator"]} for c in r["claims"]]
    if system == "llm_judge":
        data = llm.chat_json(JUDGE_SYSTEM, f"Country: {item.get('country')}\nStatement: {item['text']}")
        v = data.get("verdict")
        if v not in VERDICTS:
            v = "unverifiable"
        return [{"verdict": v, "indicator": data.get("indicator")}]
    raise ValueError(system)


def score(items: list[dict], preds: list[list[dict]]) -> dict:
    exact = coarse = total = 0
    ind_right = ind_total = 0
    neg_total = neg_fp = 0
    confusion: Counter = Counter()
    by_tag: dict[str, list[int]] = {}
    errors: list[dict] = []

    for item, pred in zip(items, preds, strict=True):
        exp = item.get("expected", [])
        if not exp:
            neg_total += 1
            # Saying "unverifiable" about a non-claim is harmless; any other verdict is a false alarm.
            if any(p["verdict"] != "unverifiable" for p in pred):
                neg_fp += 1
                errors.append({"id": item["id"], "text": item["text"], "expected": "no claim", "got": pred})
            continue
        # Items are scored claim by claim in order. Extra predicted claims are ignored here
        # and show up in the false-alarm rate on no-claim items instead.
        for i, e in enumerate(exp):
            p = pred[i] if i < len(pred) else {"verdict": MISSED, "indicator": None}
            total += 1
            ok = p["verdict"] == e["verdict"]
            exact += ok
            coarse += COARSE[p["verdict"]] == COARSE[e["verdict"]]
            confusion[(e["verdict"], p["verdict"])] += 1
            tag = item.get("tag", "all")
            by_tag.setdefault(tag, [0, 0])
            by_tag[tag][0] += ok
            by_tag[tag][1] += 1
            if e.get("indicator"):
                ind_total += 1
                ind_right += p.get("indicator") == e["indicator"]
            if not ok:
                errors.append({"id": item["id"], "text": item["text"], "expected": e["verdict"], "got": p["verdict"]})

    def unv_stats():
        tp = confusion_sum(lambda e, p: e == "unverifiable" and p == "unverifiable")
        pred_u = confusion_sum(lambda e, p: p == "unverifiable")
        exp_u = confusion_sum(lambda e, p: e == "unverifiable")
        return (tp / pred_u if pred_u else None, tp / exp_u if exp_u else None)

    def confusion_sum(f):
        return sum(n for (e, p), n in confusion.items() if f(e, p))

    # Of claims that are actually false, how many did we flag as not supported?
    false_exp = confusion_sum(lambda e, p: COARSE[e] == "false")
    false_caught = confusion_sum(lambda e, p: COARSE[e] == "false" and COARSE[p] == "false")
    true_exp = confusion_sum(lambda e, p: e == "supported")
    true_flagged = confusion_sum(lambda e, p: e == "supported" and COARSE[p] == "false")
    up, ur = unv_stats()
    return {
        "claims": total,
        "verdict_accuracy": exact / total if total else None,
        "coarse_accuracy": coarse / total if total else None,
        "false_claims_caught": false_caught / false_exp if false_exp else None,
        "true_claims_wrongly_flagged": true_flagged / true_exp if true_exp else None,
        "series_match_accuracy": ind_right / ind_total if ind_total else None,
        "unverifiable_precision": up,
        "unverifiable_recall": ur,
        "no_claim_items": neg_total,
        "no_claim_false_alarm_rate": neg_fp / neg_total if neg_total else None,
        "by_tag": {t: {"accuracy": r / n, "n": n} for t, (r, n) in sorted(by_tag.items())},
        "confusion": {f"{e}->{p}": n for (e, p), n in sorted(confusion.items())},
        "errors": errors,
    }


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def report(results: dict[str, dict]) -> str:
    rows = [
        ("Verdict accuracy (5-way)", "verdict_accuracy"),
        ("Coarse accuracy (true/false/unverifiable)", "coarse_accuracy"),
        ("False claims caught", "false_claims_caught"),
        ("True claims wrongly flagged", "true_claims_wrongly_flagged"),
        ("Correct series picked", "series_match_accuracy"),
        ("'Unverifiable' precision", "unverifiable_precision"),
        ("'Unverifiable' recall", "unverifiable_recall"),
        ("False alarms on no-claim text", "no_claim_false_alarm_rate"),
    ]
    names = list(results)
    out = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for label, key in rows:
        out.append(f"| {label} | " + " | ".join(_pct(results[n][key]) for n in names) + " |")
    out.append("| Claims scored | " + " | ".join(str(results[n]["claims"]) for n in names) + " |")
    out.append("| Mean latency per item | " + " | ".join(f"{results[n]['latency_ms_mean']:.0f} ms" for n in names) + " |")
    out.append("| p95 latency per item | " + " | ".join(f"{results[n]['latency_ms_p95']:.0f} ms" for n in names) + " |")
    lines = ["\n".join(out), ""]
    for n in names:
        lines.append(f"Accuracy by item type ({n}):")
        for tag, s in results[n]["by_tag"].items():
            lines.append(f"  {tag:18} {_pct(s['accuracy']):>7}  (n={s['n']})")
        lines.append("")
    return "\n".join(lines)


def run(data: str, source_spec: str, systems: list[str], limit: int | None = None) -> dict[str, dict]:
    items = load(data)[:limit] if limit else load(data)
    source = from_spec(source_spec)
    llm = LLMClient()
    results: dict[str, dict] = {}
    for system in systems:
        if system in ("ai", "llm_judge") and not llm.available:
            print(f"skipping {system}: no LLM reachable (install Ollama and run `ollama pull {llm.model}`)", file=sys.stderr)
            continue
        preds, times = [], []
        for item in items:
            t = time.perf_counter()
            try:
                preds.append(predict(system, item, source, llm))
            except LLMError as e:
                print(f"{system} {item['id']}: {e}", file=sys.stderr)
                preds.append([])
            times.append((time.perf_counter() - t) * 1000)
        s = score(items, preds)
        s["latency_ms_mean"] = statistics.fmean(times) if times else 0.0
        s["latency_ms_p95"] = sorted(times)[int(0.95 * (len(times) - 1))] if times else 0.0
        results[system] = s
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--systems", default="rules,search,ai,llm_judge")
    p.add_argument("--limit", type=int)
    p.add_argument("--out", help="directory for results.json and report.md")
    args = p.parse_args(argv)

    results = run(args.data, args.source, [s.strip() for s in args.systems.split(",") if s.strip()], args.limit)
    if not results:
        print("No systems ran.", file=sys.stderr)
        return 1
    text = report(results)
    print(text)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(results, indent=1))
        (out / "report.md").write_text(text)
        print(f"Saved {out / 'results.json'} and {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
