"""Tests use a FICTIONAL country (Freedonia) so no fake numbers look real."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from validai import check_text
from validai.eval.generate import generate
from validai.eval.run import load, run, score
from validai.extract import parse_numbers
from validai.llm import LLMClient, LLMError, extract_claims_llm
from validai.models import Verdict
from validai.pipeline import catalog_for
from validai.server import make_handler
from validai.snapshot import build_snapshot
from validai.sources import SnapshotSource, SourceError
from validai.sources.worldbank import parse_response
from validai.verify import grade

FIX = Path(__file__).parent / "fixtures"
SNAP = f"snapshot:{FIX / 'freedonia.json'}"


@pytest.fixture(scope="module")
def src():
    return SnapshotSource(FIX / "freedonia.json")


def verdicts(text, src):
    r = check_text(text, src, default_country="FRD", ref_year=2026)
    return [c["verdict"] for c in r["claims"]]


# ------------------------------------------------------------------ parsing


@pytest.mark.parametrize(
    "text,value,unit,decimals",
    [
        ("4.2%", 4.2, "pct", 1),
        ("4.2 per cent", 4.2, "pct", 1),
        ("1.5 percentage points", 1.5, "pp", 1),
        ("50 basis points", 0.5, "pp", 2),
        ("$3.9 trillion", 3.9e12, "usd", 1),
        ("1.44 billion", 1.44e9, None, 2),
        ("140 crore", 1.4e9, None, 0),
        ("72 years", 72.0, "years", 0),
        ("24.5 per 1,000 live births", 24.5, "per1000", 1),
        ("-5.8%", -5.8, "pct", 1),
        ("₹500 crore", 5e9, "inr", 0),
    ],
)
def test_parse_numbers(text, value, unit, decimals):
    (tok,) = parse_numbers(text)
    assert tok.value == pytest.approx(value)
    assert tok.unit == unit
    assert tok.decimals == decimals


def test_numbers_inside_names_are_ignored():
    assert parse_numbers("COVID-19 hit the G20") == []


# ------------------------------------------------------------------ grading


def test_grade_rounding_tiers():
    assert grade(4.2, 4.24, 1) is Verdict.SUPPORTED
    assert grade(4.2, 4.29, 1) is Verdict.INACCURATE  # within 2x rounding
    assert grade(4.2, 4.6, 1) is Verdict.CONTRADICTED
    assert grade(4, 4.4, 0) is Verdict.SUPPORTED
    assert grade(4, 4.9, 0) is Verdict.INACCURATE


def test_grade_qualifiers():
    assert grade(99, 99.5, 0, qualifier="over") is Verdict.SUPPORTED
    assert grade(25, 24.5, 0, qualifier="under") is Verdict.SUPPORTED
    assert grade(20, 24.5, 0, qualifier="under") is Verdict.CONTRADICTED
    assert grade(50, 47.5, 0, qualifier="nearly") is Verdict.SUPPORTED
    assert grade(50, 52, 0, qualifier="nearly") is Verdict.INACCURATE


# ------------------------------------------------------------------ end to end


def test_level_true_and_false(src):
    assert verdicts("Freedonia's unemployment rate was 4.2% in 2023.", src) == ["supported"]
    assert verdicts("Freedonia's unemployment rate was 6.5% in 2023.", src) == ["contradicted"]


def test_stale_number_is_misleading(src):
    r = check_text("Unemployment is currently 5.3%.", src, default_country="FRD", ref_year=2026)
    (c,) = r["claims"]
    assert c["verdict"] == "misleading"
    assert "2019" in c["explanation"]


def test_wrong_direction_is_contradicted(src):
    assert verdicts("Unemployment rose by 2 points since 2019.", src) == ["contradicted"]


def test_right_number_wrong_direction_is_misleading(src):
    # 2024 is 4.2 (correct) but it went up from 4.17 in 2023... within noise, so check a clear case:
    assert verdicts("Life expectancy fell to 72 years in 2023.", src) == ["misleading"]


def test_growth_redirect_and_contraction(src):
    assert verdicts("GDP grew 7% in 2022.", src) == ["supported"]
    assert verdicts("The economy contracted 5.8% in 2020.", src) == ["supported"]


def test_multiple_claims_and_clause_inheritance(src):
    assert verdicts("Inflation was 6.7% in 2022 and 5.7% in 2023.", src) == ["supported", "supported"]


def test_country_switch(src):
    r = check_text("Sylvania's unemployment rate is 6.6%, much higher than Freedonia's 4.2%.", src, default_country="FRD")
    assert [c["claim"]["country"] for c in r["claims"]] == ["SYL", "FRD"]
    assert [c["verdict"] for c in r["claims"]] == ["supported", "supported"]


def test_unknown_and_subgroup_claims_are_unverifiable(src):
    assert verdicts("Hospital beds rose 12% in 2023.", src) == ["unverifiable"]
    assert verdicts("Unemployment among youth is 15%.", src) == ["unverifiable"]  # no finder
    assert verdicts("Inflation has averaged 3% over the past five years.", src) == ["unverifiable"]


def test_non_claims_are_skipped(src):
    assert verdicts("The minister visited 14 districts in 2023.", src) == []
    assert verdicts("The economy added 2 million jobs.", src) in ([], ["unverifiable"])


def test_future_year_unverifiable(src):
    assert verdicts("Unemployment will be 3% in 2031.", src) == ["unverifiable"]


def test_offsets_point_at_claim_text(src):
    text = "Intro sentence here. Freedonia's unemployment rate was 4.2% in 2023."
    (c,) = check_text(text, src, default_country="FRD")["claims"]
    assert text[c["claim"]["start"] : c["claim"]["end"]] == c["claim"]["text"]


def test_hard_cases_floor(src):
    """Hand-written phrasings. The series finder should beat the catalog alone."""
    items = load(str(FIX / "hard_cases.jsonl"))
    res = run(str(FIX / "hard_cases.jsonl"), SNAP, ["rules", "search"])
    assert res["search"]["verdict_accuracy"] >= 0.97, res["search"]["errors"]
    assert res["search"]["verdict_accuracy"] > res["rules"]["verdict_accuracy"]
    assert res["rules"]["true_claims_wrongly_flagged"] == 0
    assert res["search"]["no_claim_false_alarm_rate"] == 0
    assert len(items) >= 45


def test_generated_set_scores(src):
    items = generate(SNAP, n_per_series=3, seed=1)
    preds = [
        [{"verdict": c["verdict"], "indicator": c["claim"]["indicator"]} for c in check_text(it["text"], src, default_country=it["country"], ref_year=2026)["claims"]]
        for it in items
    ]
    s = score(items, preds)
    assert s["claims"] > 50
    assert s["verdict_accuracy"] >= 0.95, s["errors"][:5]


# ------------------------------------------------------------------ LLM path


def _fake_llm(payload):
    def transport(url, headers, body):
        assert body["response_format"] == {"type": "json_object"}
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    return LLMClient(api_key="x", model="fake", transport=transport)


def test_llm_claims_are_validated(src):
    text = "Freedonia's unemployment rate was 4.2% in 2023. Inflation hit 9% too."
    client = _fake_llm({"claims": [
        {"span": "Freedonia's unemployment rate was 4.2% in 2023.", "indicator": "SL.UEM.TOTL.ZS", "country": "FRD", "kind": "level", "number_as_written": "4.2%", "year": 2023},
        # hallucinated number not in the span: must be dropped
        {"span": "Inflation hit 9% too.", "indicator": "FP.CPI.TOTL.ZG", "country": "FRD", "kind": "level", "number_as_written": "7.5%", "year": 2023},
        # span not in the text: must be dropped
        {"span": "GDP was $9 trillion", "indicator": "NY.GDP.MKTP.CD", "kind": "level", "number_as_written": "$9 trillion"},
        "garbage",
    ]})
    claims = extract_claims_llm(text, catalog_for(src), client, "FRD", 2026)
    assert len(claims) == 1
    assert claims[0].value == 4.2 and claims[0].year == 2023


def test_llm_failure_falls_back_to_rules(src):
    def boom(url, headers, body):
        return {"choices": [{"message": {"content": "not json"}}]}

    r = check_text("Unemployment was 4.2% in 2023.", src, extractor="llm", default_country="FRD", llm_client=LLMClient(api_key="x", model="m", transport=boom))
    assert r["extractor"].startswith("rules (LLM failed")
    assert [c["verdict"] for c in r["claims"]] == ["supported"]


def test_llm_unreachable_raises():
    client = LLMClient(api_key="", base_url="http://127.0.0.1:9/v1", model="m")
    assert client.available is False
    with pytest.raises(LLMError):
        client.chat_json("s", "u")


def test_llm_defaults_to_local_ollama(monkeypatch):
    for k in ("VALIDAI_LLM_API_KEY", "VALIDAI_LLM_BASE_URL", "VALIDAI_LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    c = LLMClient()
    assert c.base_url == "http://localhost:11434/v1" and c.is_local and not c.api_key


# ------------------------------------------------------------------ sources


def test_worldbank_parse_response():
    payload = [{"page": 1}, [{"date": "2023", "value": 4.17}, {"date": "2022", "value": None}, {"date": "x"}]]
    assert parse_response(payload) == {2023: 4.17}
    with pytest.raises(SourceError):
        parse_response([{"message": [{"id": "120", "value": "Invalid value"}]}])
    assert parse_response([{"page": 1}, None]) == {}


def test_build_snapshot_with_fake_source():
    class Fake:
        name = "fake"

        def get_series(self, code, c):
            if code == "SP.POP.TOTL":
                return {2023: 5.0}
            raise SourceError("nope")

    snap = build_snapshot(["IND"], Fake())
    assert snap["series"] == {"SP.POP.TOTL": {"IND": {"2023": 5.0}}}
    assert snap["failures"]


# ------------------------------------------------------------------ server


@pytest.fixture()
def server(src):
    from validai.llm import LLMClient as C

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(src, "FRD", C(enabled=False)))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_server_endpoints(server):
    with urllib.request.urlopen(server + "/") as r:
        assert b"Valid.ai" in r.read()
    with urllib.request.urlopen(server + "/api/meta") as r:
        meta = json.loads(r.read())
    assert meta["default_country"] == "FRD" and meta["llm_available"] is False
    status, data = _post(server + "/api/check", {"text": "Unemployment was 4.2% in 2023."})
    assert status == 200 and data["claims"][0]["verdict"] == "supported"
    status, data = _post(server + "/api/check", {"text": "Youth unemployment fell to 14.9% in 2024."})
    assert data["claims"][0]["series_match"]["code"] == "SL.UEM.1524.ZS"
    status, data = _post(server + "/api/check", {"text": "Youth unemployment fell to 14.9% in 2024.", "find_series": False})
    assert data["claims"][0]["series_match"] is None
    assert _post(server + "/api/check", {"text": ""})[0] == 400
    assert _post(server + "/api/check", {"text": "x", "country": "ZZZ"})[0] == 400
    assert _post(server + "/api/check", {"text": "x", "extractor": "magic"})[0] == 400


# ------------------------------------------------------------------ series finder

from validai.finder import SeriesFinder, SeriesIndex, tokenize, unit_from_name  # noqa: E402


def test_unit_from_name():
    assert unit_from_name("Hospital beds (per 1,000 people)") == ("per1000", False)
    assert unit_from_name("GDP growth (annual %)") == ("percent", True)
    assert unit_from_name("Personal remittances, received (current US$)") == ("usd", False)
    assert unit_from_name("Official exchange rate (LCU per US$, period average)") == ("number", False)
    assert unit_from_name("Maternal mortality ratio (modeled estimate, per 100,000 live births)") == ("per100k", False)


def test_tokenize_normalises():
    assert tokenize("Women's labour enrolment") == ["female", "labor", "enrollment"]


def test_search_finds_and_refuses():
    f = SeriesFinder(llm=None)
    ok = lambda ind: True  # noqa: E731
    assert f.find("Youth unemployment is 15%", ok).candidate.code == "SL.UEM.1524.ZS"
    assert f.find("Remittances reached $125 billion", ok).candidate.code == "BX.TRF.PWKR.CD.DT"
    assert f.find("Metro ridership grew 20%", ok) is None
    # unit filter: a dollar amount can't be a % of GDP series
    assert f.find("The government spent $12 billion on health", lambda i: i.unit == "usd") is None


def test_local_ai_picks_only_from_candidates(src):
    calls = []

    def pick(code):
        def transport(url, headers, body):
            calls.append(body["messages"][1]["content"])
            return {"choices": [{"message": {"content": json.dumps({"code": code, "reason": "same thing"})}}]}
        return LLMClient(api_key="", model="tiny", transport=transport, base_url="http://localhost:11434/v1")

    f = SeriesFinder(llm=pick("SL.UEM.1524.ZS"))
    r = check_text("Unemployment among youth is 15%.", src, default_country="FRD", finder=f)
    (c,) = r["claims"]
    assert c["verdict"] == "supported"
    assert c["series_match"]["method"] == "local AI (tiny)"
    assert "SL.UEM.1524.ZS" in calls[0]  # the model saw it as a candidate
    # off-list or null answers mean "no series", never a made-up code
    for bad in ("XX.FAKE.CODE", None):
        r = check_text("Unemployment among youth is 15%.", src, default_country="FRD", finder=SeriesFinder(llm=pick(bad)))
        assert r["claims"][0]["verdict"] == "unverifiable"
        assert r["claims"][0]["series_match"] is None


def test_share_of_gdp_is_not_gdp(src):
    r = check_text("Defence spending is 2.4% of GDP.", src, default_country="FRD", finder=SeriesFinder(llm=None))
    assert r["claims"][0]["series_match"]["code"] == "MS.MIL.XPND.GD.ZS"


def test_unit_a_few_words_after_number():
    (tok,) = parse_numbers("just 0.6 hospital beds per 1,000 people")
    assert tok.unit == "per1000" and tok.value == 0.6


def test_index_loads_seed():
    assert len(SeriesIndex.load()) >= 50
