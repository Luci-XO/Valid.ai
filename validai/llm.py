"""Optional LLM extractor (any OpenAI-compatible chat API).

The model only reads text and labels it: which span is a claim, which series
and country it refers to, which years. It never supplies the numbers we
check. Each number it reports must appear verbatim in the span, and we
re-parse it with the same parser the rule extractor uses.

No API key is needed by default: Valid.ai talks to a local model served by
Ollama (https://ollama.com) on this machine, so no text leaves the laptop.

    ollama pull qwen2.5:3b      # once, about 2 GB
    ollama serve                # usually already running after install

Environment variables (all optional):
  VALIDAI_LLM_BASE_URL  default http://localhost:11434/v1 (Ollama)
  VALIDAI_LLM_MODEL     default qwen2.5:3b
  VALIDAI_LLM_API_KEY   only for hosted APIs; if set without a base URL,
                        the OpenAI endpoint is used
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from .catalog import Catalog
from .extract import NumTok, _compatible, parse_numbers
from .models import Claim, ClaimKind


OLLAMA_URL = "http://localhost:11434/v1"
DEFAULT_LOCAL_MODEL = "qwen2.5:3b"


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        transport=None,
        enabled: bool = True,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("VALIDAI_LLM_API_KEY", "")
        default_base = "https://api.openai.com/v1" if self.api_key else OLLAMA_URL
        self.base_url = (base_url or os.environ.get("VALIDAI_LLM_BASE_URL") or default_base).rstrip("/")
        self.model = model or os.environ.get("VALIDAI_LLM_MODEL") or DEFAULT_LOCAL_MODEL
        self.timeout = timeout
        self.enabled = enabled
        self._transport = transport  # for tests: callable(url, headers, body) -> dict
        self._probed: bool | None = None
        self.missing_model = False  # Ollama is running but the model isn't pulled

    @property
    def is_local(self) -> bool:
        return self.base_url.startswith(("http://localhost", "http://127.0.0.1"))

    @property
    def available(self) -> bool:
        """True if a model is reachable. Probes the local server once (1.5 s timeout)."""
        if not self.enabled:
            return False
        if self._transport is not None:
            return True
        if self._probed is None:
            self._probed = self._probe()
        return self._probed

    def _probe(self) -> bool:
        if not self.is_local:
            return bool(self.api_key)
        try:
            with urllib.request.urlopen(f"{self.base_url}/models", timeout=1.5) as r:
                models = json.loads(r.read().decode()).get("data", [])
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, AttributeError):
            return False
        names = {m.get("id", "") for m in models if isinstance(m, dict)}
        if self.model not in names and f"{self.model}:latest" not in names:
            self.missing_model = True
            return False
        return True

    def describe(self) -> str:
        where = "local" if self.is_local else "hosted"
        return f"{self.model} ({where})"

    def chat_json(self, system: str, user: str) -> dict:
        if not self.available:
            raise LLMError(
                f"no model reachable at {self.base_url}. Install Ollama and run `ollama pull {self.model}`"
            )
        body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self._transport:
            resp = self._transport(url, headers, body)
        else:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.loads(r.read().decode())
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                raise LLMError(f"request failed: {e}") from e
        try:
            content = resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError("malformed response") from e
        return parse_json_content(content)


def parse_json_content(content: str) -> dict:
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError("model did not return JSON") from e
    if not isinstance(data, dict):
        raise LLMError("model returned non-object JSON")
    return data


SYSTEM = """You extract checkable statistical claims from text. Output JSON only.

Return {"claims": [ ... ]}. One object per claim:
- "span": the exact substring of the input containing the claim (copy it character for character)
- "indicator": one code from the catalog below, or null if the number isn't about any listed series
- "country": ISO3 code, or null if unclear
- "kind": "level" (a value at a point in time) or "change" (a rise/fall by an amount, or from A to B)
- "number_as_written": the claimed number exactly as written, with its unit, e.g. "4.2%", "$3.5 trillion", "1.2 percentage points"
- "start_number_as_written": for "from A to B" claims, A as written; otherwise null
- "year": the year the claim refers to (end year for changes), or null if it means "now/latest"
- "start_year": start year for changes or "since YEAR", else null
- "direction": "up", "down", or null
- "negative": true only if the text says a growth rate was negative (e.g. "the economy contracted 5.8%")
- "qualifier": "approx", "nearly", "over", "under", or null
- "change_unit": for change claims: "pp" (percentage points), "percent" (relative), "absolute", or "ambiguous"

Rules: never invent numbers; skip sentences without a number. "GDP grew 7%" is a level claim about GDP growth.
Inflation and GDP growth are already rates, so "inflation rose 6.7%" is a level of 6.7.
"""


def _catalog_text(catalog: Catalog) -> str:
    inds = "\n".join(f"{i.code}: {i.name}" for i in catalog.indicators.values())
    ctys = ", ".join(f"{c}={n}" for c, (n, _) in catalog.countries.items())
    return f"Catalog of indicators:\n{inds}\n\nCountries: {ctys}"


def extract_claims_llm(
    text: str,
    catalog: Catalog,
    client: LLMClient | None = None,
    default_country: str | None = None,
    ref_year: int = 2026,
) -> list[Claim]:
    client = client or LLMClient()
    user = f"{_catalog_text(catalog)}\n\nDefault country if none is named: {default_country or 'none'}\nCurrent year: {ref_year}\n\nTEXT:\n{text}"
    data = client.chat_json(SYSTEM, user)
    raw = data.get("claims")
    if not isinstance(raw, list):
        raise LLMError("response has no 'claims' list")
    claims = [c for c in (validate_llm_claim(item, text, catalog, default_country, ref_year) for item in raw) if c]
    claims.sort(key=lambda c: c.start)
    return claims


def _one_number(written: object, span: str) -> NumTok | None:
    if not isinstance(written, str) or not written.strip():
        return None
    toks = parse_numbers(written)
    if len(toks) != 1:
        return None
    digits = re.search(r"\d[\d,]*(?:\.\d+)?", written)
    if not digits or digits.group(0) not in span:
        return None  # the model reported a number that isn't in the text
    return toks[0]


def _year(v: object, ref_year: int) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and 1900 <= int(v) <= ref_year + 1:
        return int(v)
    if isinstance(v, str) and v.isdigit():
        return _year(int(v), ref_year)
    return None


def validate_llm_claim(item: object, text: str, catalog: Catalog, default_country: str | None, ref_year: int) -> Claim | None:
    if not isinstance(item, dict):
        return None
    span = item.get("span")
    if not isinstance(span, str) or not span.strip():
        return None
    start = text.find(span)
    if start < 0:
        start = text.lower().find(span.lower())
    if start < 0:
        return None
    span_text = text[start : start + len(span)]

    tok = _one_number(item.get("number_as_written"), span_text)
    if tok is None:
        return None

    code = item.get("indicator")
    ind = catalog.indicators.get(code) if isinstance(code, str) else None
    country = item.get("country") if item.get("country") in catalog.countries else default_country
    kind = ClaimKind.CHANGE if item.get("kind") == "change" else ClaimKind.LEVEL
    direction = item.get("direction") if item.get("direction") in ("up", "down") else None
    qualifier = item.get("qualifier") if item.get("qualifier") in ("approx", "nearly", "over", "under") else None
    base = dict(text=span_text, start=start, end=start + len(span_text), country=country, decimals=tok.decimals, scale=tok.scale, qualifier=qualifier)

    if ind is None:
        return Claim(kind=ClaimKind.LEVEL, value=tok.value, **base)
    if not _compatible(ind, tok):
        return Claim(kind=ClaimKind.LEVEL, indicator=ind.code, value=tok.value, unit=f"mismatch:{tok.unit}", **base)

    year = _year(item.get("year"), ref_year)
    start_year = _year(item.get("start_year"), ref_year)

    if kind is ClaimKind.CHANGE:
        start_tok = _one_number(item.get("start_number_as_written"), span_text)
        if start_tok is not None and _compatible(ind, start_tok):
            return Claim(kind=kind, indicator=ind.code, value=tok.value, start_value=start_tok.value, unit=ind.unit, direction=direction or ("up" if tok.value > start_tok.value else "down"), year=year, start_year=start_year, since=start_year is not None and year is None, **base)
        cu = item.get("change_unit")
        if ind.unit == "percent":
            unit = {"pp": "pp", "percent": "rel%"}.get(cu, "pp" if tok.unit == "pp" else "pct_or_pp")
        else:
            unit = "rel%" if tok.unit == "pct" else ind.unit
        return Claim(kind=kind, indicator=ind.code, value=abs(tok.value), unit=unit, direction=direction, year=year, start_year=start_year, since=start_year is not None and year is None, **base)

    value = -abs(tok.value) if item.get("negative") is True and ind.rate_of_change else tok.value
    return Claim(kind=kind, indicator=ind.code, value=value, unit=ind.unit, direction=direction if not ind.rate_of_change else None, year=year, start_year=start_year, **base)
