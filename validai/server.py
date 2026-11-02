"""Small HTTP server: the web UI plus a JSON API. Standard library only.

    python -m validai.server --source worldbank --country IND
    python -m validai.server --source snapshot:data/snapshot.json

API
  GET  /api/meta   -> catalog, source, whether an LLM is configured
  POST /api/check  {"text": "...", "country": "IND", "extractor": "rules"|"llm"}
"""

from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .finder import SeriesFinder
from .llm import LLMClient
from .pipeline import catalog_for, check_text
from .sources import from_spec

STATIC = Path(__file__).parent / "static"
MAX_BODY = 50_000  # bytes; about 8,000 words


def make_handler(source, default_country: str | None, llm: LLMClient, finder: SeriesFinder | None = None):
    catalog = catalog_for(source)
    finder = finder if finder is not None else SeriesFinder(llm=llm)

    class Handler(BaseHTTPRequestHandler):
        server_version = "validai/0.1"

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj: object) -> None:
            self._send(status, json.dumps(obj).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/meta":
                self._json(200, {
                    "source": source.name,
                    "default_country": default_country,
                    "llm_available": llm.available,
                    "llm": llm.describe(),
                    "llm_missing_model": llm.missing_model,
                    "series_index_size": len(finder.index),
                    "countries": {c: n for c, (n, _) in catalog.countries.items()},
                    "indicators": {i.code: i.name for i in catalog.indicators.values()},
                })
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/check":
                return self._json(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._json(400, {"error": "bad Content-Length"})
            if length <= 0 or length > MAX_BODY:
                return self._json(413 if length > MAX_BODY else 400, {"error": f"body must be 1-{MAX_BODY} bytes"})
            try:
                req = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return self._json(400, {"error": "invalid JSON"})
            if not isinstance(req, dict) or not isinstance(req.get("text"), str) or not req["text"].strip():
                return self._json(400, {"error": "'text' is required"})
            country = req.get("country") or default_country
            if country is not None and country not in catalog.countries:
                return self._json(400, {"error": f"unknown country {country!r}"})
            extractor = req.get("extractor", "rules")
            if extractor not in ("rules", "llm"):
                return self._json(400, {"error": "extractor must be 'rules' or 'llm'"})
            use_finder = req.get("find_series", True) is not False
            try:
                result = check_text(
                    req["text"], source, extractor=extractor, default_country=country,
                    llm_client=llm, finder=finder if use_finder else None,
                )
            except Exception as e:  # keep the server up during a demo
                self.log_error("check failed: %r", e)
                return self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal error"})
            self._json(200, result)

    return Handler


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run the Valid.ai web UI")
    p.add_argument("--source", default=os.environ.get("VALIDAI_SOURCE", "worldbank"))
    p.add_argument("--country", default=os.environ.get("VALIDAI_COUNTRY", "IND"), help="default ISO3 country")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    p.add_argument("--no-ai", action="store_true", help="don't use a local LLM even if one is running")
    args = p.parse_args(argv)

    source = from_spec(args.source)
    p_llm = LLMClient(enabled=not args.no_ai)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(source, args.country, p_llm))
    ai = f"AI: {p_llm.describe()}" if p_llm.available else (
        f"AI: off (run `ollama pull {p_llm.model}` to enable)" if not args.no_ai else "AI: disabled"
    )
    print(f"Valid.ai on http://{args.host}:{args.port}  (source: {source.name}; {ai})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
