# Valid.ai

**An AI fact-checker for numbers that can't make numbers up.**

Paste a speech, news article, or post. Valid.ai finds each number claim, works out which official statistic it refers to, fetches the real figure from the World Bank, and gives a verdict with a link to the source.

It runs **entirely on your laptop with no API key**. The AI is an open model served locally by [Ollama](https://ollama.com), so no text leaves the machine.

## How it works

```
"Youth unemployment fell to 15% in 2023"
        │
        ▼
1. Find claims             rules, or the local AI           → number 15, unit %, year 2023, direction down
2. Find the right series   catalog → keyword search → AI    → SL.UEM.1524.ZS "Unemployment, youth total"
3. Fetch the real data     World Bank API / offline snapshot → 2022: 15.8, 2023: 15.1
4. Check it in code        rounding rules, direction, years  → SUPPORTED, with a source link
```

The AI handles reading (steps 1 and 2). Code handles numbers (steps 3 and 4). This split is the core design:

- **The AI can't invent a number.** Every number it reports must appear in the input text, and it's re-parsed by code.
- **The AI can't invent a series.** It picks from a shortlist found by keyword search, or answers "none fits". An answer outside the list is rejected.
- **If the AI is off or fails**, Valid.ai still works using rules and keyword search, and the UI shows which method was used.

## Verdicts

| Verdict | Meaning | Example |
|---|---|---|
| supported | Matches official data within rounding | "4.2%" when the data says 4.17% |
| inaccurate | Close, but outside rounding | "9%" when the data says 8.2% |
| contradicted | Clearly wrong, or the change went the other way | "rose by 2 points" when it fell |
| misleading | A real figure framed wrongly | an old year's number described as "currently" |
| unverifiable | No matching series, year, or country | "metro ridership", a year with no data |

## Run it

You need Python 3.10 or newer. There are no Python dependencies.

```bash
git clone https://github.com/Luci-XO/Valid.ai.git && cd Valid.ai

# Optional but recommended: turn on the local AI (one time, about 2 GB)
#   install Ollama from https://ollama.com, then:
ollama pull qwen2.5:3b

# Download real data once, so the demo works without venue Wi-Fi
python -m validai.snapshot --countries IND,USA,WLD --with-finder-series --out data/snapshot.json
python -m validai.indicators --out data/indicators.json     # full WDI list (~1,500 series) for the finder

# Start the web UI at http://127.0.0.1:8000
python -m validai.server --source snapshot:data/snapshot.json --country IND
#   or use live World Bank data (cached for 7 days):
python -m validai.server --source worldbank --country IND
```

The startup line and the top of the page both say whether the local AI is on.

Settings (all optional): `VALIDAI_LLM_MODEL` (default `qwen2.5:3b`; any Ollama model works), `VALIDAI_LLM_BASE_URL` (default `http://localhost:11434/v1`), and `--no-ai` to turn it off. A hosted API also works if you set `VALIDAI_LLM_API_KEY`, but you don't need one.

API: `POST /api/check` with `{"text": "...", "country": "IND", "extractor": "llm" | "rules", "find_series": true}`.

## Evaluate it

```bash
python -m validai.eval.generate --source snapshot:data/snapshot.json --out data/eval.jsonl
python -m validai.eval.run --data data/eval.jsonl --source snapshot:data/snapshot.json \
    --systems rules,search,ai,llm_judge --out eval_results
```

| System | What it is |
|---|---|
| `rules` | rule extractor + built-in catalog of 12 series |
| `search` | + keyword series finder |
| `ai` | local LLM extractor + LLM series finder (full Valid.ai) |
| `llm_judge` | **baseline:** the same local LLM answers from memory, with no data lookup |

To build a real-world test set quickly, put one sentence per line in a text file and run `python -m validai.eval.prelabel --input data/real_sentences.txt --out data/real_claims.jsonl`. This pre-fills each label with Valid.ai's current answer and marks it `"reviewed": false`. Check every item against the official source, correct it, set `"reviewed": true`, then score with `--reviewed-only` so unchecked guesses never count.

`ai` vs `llm_judge` is the headline comparison. It shows what happens when the same model has to check against real data instead of answering from memory.

Current results on `tests/fixtures/hard_cases.jsonl` (49 hand-written phrasings, fictional data):

| Metric | rules | search |
|---|---|---|
| Verdict accuracy | 90.0% | 98.0% |
| Correct series picked | 85.4% | 100.0% |
| True claims wrongly flagged | 0% | 0% |
| False alarms on text with no claims | 0% | 0% |

The `ai` and `llm_judge` columns aren't filled in yet: run them on a machine with Ollama installed. The generated set scores close to 100% because its templates match the parser, so don't present that as real-world accuracy. **For the hackathon, label 150 or more real claims** from Indian speeches and news in the JSONL format described in `validai/eval/run.py`, and report all four systems on that set.

See [DEMO.md](DEMO.md) for the pitch outline and pre-demo checklist.

## Layout

```
validai/
  catalog.py      built-in indicators (aliases -> World Bank codes) and countries
  extract.py      rule-based claim extraction
  llm.py          local LLM client (Ollama, OpenAI-compatible) + AI extractor with validation
  finder.py       series finder: BM25 keyword search + AI pick from a shortlist
  verify.py       deterministic verdicts and explanations
  pipeline.py     check_text(): extract -> find series -> verify
  sources/        worldbank.py (live API + cache), snapshot.py (offline JSON)
  snapshot.py     CLI: download data for offline demos
  indicators.py   CLI: download the full WDI series list
  server.py       stdlib HTTP server + JSON API
  static/         single-page UI
  eval/           generate.py (labeled set), run.py (scoring + baselines)
tests/            pytest; fixtures use a fictional country (Freedonia)
```

## Known limits

- World Bank data lags by one to two years, so claims about last year are often unverifiable. Adding national sources (RBI, MOSPI) in the snapshot format is the next step.
- Multi-year averages are refused rather than checked.
- Fiscal years such as "2022-23" are read as the first calendar year.
- Input must be in English.
- The local-AI code path is tested with a simulated model. It hasn't yet been run against a real Ollama model.

## Tests

```bash
pip install pytest && python -m pytest -q
```
