# Demo guide

A 3-minute pitch plus a checklist so the demo doesn't fail on stage.

## The day before

- [ ] `ollama pull qwen2.5:3b`, then confirm the server's startup line says `AI: qwen2.5:3b (local)`
- [ ] `python -m validai.snapshot --countries IND,USA,WLD --with-finder-series --out data/snapshot.json`
- [ ] `python -m validai.indicators --out data/indicators.json`
- [ ] Turn Wi-Fi off and run `python -m validai.server --source snapshot:data/snapshot.json --country IND`. Everything should still work.
- [ ] Prepare 3 real quotes from the past month (a speech, a news article, a viral post). Check each against the official source yourself, so you know the right answers before you go on stage.
- [ ] Fill in the results slide from `python -m validai.eval.run ... --systems rules,search,ai,llm_judge --reviewed-only` on your labeled real claims.

## The pitch (about 3 minutes)

**1. Problem (20 s).** Politicians and viral posts quote numbers that sound official. Checking one takes a journalist 20 minutes: find the right dataset, the right year, the right definition. Most numbers never get checked.

**2. Why not just ask ChatGPT? (20 s).** Ask a chatbot "is unemployment 3.2%?" and it answers from memory. It can be out of date or simply wrong, and it gives no source. A fact-checker that makes up numbers is part of the problem.

**3. Live demo (90 s).**
- Paste a real quote. Point out the colored underlines and the verdicts.
- Open a **misleading** card: "This number is real, but it's from 2019 and was presented as current."
- Open a card with a **"Series found by local AI"** box: "The AI worked out that this is youth unemployment, not the national rate, picked the official series, and the code fetched the real figure."
- Mention that the Wi-Fi is off: "The whole thing runs on this laptop. No API key, and no text leaves the machine."

**4. Results (30 s).** One slide, one table: the same model answering from memory (`llm_judge`) vs Valid.ai (`ai`) on your labeled real claims. Accuracy, false claims caught, and true claims wrongly flagged.

**5. Close (20 s).** Next steps: RBI and MOSPI data for recent years, plus Hindi input.

## Questions judges will ask

- **"Did you train the model?"** No. It's an open model, Qwen 2.5 3B, running locally. Our work is the pipeline that stops it from making up numbers, plus the evaluation.
- **"What if the AI picks the wrong series?"** It can only choose from a shortlist found by search, and the card shows which series was used and why, so a reader can check it. Our series-match accuracy is on the results slide.
- **"How is this different from StatCite?"** StatCite gives developers international economic series. Valid.ai reads raw text from speeches and posts, handles framing problems like stale numbers, wrong direction, and subgroups, and runs offline. The national data sources are next.
- **"Why is this claim unverifiable?"** Official data lags by one to two years, and we'd rather say "can't check" than guess.
