# metaculus-forecast-bot

A stdlib-only forecasting bot for the [Metaculus AI Benchmark](https://www.metaculus.com/aib/)
tournaments. No dependencies, no framework — it runs on a bare `python:3.x` image with no
install step, which is the whole point: the thing that decides your score is the forecast,
not the scaffolding around it.

It handles all four question types, and it will **abstain rather than submit a malformed
CDF**, because a rejected payload scores worse than silence.

```bash
export METACULUS_TOKEN=...           # bot account token, from your Metaculus settings page
export OPENROUTER_API_KEY=...        # or HF_INFERENCE_TOKEN, or METACULUS_PROXY_MODEL

python metaculus_bot/forecast.py --selftest                    # no token needed
python metaculus_bot/forecast.py --tournament minibench --dry-run
python metaculus_bot/forecast.py --tournament minibench --submit
python metaculus_bot/coverage.py                               # what is unanswered
```

## What this repo is really for

Most of the value here is not the code, it is the list of things that are true about this
API and cost a day to find out. Every one of them was measured, not assumed.

### A tournament that excludes bots will still let you forecast it

Each tournament object carries **`bot_leaderboard_status`**:

| value | meaning |
| --- | --- |
| `bots_only`, `include` | you are ranked for prizes |
| `exclude_and_show`, `exclude_and_hide` | you may forecast, and can win nothing |

The Metaculus Cup is `exclude_and_show`. Forecasting it is a fine calibration benchmark
against humans and it is not income. `coverage.py` reads this live and labels each
tournament, because nothing warns you.

### `my_forecasts` is absent from the list endpoint

`GET /posts/?...` does not include `question.my_forecasts` at all — not empty, absent. Read
coverage off the list and you will get a confident, wrong `0%`. It is only on
`GET /posts/{id}/`, one request per question, and it cannot be batched: `with_my_forecasts`,
`include_my_forecasts` and `with_cp` do nothing, and `forecaster_id` is a 400.

### `forecast_type` silently drops group questions

`GET /posts/?forecast_type=binary,multiple_choice,numeric,discrete` drops any post whose
questions live under `group_of_questions`, because such a post has no top-level question to
match. A whole tournament made of group posts reads as **0 open**. Market Pulse is exactly
that: 10 posts holding 64 scored subquestions.

### An open post is not an open question

Group subquestions run on staggered windows. A post can be `open` while every one of its
subquestions is either closed or not yet open — so counting subquestions naively produces
"gaps" nobody can act on. `coverage.py` counts only what is forecastable right now and
otherwise tells you when the next window opens.

### The CDF rule that is not documented

The minimum rise between adjacent CDF points is **not** a constant. It is `0.01 / (n - 1)`,
spread across the whole curve:

* a 201-point numeric question needs `5e-05`
* a 17-point discrete question needs `0.000625` — and **rejects** what the numeric accepts

Read the grid off `scaling.continuous_range`; its length is `inbound_outcome_count + 1`.
Never assume 201. `_conform_cdf()` here satisfies every rule algebraically rather than
iterating until the checks pass, so a valid CDF is the only thing it can return, and
`validate_cdf()` re-checks the finished payload before it leaves the process.

### Multiple-choice options must match exactly

Character for character, diacritics included. `"Jakub Dolejs"` is rejected where
`"Jakub Dolejš"` is accepted. `submit_manual.py` refuses a mismatch rather than sending it.

### Pagination is a silent cap

The page size is capped at 100 however much you ask for, and the `next` link stays populated
past the last page — so "follow `next` until it is null" never terminates. Stop on an empty
batch.

Measured 2026-08-11 against a corpus of thousands, because "I asked for 500 and got 26" only
proves the corpus was small:

```
asked   99 -> got  99      # under the cap, honoured exactly
asked  100 -> got 100
asked  101 -> got 100      # the wall
asked 1000 -> got 100
```

And walking a 26-post tournament to its end: `offset=26` returns 0 results with `next` still
set, and so does `offset=226`. There is no null to wait for.

## Files

| file | what it does |
| --- | --- |
| `metaculus_bot/forecast.py` | the bot: fetch, research, ensemble, build a payload, submit |
| `metaculus_bot/coverage.py` | what is unanswered, per tournament, with prize eligibility; `--brief` prints everything needed to forecast a question by hand |
| `metaculus_bot/submit_manual.py` | place forecasts written outside the LLM call, through the same CDF construction, the same `validate_cdf`, the same ledger |

`submit_manual.py` exists because the LLM backend is the single point of failure: when it is
rate-limited or out of credit the bot abstains on everything, and a tournament question that
closes unanswered is a scored zero, not a neutral skip. It also queues: a forecast written
before its window opens is skipped with the opening time and lands on a later run, so a
window cannot open and close unattended.

**If you use it for an AI Benchmark tournament, disclose it.** The participant survey asks
about manual involvement and the code review exists to check there is no human in the loop.
Disclosure is a supported answer; a code review that does not match how the forecasts were
produced is not.

## Licence

MIT.
