#!/usr/bin/env python3
"""Submit forecasts produced OUTSIDE the bot's own LLM call, through the bot's own checks.

⚠ PRIZE-INTEGRITY WARNING, read before using this on an AI Benchmark tournament.
Metaculus' participant survey (a Google Form, and mandatory to receive ANY prize money)
states: "By reviewing code we seek to do another level of verification to make sure there is
no human-in-the-loop." Prize winners must provide their code for review — they invite the
GitHub user `CodexVeritas` and share a commit snapshot of how the bot ran that season.

A forecast placed through this file is NOT the bot's output. That is fine for tournaments
where we are one entrant among humans (Market Pulse, Animal Futures) and irrelevant where
bots cannot win at all (the Metaculus Cup). It is a live question for the AIB series —
MiniBench and FutureEval — whose whole purpose is measuring autonomous bot capability.

So: use it there only as a stopgap while the bot has no backend, keep every entry's `source`
field (this script writes "submit_manual"), and DISCLOSE it on the survey. The survey has a
checkbox for manual involvement, so disclosure is a supported answer, not an admission —
what is not supported is a code review that fails to match how the forecasts were produced.


Why this exists: the forecasting pipeline is sound, but it is hostage to one external LLM
backend. When the Hugging Face router is inside a rate window, `forecast.py` abstains on
every question — and a tournament question that closes unanswered is not a neutral skip, it
is a scored zero while rivals bank points. On 2026-08-11 that cost us both open MiniBench
questions with under an hour left on them.

So the reasoning step becomes pluggable to a human or an agent, while everything that makes
a forecast *safe* stays exactly where it was: the same CDF construction, the same
`validate_cdf` last line of defence, the same payload builder, the same ledger.

Group posts (Market Pulse) hold 6-8 subquestions and have NO top-level question, so an entry
may name the subquestion with `"question_id"`. Without it the post's own question is used.
A subquestion whose window has not opened yet is SKIPPED with a message rather than failing,
so a prepared file can be written early and simply run again once the window opens.

Input is a JSON file mapping post id -> forecast, in the shape the question type needs:

  {
    "45344": {"type": "binary", "p": 0.12, "why": "one line of rationale"},
    "44790": {"type": "multiple_choice", "options": {"A": 0.5, "B": 0.5}},
    "43939": {"type": "numeric", "percentiles": {"5": 120, "10": 130, "20": 145,
                                                 "40": 160, "60": 175, "80": 190,
                                                 "90": 205, "95": 220}}
  }

  python3 scripts/metaculus_bot/submit_manual.py forecasts.json --dry-run
  python3 scripts/metaculus_bot/submit_manual.py forecasts.json --submit
  python3 scripts/metaculus_bot/submit_manual.py forecasts.json --submit --force
  python3 scripts/metaculus_bot/submit_manual.py forecasts.json --validate

`--validate` builds every payload against the question's CURRENT scale and runs `validate_cdf`
on it WITHOUT submitting and without waiting for the window. A queued forecast is written
against a grid read weeks earlier; if Metaculus adjusts a range or an option list before the
window opens, the ordinary path finds out only at submit time and drops the question silently.

A question we have already forecast is skipped, so the file is safe to run on a schedule —
the cron runs it every three hours and a standing forecast does not need re-POSTing. Pass
`--force` after editing a value to send the new one.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# This file prints em-dashes and ANSI-labelled status lines. On a Windows console the default
# stdout encoding is cp932/cp1252, which raises UnicodeEncodeError on the first such line and
# crashes the whole validation mid-run (CI is UTF-8 and never hit this, so it hid on Linux).
# Force UTF-8 so local `--validate`/dry runs on this box are reliable; a no-op where already UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast import (  # noqa: E402  - sibling module, path fixed just above
    build_cdf,
    create_forecast_payload,
    get_post_details,
    load_ledger,
    save_ledger,
    submit_forecast,
    validate_cdf,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env() -> None:
    if os.getenv("METACULUS_TOKEN"):
        return
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("METACULUS_TOKEN="):
                os.environ["METACULUS_TOKEN"] = line.split("=", 1)[1].strip().strip("\"'")


def build(entry: dict, question: dict) -> tuple[object | None, str, str | None]:
    """(forecast, question_type, error). Mirrors what forecast_question() would have returned."""
    qtype = entry.get("type") or question.get("type")
    if qtype == "binary":
        p = entry.get("p")
        if not isinstance(p, (int, float)) or not 0 < p < 1:
            return None, qtype, f"binary p must be strictly between 0 and 1, got {p!r}"
        # Metaculus clamps at the extremes anyway; refusing them here keeps the ledger honest.
        return min(max(float(p), 0.001), 0.999), qtype, None

    if qtype == "multiple_choice":
        opts = entry.get("options") or {}
        expected = question.get("options") or []
        if expected and set(opts) != set(expected):
            return None, qtype, f"options must be exactly {expected}, got {sorted(opts)}"
        total = sum(opts.values())
        if total <= 0:
            return None, qtype, "option probabilities sum to zero"
        return {k: v / total for k, v in opts.items()}, qtype, None

    # numeric / discrete: percentiles -> CDF on the question's own grid
    pcts = entry.get("percentiles") or {}
    if len(pcts) < 3:
        return None, qtype, "need at least 3 percentiles"
    cdf = build_cdf({float(k): float(v) for k, v in pcts.items()}, question)
    if cdf is None:
        return None, qtype, "no valid CDF could be built from those percentiles"
    return {"cdf": cdf}, qtype, None


def main() -> int:
    load_env()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    submit = "--submit" in sys.argv
    force = "--force" in sys.argv
    validate_only = "--validate" in sys.argv
    path = Path(args[0])
    plan = json.loads(path.read_text(encoding="utf-8"))

    ledger = load_ledger()
    ok = failed = skipped = 0
    now = datetime.now(timezone.utc)
    for key, entry in plan.items():
        if key.startswith("_"):         # "_note" and friends are documentation, not work
            continue
        # One post holds several subquestions on staggered windows, and JSON keys must be
        # unique — so a key may be "<post id>" or "<post id>#<anything>", the suffix existing
        # only to keep sibling windows of the same post apart. Without this a group post could
        # carry exactly one queued forecast, which is one window out of six.
        post_id = key.split("#", 1)[0]
        details = get_post_details(int(post_id))
        title = (details.get("title") or "")[:70]
        want = entry.get("question_id")
        if want:
            subs = ((details.get("group_of_questions") or {}).get("questions")
                    or ([details["question"]] if details.get("question") else []))
            question = next((q for q in subs if q.get("id") == want), None)
            if question is None:
                print(f"\n[{post_id}] {title}\n    REJECTED: no subquestion {want} on this post")
                failed += 1
                continue
            title = f"{title} [{question.get('label')}]"
        else:
            question = details.get("question") or {}
        qid = question.get("id")
        print(f"\n[{post_id}] {title}")

        # A window that has not opened yet is not an error — it is "come back later".
        def _at(value):
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except Exception:  # noqa: BLE001
                return None
        opens, closes = _at(question.get("open_time")), _at(question.get("scheduled_close_time"))
        # --validate builds and checks the payload against the question's CURRENT scale
        # without submitting, and it runs BEFORE the window test on purpose. A forecast
        # written weeks ahead is checked against a grid read weeks ago; if Metaculus adjusts
        # the range or the option list before the window opens, the ordinary path would
        # discover that only at submit time, silently dropping a question from a position
        # that is sitting on the pay threshold.
        if validate_only:
            forecast, qtype, err = build(entry, question)
            if err:
                print(f"    \033[31mWOULD FAIL\033[0m: {err}")
                failed += 1
                continue
            payload = create_forecast_payload(forecast, qtype)
            if payload["continuous_cdf"] is not None:
                problem = validate_cdf(payload["continuous_cdf"], question)
                if problem:
                    print(f"    \033[31mWOULD FAIL\033[0m: CDF rejected ({problem})")
                    failed += 1
                    continue
                sc = question.get("scaling") or {}
                print(f"    \033[32mvalid\033[0m  {len(payload['continuous_cdf'])}-point cdf on "
                      f"[{sc.get('range_min')}, {sc.get('range_max')}]")
            else:
                shown = payload["probability_yes"] or payload["probability_yes_per_category"]
                print(f"    \033[32mvalid\033[0m  {qtype}: "
                      f"{json.dumps(shown) if not isinstance(shown, float) else f'{shown:.3f}'}")
            ok += 1
            continue
        if opens and opens > now:
            print(f"    \033[36mnot open yet\033[0m — opens {opens:%Y-%m-%d %H:%M}Z "
                  f"({(opens - now).total_seconds() / 3600:.1f}h). Re-run this same file then.")
            skipped += 1
            continue
        # Already ours? Then this file has done its job for this question. The cron runs the
        # queue every three hours, and without this check a forecast whose window is open
        # would be re-POSTed with identical values on every tick — around 56 times before a
        # week-long Market Pulse window closes. It adds nothing (the standing forecast is
        # what scores) and it spends rate limit we have already been 429'd on today.
        # --force re-sends anyway, which is what you want after editing a value in the file.
        if not force:
            latest = ((question.get("my_forecasts") or {}).get("latest")) or {}
            if latest.get("forecast_values") is not None:
                when = latest.get("start_time")
                stamp = ""
                if isinstance(when, (int, float)):
                    stamp = f" at {datetime.fromtimestamp(when, tz=timezone.utc):%Y-%m-%d %H:%M}Z"
                print(f"    \033[90malready forecast{stamp} — not re-sending "
                      f"(use --force to override)\033[0m")
                skipped += 1
                continue
        if closes and closes <= now:
            print(f"    \033[33mclosed\033[0m {closes:%Y-%m-%d %H:%M}Z — nothing to send")
            skipped += 1
            continue

        forecast, qtype, err = build(entry, question)
        if err:
            print(f"    REJECTED: {err}")
            failed += 1
            continue

        payload = create_forecast_payload(forecast, qtype)
        if payload["continuous_cdf"] is not None:
            problem = validate_cdf(payload["continuous_cdf"], question)
            if problem:
                print(f"    ABSTAINED: CDF rejected by our own check ({problem})")
                failed += 1
                continue
            print(f"    cdf ok ({len(payload['continuous_cdf'])} points)")
        else:
            shown = payload["probability_yes"] or payload["probability_yes_per_category"]
            print(f"    {qtype}: {json.dumps(shown) if not isinstance(shown, float) else f'{shown:.3f}'}")

        if not submit:
            print("    dry-run — not submitted")
            continue
        try:
            submit_forecast(qid, payload)
            print("    \033[32msubmitted\033[0m")
            ok += 1
            # Key the ledger by the SUBQUESTION where there is one, otherwise a group post's
            # six windows would each overwrite the last and the ledger would show one.
            ledger.setdefault("forecasts", {})[str(qid or post_id)] = {
                "forecast": forecast if not isinstance(forecast, dict) else "cdf",
                "submitted": True,
                "title": details.get("title"),
                "tournament": entry.get("tournament", "manual"),
                "type": qtype,
                "source": "submit_manual",   # so the ledger never implies the LLM produced it
                "why": entry.get("why"),
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:  # noqa: BLE001
            print(f"    SUBMIT FAILED: {e}")
            failed += 1

    if submit and not validate_only:
        save_ledger(ledger)
    # Say what actually happened. In --validate mode nothing was sent, and reporting
    # "submitted 29" for a run that submitted nothing is the same class of false label this
    # repo spent a day removing from its own instruments.
    verb = "validated" if validate_only else "submitted"
    print(f"\n=== {verb} {ok}, failed {failed}, skipped {skipped} ===")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
