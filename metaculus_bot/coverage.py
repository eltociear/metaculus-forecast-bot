#!/usr/bin/env python3
"""How much of each funded tournament have we actually forecast  - and where do we stand?

A tournament score is RELATIVE. A question we never answered is not a neutral skip: rivals
bank points on it while we bank none, so coverage is the cheapest lever we own. Nothing in
the repo measured it  - the blitz prints "19 forecasts logged" with no denominator, which
looks like progress whether the round has 20 questions or 200.

Coverage is read from Metaculus's own record (`question.my_forecasts.latest`), not from our
local ledger, because the ledger says what we *tried* to send. Only the server knows what
landed  - a CDF we rejected as malformed is logged locally and absent here.

  python3 scripts/metaculus_bot/coverage.py             # all funded tournaments
  python3 scripts/metaculus_bot/coverage.py --open      # list the unforecast ones
  python3 scripts/metaculus_bot/coverage.py --brief     # full text of the unforecast ones

`--brief` exists because the LLM backend is not always available (HuggingFace's free tier is
depleted until 2026-09-01) and forecasts still have to be placed. It prints everything needed
to reason about a question  - type, close time, options, scale, resolution criteria, fine
print  - so a human or an agent can write a `submit_manual.py` input without re-deriving which
fields matter. Pair it with that script, which keeps every validation the bot would apply.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast import (  # noqa: E402  - sibling module, path fixed just above
    API_BASE_URL,
    PRIZE_ELIGIBLE_BOT_STATUS,
    TOURNAMENTS,
    _request,
    metaculus_headers,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE = REPO_ROOT / "state" / "metaculus-coverage.json"

# Checking one question costs one detail request, measured at ~1.9s against the live API, and
# it cannot be batched: no list parameter brings `my_forecasts` back (probed with_my_forecasts,
# include_my_forecasts, with_cp, forecaster_id). A 500-question FutureEval season would
# therefore take ~16 minutes  - past the blitz's 280s subprocess timeout and past the daily
# grind's 10-minute step, so an opening season would have killed the whole daily run.
#
# So each run spends a bounded budget, and spends it where it buys the most:
#   1. questions whose status we have never confirmed, closing soonest first
#   2. then, only if budget remains, re-confirming ones already known forecast
# A submitted forecast cannot vanish from the server, so (2) is genuinely lower value; the
# cache below is what makes a big season cheap after the first pass.
#
# Whatever is left unchecked is reported as UNKNOWN, never folded into "forecast"  - a silent
# cap would turn "we ran out of time" into "we covered everything", which is the exact lie
# this file exists to stop.
MAX_DETAIL_DEFAULT = 60

# Tournaments worth measuring. The prize column is what a BOT can win, which is not the same
# as the advertised pool: the Metaculus Cup is `bot_leaderboard_status: exclude_and_show`, so
# its $5,000 is unreachable for us and it is kept only as a calibration benchmark against
# humans. Eligibility is re-read live from the API on every run  - see prize_status()  - so this
# table can never quietly go stale the way a hardcoded belief would.
# The fourth column is the NUMERIC project id, needed only by the leaderboard endpoint  - # `/leaderboards/project/minibench/` 404s where `/leaderboards/project/33074/` works, even
# though the posts endpoint takes the slug happily.
FUNDED = [("minibench", "MiniBench", "$1,000", 33074),
          ("animal", "Animal Futures", "$3,400", 33016),
          ("pulse", "Market Pulse 26Q3", "$7,500", 33066),
          ("summer", "Summer FutureEval", "$50,000", 33022),
          ("cup", "Metaculus Cup", "no bot prize", 33021)]

OUR_USER_ID = 301182   # eltociear_bot; leaderboard entries carry user.id


def _strip_ansi(s: str) -> str:
    """The standing string is coloured for the terminal; JSON should not carry escapes."""
    import re as _re
    return _re.sub(r"\033\[[0-9;]*m", "", s)


def load_env() -> None:
    """The GitHub Action injects METACULUS_TOKEN; locally it lives in .env."""
    if os.getenv("METACULUS_TOKEN"):
        return
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("METACULUS_TOKEN="):
                os.environ["METACULUS_TOKEN"] = line.split("=", 1)[1].strip().strip("\"'")


def my_forecast_state(post: dict) -> tuple[bool, datetime | None]:
    """(has a live forecast, when it was made).

    Costs one request per question, and that is not avoidable: `my_forecasts` is **absent
    from the list response entirely**  - not empty, absent  - so reading coverage off the list
    scores every question as unforecast and reports a confident 0%. It is only on
    `/posts/{id}/`. Same shape of trap as Apify's run list omitting chargedEventCounts.
    """
    pid = post.get("id")
    try:
        detail = _request(f"{API_BASE_URL}/posts/{pid}/", headers=metaculus_headers())
    except Exception:  # noqa: BLE001 - a read failure must not be scored as "forecast"
        return False, None
    subs, next_open = live_sub_questions(detail)
    if not subs:
        # Nothing forecastable in this post right now  - not a gap. Either it is an
        # announcement, or its windows are closed/not yet open. The next opening time is
        # returned so the report can say WHEN there will be work instead of staying silent.
        return True, next_open
    # A group post is only covered when EVERY live subquestion is. Treating "any subquestion
    # forecast" as done would call a 6-part Market Pulse group complete after one.
    for q in subs:
        latest = ((q.get("my_forecasts") or {}).get("latest")) or {}
        if latest.get("forecast_values") is None:
            return False, next_open
    return True, next_open


def hours_left(post: dict) -> float | None:
    ts = (post.get("question") or {}).get("scheduled_close_time") or post.get("scheduled_close_time")
    if not ts:
        return None
    try:
        close = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return (close - datetime.now(timezone.utc)).total_seconds() / 3600
    except Exception:  # noqa: BLE001
        return None


def leaderboard(tid, numeric_id=None) -> str:
    """Our actual standing, or why we have none.

    笞 This used to try three guessed paths, get 404 on each, and print "no headless
    leaderboard endpoint". Three failed guesses are not a finding. The real path is
    `/leaderboards/project/{numeric id}/`  - it works, it is authenticated, and it carries
    rank, score, coverage, contribution_count and the actual prize each entrant is on track
    for. Reporting an unknown as a conclusion hid a whole instrument for a day.

    Needs the NUMERIC project id: a slug like "minibench" 404s here even though the posts
    endpoint accepts it.
    """
    pid = numeric_id if numeric_id is not None else tid
    try:
        d = _request(f"{API_BASE_URL}/leaderboards/project/{pid}/", headers=metaculus_headers())
    except Exception as e:  # noqa: BLE001
        return f"unreadable ({getattr(e, 'code', '?')})"
    board = (d if isinstance(d, list) else [d])[0] if d else {}
    entries = board.get("entries") or []
    if not entries:
        return "board not computed yet (0 entries)"
    paid = [e for e in entries if e.get("prize")]
    floor = f", smallest prize ${min(e['prize'] for e in paid):.0f}" if paid else ""
    mine = [e for e in entries if (e.get("user") or {}).get("id") == OUR_USER_ID]
    if mine:
        e = mine[0]
        return (f"\033[32mrank {e.get('rank')} of {len(entries)}\033[0m, "
                f"score {e.get('score')}, on track for ${e.get('prize') or 0:.2f} "
                f"({len(paid)} of {len(entries)} are paid{floor})")
    return (f"\033[33mnot on the board\033[0m  - {len(entries)} entrants, "
            f"{len(paid)} paid{floor}, top coverage {max((e.get('coverage') or 0) for e in entries):g}")


def brief(post: dict) -> None:
    """Everything needed to forecast one question, and nothing else.

    Deliberately prints the fields that decide a forecast and are easy to miss: the exact
    option strings (a multiple-choice payload is rejected unless they match character for
    character, diacritics included), the scale and whether each bound is open (a numeric CDF
    is built on that grid), and the fine print (which is where the traps live  - "statements
    that the hardware is 'in production' do NOT count").
    """
    detail = _request(f"{API_BASE_URL}/posts/{post['id']}/", headers=metaculus_headers())
    q = detail.get("question") or {}
    h = hours_left(post)
    print("\n" + "=" * 96)
    print(f"POST {post['id']}  type={q.get('type')}  closes={q.get('scheduled_close_time')}"
          f"  ({h:.1f}h)" if h is not None else f"POST {post['id']}  type={q.get('type')}")
    print(f"TITLE: {detail.get('title')}")
    if q.get("options"):
        print(f"OPTIONS (exact strings): {json.dumps(q['options'], ensure_ascii=False)}")
    if q.get("type") in ("numeric", "discrete"):
        sc = q.get("scaling") or {}
        print(f"SCALE: min={sc.get('range_min')} max={sc.get('range_max')} "
              f"zero_point={sc.get('zero_point')} unit={q.get('unit')!r} "
              f"open_lower={q.get('open_lower_bound')} open_upper={q.get('open_upper_bound')}")
    print("--- resolution criteria ---")
    print((q.get("resolution_criteria") or "").strip()[:1600])
    print("--- description ---")
    print((q.get("description") or "").strip()[:2200])
    if q.get("fine_print"):
        print("--- fine print ---")
        print(q["fine_print"].strip()[:900])


def list_all_open_posts(tid) -> list[dict]:
    """Every open post in a tournament, INCLUDING group posts.

    `forecast.py:list_open_questions` filters on `forecast_type`, which silently drops posts
    whose questions live under `group_of_questions`  - they have no top-level question to
    match. Market Pulse 26Q3 is entirely group posts, so that filter reported "0 open" for a
    live $7,500 bot-eligible tournament and the bot never touched it.
    """
    out, seen, offset = [], set(), 0
    while True:
        params = {"limit": 100, "offset": offset, "order_by": "-hotness",
                  "tournaments": tid, "statuses": "open", "include_description": "true"}
        d = _request(f"{API_BASE_URL}/posts/?{urllib.parse.urlencode(params)}",
                     headers=metaculus_headers())
        batch = d.get("results") or []
        if not batch:
            break
        for p in batch:
            if p.get("id") not in seen:
                seen.add(p.get("id"))
                out.append(p)
        offset += len(batch)
        if offset >= (d.get("count") or d.get("total") or offset):
            break
    return out


_TOURNAMENT_INDEX: dict | None = None


def prize_status(tid, sample_post: dict | None = None) -> tuple[str, str]:
    """(bot_leaderboard_status, pool), read live.

    Forecasting into a tournament that excludes bots earns exactly nothing, and nothing in
    the repo checked this until 25 Cup forecasts had already been placed. It is one field.

    `/projects/tournaments/` is not enough on its own: MiniBench is `type: question_series`
    and is absent from those 193 entries, so it read as "unknown" forever  - while being
    `include` with a $1,000 pool, i.e. the very tournament we were submitting to. There is
    no endpoint that lists question series (every guess 404s), so the fallback is a post's
    own `projects` block, which carries the same fields and is authoritative.
    """
    global _TOURNAMENT_INDEX
    if _TOURNAMENT_INDEX is None:
        _TOURNAMENT_INDEX = {}
        try:
            d = _request(f"{API_BASE_URL}/projects/tournaments/", headers=metaculus_headers())
            for t in (d if isinstance(d, list) else (d.get("results") or [])):
                for k in (t.get("id"), t.get("slug")):
                    if k is not None:
                        _TOURNAMENT_INDEX[str(k)] = t
        except Exception:  # noqa: BLE001 - unknown eligibility must not stop the count
            pass
    t = _TOURNAMENT_INDEX.get(str(tid)) or {}
    if not t and sample_post:
        for entries in (sample_post.get("projects") or {}).values():
            for cand in (entries if isinstance(entries, list) else [entries]):
                if not isinstance(cand, dict):
                    continue
                if str(cand.get("id")) == str(tid) or str(cand.get("slug")) == str(tid):
                    t = cand
                    break
            if t:
                break
    return t.get("bot_leaderboard_status") or "unknown", t.get("prize_pool") or "?"


def sub_questions(post: dict) -> list[dict]:
    """The question(s) a post actually scores on.

    A Market Pulse post holds 6-8 subquestions under `group_of_questions` and has NO
    top-level question. Counting posts instead of subquestions would report "10 open" for a
    tournament that scores 64 separate forecasts.
    """
    grp = (post.get("group_of_questions") or {}).get("questions") or []
    if grp:
        return grp
    q = post.get("question")
    return [q] if q else []


def _when(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        return None


def live_sub_questions(post: dict) -> tuple[list[dict], datetime | None]:
    """(subquestions we can forecast right now, when the next one opens).

    A post being "open" does not mean any of its questions are. Market Pulse staggers a
    group across biweekly windows: on 2026-08-11 all 64 subquestions were either already
    closed or not yet open, so a coverage check that counted subquestions reported 10 gaps
    that nobody could close. An ACTIONABLE nobody can act on is how an alert channel dies.
    """
    now = datetime.now(timezone.utc)
    live, next_open = [], None
    for q in sub_questions(post):
        opens, closes = _when(q.get("open_time")), _when(q.get("scheduled_close_time"))
        if opens and opens > now:
            next_open = opens if next_open is None else min(next_open, opens)
            continue
        if closes and closes <= now:
            continue
        live.append(q)
    return live, next_open


def load_cache() -> dict:
    """post id -> True, for questions previously CONFIRMED forecast on the server."""
    try:
        prev = json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return {str(i): True for t in prev.get("tournaments", []) for i in (t.get("forecastIds") or [])}


def main() -> int:
    load_env()
    show_brief = "--brief" in sys.argv
    show_open = "--open" in sys.argv or show_brief
    budget = 10 ** 9 if "--full" in sys.argv else MAX_DETAIL_DEFAULT
    for i, a in enumerate(sys.argv):
        if a == "--max" and i + 1 < len(sys.argv):
            budget = int(sys.argv[i + 1])
    budget_started_at = budget
    cache = load_cache()
    out = {"generatedAt": datetime.now(timezone.utc).isoformat(), "tournaments": []}
    grand_gap = grand_unknown = 0

    for key, label, prize, numeric_id in FUNDED:
        tid = TOURNAMENTS.get(key, key)
        try:
            posts = list_all_open_posts(tid)
        except Exception as e:  # noqa: BLE001
            print(f"\n=== {label} ({prize}) ===\n  fetch failed: {e}")
            continue
        # Tournaments carry announcement posts ("The Market Pulse Challenge is Live!") that
        # hold no question at all. They cannot be forecast, so counting them as gaps would
        # keep an ACTIONABLE alert lit forever over something nobody can act on.
        posts = [p for p in posts if sub_questions(p)]

        # Spend the budget on what we have never confirmed, soonest-closing first.
        fresh = [p for p in posts if str(p.get("id")) not in cache]
        known = [p for p in posts if str(p.get("id")) in cache]
        fresh.sort(key=lambda p: (hours_left(p) is None, hours_left(p) or 1e9))

        done, missing, unknown, opens_at = list(known), [], [], None
        # Only posts with NOTHING left to open may be cached as settled. A Market Pulse post
        # holds staggered biweekly subquestions, so "covered" today can become "a new gap" on
        # its next window without the post ever changing. Caching those would make the tool
        # skip the detail fetch forever and report 10/10 through every future opening  - the
        # exact blindness this file exists to remove. Verified against the real data: all ten
        # Market Pulse posts had been cached that way.
        settled = list(known)
        for p in fresh:
            if budget <= 0:
                unknown.append(p)
                continue
            budget -= 1
            has, nxt = my_forecast_state(p)
            if nxt is not None:
                opens_at = nxt if opens_at is None else min(opens_at, nxt)
            (done if has else missing).append(p)
            if has and nxt is None:
                settled.append(p)

        grand_gap += len(missing)
        grand_unknown += len(unknown)
        checked = len(posts) - len(unknown)
        pct = 100 * len(done) // max(checked, 1)
        colour = "\033[32m" if pct >= 90 and not unknown else ("\033[33m" if pct >= 50 else "\033[31m")

        status, pool = prize_status(tid, posts[0] if posts else None)
        # "unknown" means the tournament list did not carry this id  - MiniBench is addressed
        # by a slug the index does not key on. Never render that as EXCLUDED: a false
        # "no prize" on a paying tournament is exactly the error that would make a later
        # session abandon real money.
        if status == "unknown":
            badge = "\033[90meligibility unknown (not in the tournament index)\033[0m"
        elif status in PRIZE_ELIGIBLE_BOT_STATUS:
            badge = "\033[32mbot-eligible\033[0m"
        else:
            badge = "\033[31mBOTS EXCLUDED  - no prize\033[0m"
        subs = sum(len(sub_questions(p)) for p in posts)
        print(f"\n=== {label} ({prize})  id={tid} ===")
        print(f"  {badge}  (bot_leaderboard_status={status}, advertised pool ${pool})")
        print(f"  open posts: {len(posts)}"
              + (f" holding {subs} scored subquestion(s)" if subs != len(posts) else "")
              + f"   forecast by us: {colour}{len(done)} "
                f"({pct}% of {checked} checked)\033[0m   \033[1mUNFORECAST: {len(missing)}\033[0m")
        if known:
            print(f"  {len(known)} of those were confirmed on an earlier run and not re-fetched")
        if opens_at and not missing:
            hrs = (opens_at - datetime.now(timezone.utc)).total_seconds() / 3600
            print(f"  \033[36mnext forecasting window opens {opens_at:%Y-%m-%d %H:%M}Z "
                  f"({hrs:.0f}h)\033[0m  - nothing is forecastable before then")
        if unknown:
            print(f"  \033[33mUNKNOWN: {len(unknown)}\033[0m  - the {budget_started_at}-question "
                  f"detail budget ran out, so these were never checked. They are NOT counted "
                  f"as covered. Re-run with --full to settle them.")
        standing = leaderboard(tid, numeric_id)
        print(f"  standing: {standing}")

        # Closing soonest first: an unforecast question that closes in 6 hours is the one
        # that is actually about to be lost.
        missing.sort(key=lambda p: (hours_left(p) is None, hours_left(p) or 1e9))
        if missing and show_open:
            for p in missing[:40]:
                h = hours_left(p)
                hh = f"{h:6.1f}h" if h is not None else "     ?"
                print(f"    {hh}  [{p.get('id')}] {(p.get('title') or '')[:78]}")
        if missing and show_brief:
            print(f"\n--- full briefs for the {len(missing)} unforecast question(s), "
                  f"closing soonest first ---")
            for p in missing:
                brief(p)
        elif missing:
            urgent = [p for p in missing if (hours_left(p) or 1e9) < 72]
            print(f"  {len(urgent)} of them close within 72h  - run with --open to list them")

        out["tournaments"].append({
            "key": key, "label": label, "prize": prize, "id": tid,
            "open": len(posts), "forecast": len(done), "unforecast": len(missing),
            "unknown": len(unknown), "checked": checked,
            "coveragePct": pct,
            # Persisted so the blitz can show it without re-fetching, and so "we appeared on
            # a board" is visible as a CHANGE rather than something you have to notice.
            "standing": _strip_ansi(standing), "botStatus": status,
            # Only ids we CONFIRMED forecast AND that have no window left to open, so the
            # next run's cache can never inherit a guess or freeze a post whose next
            # subquestion is still coming. An unknown stays unknown until something checks it.
            "forecastIds": [p.get("id") for p in settled],
            "missingIds": [p.get("id") for p in missing],
            "unknownIds": [p.get("id") for p in unknown],
            "missingClosingSoon": [
                {"id": p.get("id"), "title": p.get("title"), "hoursLeft": round(hours_left(p) or -1, 1)}
                for p in missing if (hours_left(p) or 1e9) < 72],
        })

    print("\n" + "=" * 70)
    print(f"\033[1mTOTAL UNFORECAST ACROSS FUNDED TOURNAMENTS: {grand_gap}\033[0m")
    if grand_unknown:
        print(f"\033[33m{grand_unknown} question(s) were never checked this run "
              f"(detail budget). Not counted as covered.\033[0m")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved -> {STATE.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
