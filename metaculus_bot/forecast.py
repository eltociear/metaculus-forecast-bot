#!/usr/bin/env python3
"""
Metaculus AI Forecasting Benchmark bot — our entry into the one income lane that
needs no buyer, no listing and no audience.

Why this lane (measured 2026-08-04): every marketplace we ever listed on failed on
distribution, not capability. A bot tournament inverts that — WE are the intended
participant, the prize pool is already funded, and nobody has to discover us.

  Summer 2026 FutureEval  id=33022  slug=summer-futureeval-2026  $50,000
  MiniBench               id=minibench                            $1,000, bi-weekly, always-on
  Bot Testing Area        id=bot-testing-area                     no prize, safe to test against

Proven payouts: the Q2 benchmark ran 96 bots over 300+ questions and paid $30,000,
top bot $7,550.

Design notes
  * stdlib only — runs on a bare `python:3.x` GitHub Actions runner with no install step.
  * LLM backend is pluggable: OpenRouter if OPENROUTER_API_KEY is set (Metaculus hands
    participants free credits), otherwise the Hugging Face router with HF_INFERENCE_TOKEN.
  * Every question is forecast by an ENSEMBLE of independent reasoning passes and
    aggregated by median — a single sample is noisy and the tournament scores calibration.
  * All four question types are supported. Numeric and discrete questions need a CDF that
    Metaculus validates strictly, so we ask the model for percentiles, interpolate them onto
    the exact grid the question ships in `scaling.continuous_range`, and then push the result
    through _conform_cdf(), which is constructed so a valid CDF is the only thing it can
    return. Every payload is re-checked by validate_cdf() before it leaves the process —
    a malformed CDF scores worse than abstaining, so we abstain rather than send one.

Usage
  python scripts/metaculus_bot/forecast.py --selftest        # no Metaculus token needed
  python scripts/metaculus_bot/forecast.py --dry-run         # fetch real questions, don't submit
  python scripts/metaculus_bot/forecast.py --tournament minibench --submit
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Status lines print em-dashes and arrows. A Windows console (cp932/cp1252) raises
# UnicodeEncodeError on the first such character and kills the run — including an operator's
# `--submit`. Force UTF-8 so a real terminal behaves like the pipe/CI path; no-op where already UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

API_BASE_URL = "https://www.metaculus.com/api"
REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "state" / "metaculus-forecasts.json"

TOURNAMENTS = {
    "summer": 33022,        # $50k FutureEval season, bots_only, closes 2026-11-05
    "pulse": 33066,         # $7.5k Market Pulse 26Q3, bots INCLUDED, forecasting ends 09-16
    "animal": 33016,        # $3.4k Animal Futures, bots INCLUDED, forecasting open to 2027-07
    "cup": 33021,           # $5k Metaculus Cup — bots are EXCLUDED from the prize, see below
    "minibench": "minibench",
    "test": "bot-testing-area",
}

# Every tournament object carries `bot_leaderboard_status`, and it decides whether our
# forecasts can win anything at all:
#   bots_only / include            -> we are ranked for prizes
#   exclude_and_show / _and_hide   -> we may forecast, and are ineligible for the pool
#
# Checked across all 193 tournaments on 2026-08-11. The Metaculus Cup is `exclude_and_show`:
# 25 forecasts were placed there believing $5,000 was in play, and none of it ever was. It is
# still worth forecasting as a calibration benchmark against humans — but it is not income,
# and coverage.py now labels it so nobody re-learns this the expensive way.
#
# The same sweep found Market Pulse 26Q3: $7,500, ongoing, `include`, and we had zero
# forecasts in it because its questions are all GROUPS — `list_open_questions` filters on
# `forecast_type`, which drops group posts, so the whole tournament read as "0 open".
PRIZE_ELIGIBLE_BOT_STATUS = ("bots_only", "include")

NUMERIC_SUPPORTED = True

# Metaculus validates a submitted CDF strictly. The rules, as the API states them:
MIN_CDF_RISE = 0.01          # every CDF must climb at least this much in total...
OPEN_BOUND_MIN_MASS = 0.001  # ...and an open bound carries at least this much probability


def min_cdf_step(points: int) -> float:
    """The smallest rise Metaculus accepts between adjacent CDF points.

    It is not a fixed number: the requirement is MIN_CDF_RISE spread over the whole curve, so
    a coarse question demands a much bigger step than a fine one. A 201-point numeric needs
    5e-05, while a 17-point discrete needs 0.000625 — submitting the former's step on the
    latter is rejected with "must be increasing by at least 0.000625 at every step".
    """
    return MIN_CDF_RISE / max(1, points - 1)

# Probed 2026-08-11: these three answer on the free tier, because the providers the router
# sends them to are included in it. Small models are NOT a safe fallback here — Qwen2.5-7B
# and Llama-3.1-8B both 402 with "monthly included credits depleted" on a token that serves
# all three of these fine, so falling back to something cheaper would fail closed.
HF_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "deepseek-ai/DeepSeek-V3-0324",
    "meta-llama/Llama-3.3-70B-Instruct",
]

# Measured: the free-tier 402 cleared between t+60s and t+90s. 60 gives the window room to
# pass on the first retry without stalling a run that could have carried on sooner.
HF_402_BACKOFF_SECONDS = 60
# How many 60s waits one LLM call will spend on a 402 before giving up as RateLimited. Two
# covers the ~90s window; capping it stops a sustained window from wedging a call for minutes.
HF_402_MAX_WAITS = 2
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")
# BlockRun's free NVIDIA tier: an OpenAI open-weight 120B, paid by wallet SIGNATURE only ($0
# charged), pinned to a free `nvidia/*` model so smart-routing can never pick a paid one. This
# is the backend that keeps the bot alive while HF's free tier is depleted (canPay:false until
# ~2026-09-01) — no operator email, no USDC, no HF quota. Override with BLOCKRUN_MODEL.
BLOCKRUN_MODEL = os.getenv("BLOCKRUN_MODEL", "nvidia/gpt-oss-120b")


# --------------------------------------------------------------------------- env


def load_env() -> None:
    """Populate os.environ from .env so local runs match CI runs."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        # Not setdefault: CI declares secrets as env vars that are the empty string when
        # the secret is absent, and an empty value must not mask the .env one.
        if not os.environ.get(k):
            os.environ[k] = v.strip().strip('"').strip("'")


# --------------------------------------------------------------------------- http


RETRY_STATUSES = (408, 429, 500, 502, 503, 504)


def _request(url: str, *, method: str = "GET", headers: dict | None = None,
             payload: dict | list | None = None, timeout: int = 90, attempts: int = 3):
    """One HTTP call, retrying only the failures that are safe and worth retrying.

    Reads are retried: a single transient HTTPError on get_post_details cost us a whole
    question for that run, and questions are what the tournament pays for. Writes are not,
    because a forecast POST that timed out may well have been recorded, and a blind retry
    would submit it twice. Only GET gets more than one attempt.
    """
    data = json.dumps(payload).encode() if payload is not None else None
    hdrs = {"User-Agent": "eltociear-forecast-bot/1.0", "Accept": "application/json"}
    if data is not None:
        hdrs["Content-Type"] = "application/json"
    hdrs.update(headers or {})
    if method != "GET":
        attempts = 1

    last: Exception | None = None
    for attempt in range(attempts):
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read().decode("utf-8", errors="replace")
            return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in RETRY_STATUSES:
                raise  # a 400 or 404 will say the same thing however often we ask
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = e
        if attempt < attempts - 1:
            # Cloudflare's 1015 rate limit is the one we actually hit, and it does not clear
            # in the couple of seconds that suffice for a flaky connection — backing off for
            # two seconds just spends another attempt confirming we are still limited.
            slow = isinstance(last, urllib.error.HTTPError) and last.code == 429
            time.sleep((10 if slow else 2) * (attempt + 1))
    raise last


def metaculus_headers() -> dict:
    token = os.getenv("METACULUS_TOKEN")
    if not token:
        # ⚠ This used to end "copy the token from /futureeval/participate/". The token is NOT
        # on that page — it shows a three-step panel whose own step 1 sends you to your
        # settings page. Someone followed the old wording, landed somewhere with no token,
        # and came back. Wrong instructions cost more than missing ones, and this text is
        # published, so it says where the token actually is.
        raise SystemExit(
            "METACULUS_TOKEN is not set.\n"
            "Create a bot account at https://www.metaculus.com/aib (\"Create a Bot Account\").\n"
            "You are redirected to your SETTINGS page: create the bot there and copy the\n"
            "access token it shows. Copy it immediately — it is displayed once."
        )
    return {"Authorization": f"Token {token}"}


# --------------------------------------------------------------------------- llm


class NoLLMBackend(RuntimeError):
    """No usable LLM backend exists at all (no token). Aborts the whole run — retrying
    other questions cannot help when nothing is configured."""


class RateLimited(RuntimeError):
    """The backend is temporarily rate-limited (HF's account-wide 402 window). Unlike
    NoLLMBackend this is per-moment, not per-run, so the caller skips THIS sample and moves
    on rather than aborting — the window clears in about 90s and later questions succeed."""


def call_llm(prompt: str, temperature: float = 0.4, max_tokens: int = 1400) -> str:
    """One completion from whichever backend is configured. Raises on total failure."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        body = {
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        d = _request(
            "https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            headers={"Authorization": f"Bearer {openrouter_key}"},
            payload=body,
        )
        return d["choices"][0]["message"]["content"]

    # Metaculus runs an OpenAI-compatible proxy for tournament bots, funded by OpenAI and
    # Anthropic credits. Probed 2026-08-11: it already AUTHENTICATES our METACULUS_TOKEN —
    # every model answers 400 "You don't have an allowance for model <x>", which is an
    # authorization answer, not 401. So the only thing missing is a credit grant, applied for
    # by asking Metaculus (contact is on the FutureEval resources page). Wired ahead of HuggingFace so the
    # lane starts working the moment that grant lands, with no code change.
    #
    # Note the scheme: `Token <t>`, exactly as the Metaculus API itself takes. `Bearer` gets
    # a 401 that reads like a bad token and sends you looking in the wrong place.
    proxy_model = os.getenv("METACULUS_PROXY_MODEL")
    metac_token = os.getenv("METACULUS_TOKEN")
    if proxy_model and metac_token:
        try:
            d = _request(
                "https://llm-proxy.metaculus.com/proxy/openai/v1/chat/completions",
                method="POST",
                headers={"Authorization": f"Token {metac_token}"},
                payload={"model": proxy_model,
                         "messages": [{"role": "user", "content": prompt}],
                         "temperature": temperature,
                         "max_tokens": max_tokens},
            )
            return d["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            # 400 here means "no allowance for that model" — a configuration answer, not a
            # transient one. Fall through to HuggingFace rather than abandoning the run.
            body = e.read().decode(errors="replace")[:200] if hasattr(e, "read") else ""
            print(f"    metaculus proxy unusable ({e.code}): {body}")

    # BlockRun free NVIDIA tier (blockrun_llm SDK, x402 wallet-signature auth, $0 on nvidia/*
    # models). Wired ahead of HF because HF's free tier is depleted until ~2026-09-01, so this
    # is what actually forecasts during the outage — no operator email, no USDC, no HF quota.
    # Reuses the on-chain wallet key we already hold. On any failure (SDK missing, network,
    # model EOL) it falls through to HF rather than aborting the run.
    blockrun_key = os.getenv("BLOCKRUN_WALLET_KEY") or os.getenv("BASE_WALLET_PRIVATE_KEY")
    if blockrun_key:
        try:
            from blockrun_llm import LLMClient  # noqa: E402 - optional backend, imported lazily
            text = LLMClient(private_key=blockrun_key).chat(
                BLOCKRUN_MODEL, prompt, temperature=temperature, max_tokens=max_tokens)
            if text and text.strip():
                return text
            print("    blockrun returned empty; falling through to HF")
        except ImportError:
            pass  # SDK not installed here; use HF
        except Exception as e:  # noqa: BLE001 - any backend error: fall through to HF
            print(f"    blockrun unusable ({type(e).__name__}): {str(e)[:120]}")

    # Both tokens, in preference order. This used to read `A or B`, so B was only ever a
    # default for when A was unset, never a fallback for when A stopped working. They turn
    # out to be two tokens on the SAME account (both whoami as `eltociear`), so this buys
    # nothing against rate limits — only against one token being revoked. The real fix for
    # exhaustion is the 402 backoff below.
    seen: set = set()
    tokens = [t for t in (os.getenv("HF_INFERENCE_TOKEN"), os.getenv("HF_TOKEN"))
              if t and not (t in seen or seen.add(t))]
    if not tokens:
        raise NoLLMBackend("set OPENROUTER_API_KEY or HF_INFERENCE_TOKEN")

    # A 402 is account-wide: every token and model on this account 402s together and clears
    # together (measured — the same call refused at t+0/30/60 and answered at t+90). So on a
    # 402 there is nothing to gain by trying the other five (token, model) combos — they will
    # all refuse too. Doing so anyway, each with an escalating 60/120s backoff, spent ~18 min
    # per LLM call and wedged whole tournament runs. Instead a 402 spends a SHARED, bounded
    # wait budget retrying the SAME call, and once it is gone gives up as RateLimited so the
    # caller can skip this question rather than burn the run. Non-402 errors still fall
    # through to the next model, which is what iterating the combos is actually for.
    last = None
    hf_402_waits_left = HF_402_MAX_WAITS
    for index, hf_token in enumerate(tokens):
        for model in HF_MODELS:
            body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            for attempt in range(3):
                try:
                    d = _request(
                        "https://router.huggingface.co/v1/chat/completions",
                        method="POST",
                        headers={"Authorization": f"Bearer {hf_token}"},
                        payload=body,
                    )
                    return d["choices"][0]["message"]["content"]
                except urllib.error.HTTPError as e:
                    last = f"token#{index + 1} {model} HTTP {e.code}"
                    if e.code in (429, 503):  # server overload — back off and retry
                        time.sleep(3 * (attempt + 1))
                        continue
                    if e.code == 402:
                        if hf_402_waits_left <= 0:
                            raise RateLimited(
                                "HF 402 rate window did not clear within "
                                f"{HF_402_MAX_WAITS * HF_402_BACKOFF_SECONDS}s. Set "
                                "OPENROUTER_API_KEY to bypass HF.") from e
                        hf_402_waits_left -= 1
                        time.sleep(HF_402_BACKOFF_SECONDS)
                        continue  # retry the SAME combo; the others would 402 identically
                    break  # a 400 will not change on a retry; try the next model
                except Exception as e:  # noqa: BLE001 - network flake, try next model
                    last = f"token#{index + 1} {model} {type(e).__name__}"
                    break
    raise NoLLMBackend(
        f"all {len(tokens)} HF token(s) x {len(HF_MODELS)} models failed (last: {last}). "
        "Set OPENROUTER_API_KEY to bypass HF entirely.")


# --------------------------------------------------------------------------- research


def fetch_news(query: str, limit: int = 10) -> str:
    """Recent headlines for a question, via the Google News RSS feed.

    Deliberately keyless: the paid research APIs the template assumes (AskNews, Perplexity,
    Exa) all need credentials we don't have, and an LLM forecasting current events off its
    training data alone is badly handicapped. Fails soft — no research beats no forecast.
    """
    import html as _html

    url = ("https://news.google.com/rss/search?q="
           + urllib.parse.quote(query)
           + "&hl=en-US&gl=US&ceid=US:en")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - research is best-effort
        return ""

    items = re.findall(r"<item>(.*?)</item>", body, re.S)[:limit]
    lines = []
    for item in items:
        title = re.search(r"<title>(.*?)</title>", item, re.S)
        pub = re.search(r"<pubDate>(.*?)</pubDate>", item, re.S)
        if not title:
            continue
        clean = _html.unescape(re.sub(r"<[^>]+>", "", title.group(1))).strip()
        date = (pub.group(1)[:16] if pub else "")
        lines.append(f"- [{date}] {clean}")
    return "\n".join(lines)


def build_research(question: dict, enabled: bool = True) -> str:
    if not enabled:
        return ""
    title = question.get("title", "")
    if not title:
        return ""
    headlines = fetch_news(title)
    if not headlines:
        return ""
    return (f"Recent headlines matching this question (retrieved "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}; headlines only, so treat "
            f"them as leads rather than confirmed fact):\n{headlines}")


# --------------------------------------------------------------------------- prompts

CALIBRATION_PREAMBLE = """You are a superforecaster competing in the Metaculus AI Forecasting
Benchmark. You are scored on calibration against real resolutions, so an honest 65% beats a
confident 95% that is wrong.

Today's date: {today}

Work through these steps before answering:
1. Reference class / base rate. How often do events of this general type happen in a window
   of this length? State the base rate explicitly.
2. The status quo outcome. If nothing changes between now and the resolution date, how does
   this resolve? The world usually does NOT change — weight the status quo heavily.
3. The time remaining. Short windows favour the status quo even more strongly.
4. The strongest case for YES, and the strongest case for NO.
5. Reconcile: start from the base rate, adjust only as far as the specific evidence justifies.

Avoid these failure modes: over-reacting to a vivid recent headline; treating a plausible
story as if it were likely; and clustering every answer near 50%."""

BINARY_TEMPLATE = """{preamble}

QUESTION
{title}

RESOLUTION CRITERIA
{resolution_criteria}

FINE PRINT
{fine_print}

BACKGROUND
{background}

{research}
Closes: {close_time}    Resolves: {resolve_time}

Give your reasoning, then end with the final line in exactly this format:
Probability: ZZ%"""

MULTIPLE_CHOICE_TEMPLATE = """{preamble}

QUESTION
{title}

OPTIONS
{options}

RESOLUTION CRITERIA
{resolution_criteria}

FINE PRINT
{fine_print}

BACKGROUND
{background}

{research}
Closes: {close_time}    Resolves: {resolve_time}

Give your reasoning, then end with one line per option in exactly this format, probabilities
summing to 100:
Option_A: XX%
Option_B: YY%
(using the exact option text as the label)"""

NUMERIC_TEMPLATE = """{preamble}

QUESTION
{title}

RESOLUTION CRITERIA
{resolution_criteria}

FINE PRINT
{fine_print}

BACKGROUND
{background}

{research}
Closes: {close_time}    Resolves: {resolve_time}

UNITS AND SCALE
Give every number in these units: {unit}
{bounds}

A numeric answer is scored on the whole distribution, not on your best guess, so:
6. Anchor on the most recent actual value you know, and say what it is.
7. Ask how much this quantity has moved over a window as long as the one remaining.
8. Keep the tails wide. Bots lose far more points to a surprise outside a narrow interval
   than they gain from a tight one that was right.

Give your reasoning, then end with exactly these lines — bare numbers in the units above,
with no commas, no currency symbols, no unit words, and increasing down the list:
Percentile 5: XX
Percentile 10: XX
Percentile 20: XX
Percentile 40: XX
Percentile 60: XX
Percentile 80: XX
Percentile 90: XX
Percentile 95: XX"""


def _bounds_text(question: dict) -> str:
    """Describe the scale to the model, including whether outcomes may fall outside it."""
    scaling = question.get("scaling") or {}
    low = scaling.get("nominal_min", scaling.get("range_min"))
    high = scaling.get("nominal_max", scaling.get("range_max"))
    open_lo = bool(question.get("open_lower_bound", scaling.get("open_lower_bound")))
    open_hi = bool(question.get("open_upper_bound", scaling.get("open_upper_bound")))
    lines = [f"The scale runs from {low} to {high}."]
    lines.append(
        f"Outcomes BELOW {low} are possible — say so by putting low percentiles under it."
        if open_lo else f"The outcome cannot be below {low}.")
    lines.append(
        f"Outcomes ABOVE {high} are possible — say so by putting high percentiles over it."
        if open_hi else f"The outcome cannot be above {high}.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- parsing


def parse_binary_probability(text: str) -> float | None:
    """Pull the final 'Probability: NN%' out of a completion, clamped to [1, 99]."""
    matches = re.findall(r"[Pp]robability\s*:?\s*([0-9]*\.?[0-9]+)\s*%", text)
    if not matches:
        matches = re.findall(r"([0-9]*\.?[0-9]+)\s*%", text)
    if not matches:
        return None
    try:
        value = float(matches[-1])
    except ValueError:
        return None
    if not 0 <= value <= 100:
        return None
    return min(99.0, max(1.0, value)) / 100.0


def parse_option_probabilities(text: str, options: list[str]) -> dict[str, float] | None:
    """Map each option label to a probability, normalised to sum to 1."""
    found: dict[str, float] = {}
    for option in options:
        pattern = re.escape(option) + r"\s*:?\s*([0-9]*\.?[0-9]+)\s*%"
        hits = re.findall(pattern, text, flags=re.IGNORECASE)
        if hits:
            try:
                found[option] = float(hits[-1])
            except ValueError:
                pass
    if len(found) < len(options):
        return None
    total = sum(found.values())
    if total <= 0:
        return None
    # Normalise, then floor every option so no option is ever exactly 0.
    normalised = {k: max(0.01, v / total) for k, v in found.items()}
    total = sum(normalised.values())
    return {k: v / total for k, v in normalised.items()}


# Models like to dress the final block up — bold it, bullet it, put a currency sign or a
# percent on the value. `_JUNK` is the decoration we step over between the pieces we want.
_JUNK = r"[\s_*`~|]*"
PERCENTILE_RE = re.compile(
    # The separator deliberately excludes a bare "-": on "Percentile 5 -30" it would be eaten
    # as punctuation and hand back +30, silently inverting a forecast on any question whose
    # scale crosses zero. Dropping such a line is far cheaper than flipping its sign.
    r"[Pp]ercentile" + _JUNK + r"([0-9]+(?:\.[0-9]+)?)" + _JUNK + r"[:=–—]?" + _JUNK
    + r"[$€£]?" + _JUNK + r"(-?[0-9][0-9,]*(?:\.[0-9]+)?)")

# The exact ladder NUMERIC_TEMPLATE asks for. Anything else in the reply is not an answer to
# the question we asked: one observed run relabelled the ladder as deciles (1, 2, 4, 6, 8, 9),
# which parses cleanly and means something entirely different. Off-grid samples are dropped.
REQUESTED_PERCENTILES = (5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 90.0, 95.0)
MIN_PERCENTILES = 5


def _sorted_values(pairs: dict[float, float]) -> dict[float, float]:
    """Re-sort values against ascending percentiles.

    A percentile ladder that descends somewhere is a presentation slip rather than a different
    belief, so we put the values back in order instead of discarding the sample.
    """
    keys = sorted(pairs)
    return dict(zip(keys, sorted(pairs[k] for k in keys)))


def parse_percentiles(text: str) -> dict[float, float] | None:
    """Pull 'Percentile NN: value' lines out of a completion."""
    found: dict[float, float] = {}
    for pct, value in PERCENTILE_RE.findall(text):
        try:
            p, v = float(pct), float(value.replace(",", ""))
        except ValueError:
            continue
        if p in REQUESTED_PERCENTILES:
            found[p] = v  # a later line wins, so the final summary block beats the reasoning
    if len(found) < MIN_PERCENTILES:
        return None
    return _sorted_values(found)


def _question_grid(question: dict) -> list[float] | None:
    """The exact x-values Metaculus wants a CDF evaluated at."""
    scaling = question.get("scaling") or {}
    grid = scaling.get("continuous_range")
    if isinstance(grid, list) and len(grid) >= 2:
        return [float(x) for x in grid]
    # Older payloads omit continuous_range; rebuild it from the range and the bucket count.
    low, high = scaling.get("range_min"), scaling.get("range_max")
    count = question.get("inbound_outcome_count") or scaling.get("inbound_outcome_count") or 200
    if low is None or high is None or high <= low:
        return None
    step = (float(high) - float(low)) / int(count)
    return [float(low) + step * i for i in range(int(count) + 1)]


def _interpolate(x: float, points: list[tuple[float, float]]) -> float:
    """Piecewise-linear P(X <= x), extending the end segments past the outermost points."""
    if x <= points[0][0]:
        (x0, y0), (x1, y1) = points[0], points[1]
        return y0 + (y1 - y0) / (x1 - x0) * (x - x0)
    if x >= points[-1][0]:
        (x0, y0), (x1, y1) = points[-2], points[-1]
        return y1 + (y1 - y0) / (x1 - x0) * (x - x1)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def _conform_cdf(raw: list[float], open_lo: bool, open_hi: bool) -> list[float] | None:
    """Force a rough curve into a CDF Metaculus will accept, by construction.

    Every rule is satisfied algebraically rather than by iterating until the checks pass:
    we reserve the mandatory minimum rise for each of the n-1 gaps up front, then spend what
    is left on the shape. With `t` non-decreasing in [0, 1], `step` the per-gap minimum and
    slack >= 0,

        out[i] = lo + i * step + t[i] * slack

    is >= lo, is <= hi, and rises by at least `step` every step — all three at once.
    """
    n = len(raw)
    step = min_cdf_step(n)
    lo = OPEN_BOUND_MIN_MASS if open_lo else 0.0
    hi = 1.0 - OPEN_BOUND_MIN_MASS if open_hi else 1.0
    slack = (hi - lo) - (n - 1) * step
    if slack < 0:
        return None  # too many buckets to fit the required slope; abstaining is correct

    t = []
    running = 0.0
    for value in raw:
        running = max(running, min(1.0, max(0.0, value)))  # clamp, then make non-decreasing
        t.append(running)
    # A closed bound is an exact equality, not a range: pin it before spending the slack.
    if not open_lo:
        t[0] = 0.0
    if not open_hi:
        t[-1] = 1.0

    return [lo + i * step + t[i] * slack for i in range(n)]


def validate_cdf(cdf, question: dict) -> str | None:
    """Re-check a finished payload. Returns a reason string, or None when it is valid."""
    grid = _question_grid(question)
    if grid is None:
        return "question has no usable scale"
    if not isinstance(cdf, list) or len(cdf) != len(grid):
        return f"expected {len(grid)} points, got {len(cdf) if isinstance(cdf, list) else type(cdf).__name__}"
    if any(not isinstance(v, (int, float)) or v != v for v in cdf):
        return "contains a non-number"
    if cdf[0] < -1e-12 or cdf[-1] > 1 + 1e-12:
        return f"out of [0, 1]: starts {cdf[0]}, ends {cdf[-1]}"
    required = min_cdf_step(len(cdf))
    tightest = min(b - a for a, b in zip(cdf, cdf[1:]))
    if tightest < required - 1e-12:
        return f"smallest step {tightest:.2e} is below the required {required:.2e}"
    scaling = question.get("scaling") or {}
    if not bool(question.get("open_lower_bound", scaling.get("open_lower_bound"))) and abs(cdf[0]) > 1e-12:
        return f"closed lower bound needs cdf[0] == 0, got {cdf[0]}"
    if not bool(question.get("open_upper_bound", scaling.get("open_upper_bound"))) and abs(cdf[-1] - 1) > 1e-12:
        return f"closed upper bound needs cdf[-1] == 1, got {cdf[-1]}"
    return None


def build_cdf(percentiles: dict[float, float], question: dict) -> list[float] | None:
    """Turn {percentile: value} into a CDF sampled on the question's own grid."""
    grid = _question_grid(question)
    if grid is None or not percentiles:
        return None

    # Interpolation needs strictly increasing x. Equal percentile values mean the model put a
    # spike there; nudge them apart by a hair rather than dropping the point.
    span = max(abs(grid[-1] - grid[0]), 1e-9)
    points: list[tuple[float, float]] = []
    for pct in sorted(percentiles):
        x, y = float(percentiles[pct]), pct / 100.0
        if points and x <= points[-1][0]:
            x = points[-1][0] + span * 1e-6
        points.append((x, y))
    if len(points) < 2:
        return None

    scaling = question.get("scaling") or {}
    cdf = _conform_cdf(
        [_interpolate(x, points) for x in grid],
        bool(question.get("open_lower_bound", scaling.get("open_lower_bound"))),
        bool(question.get("open_upper_bound", scaling.get("open_upper_bound"))),
    )
    if cdf is None:
        return None
    return None if validate_cdf(cdf, question) else cdf


# --------------------------------------------------------------------------- forecasting


def build_prompt(question: dict, research: str = "") -> tuple[str, str]:
    """Return (prompt, question_type) for a Metaculus question object."""
    qtype = question.get("type", "binary")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    common = {
        "preamble": CALIBRATION_PREAMBLE.format(today=today),
        "title": question.get("title", ""),
        "resolution_criteria": question.get("resolution_criteria") or "(none given)",
        "fine_print": question.get("fine_print") or "(none)",
        "background": question.get("description") or "(none)",
        "research": (f"RECENT RESEARCH\n{research}\n" if research else ""),
        "close_time": question.get("scheduled_close_time", "?"),
        "resolve_time": question.get("scheduled_resolve_time", "?"),
    }
    if qtype == "multiple_choice":
        options = question.get("options") or []
        return MULTIPLE_CHOICE_TEMPLATE.format(options="\n".join(f"- {o}" for o in options), **common), qtype
    if qtype in ("numeric", "discrete"):
        return NUMERIC_TEMPLATE.format(
            unit=question.get("unit") or "(the unit named in the question)",
            bounds=_bounds_text(question), **common), qtype
    return BINARY_TEMPLATE.format(**common), qtype


def salvage_percentiles(prompt: str, truncated: str) -> dict | None:
    """Recover the percentile ladder when a run reasoned past its token budget.

    A reasoning model handed a numeric question spends the whole budget doing arithmetic
    longhand — least-squares fits, residuals, year-over-year tables — and gets cut off
    mid-sentence before it ever reaches the summary block. `parse_percentiles` then
    returns None, the run is discarded as "unparseable", and with all 3 runs failing the
    same way the question is forfeited. Measured on Summer FutureEval question 45199,
    which closed unforecast in a $50,000 tournament for exactly this reason.

    The analysis in that output is usually fine; only the last eight lines are missing.
    So hand the model its own work back and ask for nothing but the ladder. This is the
    real fix — raising the budget alone just moves the cliff.
    """
    ask = ("You were part-way through a forecast when your output was cut off.\n\n"
           f"--- the question and required format ---\n{prompt[-1100:]}\n\n"
           f"--- your analysis so far ---\n{truncated[-3500:]}\n\n"
           "Finish now. Output ONLY the eight lines, bare numbers, no commas, no units, "
           "strictly increasing, nothing else:\n"
           "Percentile 5:\nPercentile 10:\nPercentile 20:\nPercentile 40:\n"
           "Percentile 60:\nPercentile 80:\nPercentile 90:\nPercentile 95:")
    # 2000, not the few hundred tokens eight short lines actually occupy. "Output ONLY the
    # eight lines" does not stop a reasoning model reasoning — it re-derives the whole fit
    # first and only then prints them. Measured on this exact prompt: at 400 and at 1200
    # tokens the reply is still mid-arithmetic and parses to None; at 2000 it returns the
    # bare ladder and nothing else. Budgeting for the visible answer rather than for the
    # thinking in front of it is the same mistake that lost the question in the first place.
    try:
        return parse_percentiles(call_llm(ask, temperature=0.2, max_tokens=2000))
    except Exception:  # noqa: BLE001 - salvage is best-effort; the caller already failed
        return None


def forecast_question(question: dict, runs: int, research: str = "", verbose: bool = True):
    """Run an ensemble and aggregate. Returns (forecast, question_type, rationales)."""
    prompt, qtype = build_prompt(question, research)
    options = question.get("options") or []

    # Some questions are not opinions. MiniBench's are auto-generated from public data sources,
    # so for a handful of families the number the question resolves against is simply
    # fetchable — the NWS forecast for a Central Park temperature, the NWS precipitation
    # probability, a spot price plus its measured volatility. Asking a language model to guess
    # those instead is how we produced a +/-6% interval for a one-day-ahead stock close.
    #
    # quant_forecast matches narrowly and returns None for everything else, so this can only
    # ever replace a guess with a measurement, never the other way round.
    try:
        from quant import quant_forecast  # noqa: PLC0415 - optional, never fatal
        hit = quant_forecast(question)
    except Exception as e:  # noqa: BLE001 - a data-source outage must fall back to the LLM
        hit = None
        if verbose:
            print(f"    quant unavailable ({type(e).__name__}); using the model")
    if hit:
        kind, value, why = hit
        if kind == "probability" and qtype == "binary":
            if verbose:
                print(f"    \033[32mquant\033[0m {value:.1%}  ({why})")
            return float(value), qtype, [f"quantitative source: {why}"]
        if kind == "percentiles" and qtype in ("numeric", "discrete"):
            cdf = build_cdf(value, question)
            if cdf is not None:
                if verbose:
                    print(f"    \033[32mquant\033[0m " +
                          "  ".join(f"p{int(p)}={v:g}" for p, v in sorted(value.items())))
                    print(f"      ({why})")
                return ({"cdf": cdf, "percentiles": value}, qtype,
                        [f"quantitative source: {why}"])
            if verbose:
                # the scale rejected it — say so rather than silently falling through, or a
                # broken bound would look like the source simply not matching
                print("    quant produced percentiles the question scale rejected; using the model")
    samples: list = []
    rationales: list[str] = []
    # Numeric answers carry eight numbers behind a wall of arithmetic, so they need far
    # more room than a binary "Probability: NN%". The shared 1400 default was enough for
    # binary and silently short for numeric.
    budget = 3000 if qtype in ("numeric", "discrete") else 1400

    for i in range(runs):
        try:
            text = call_llm(prompt, temperature=0.3 + 0.1 * i, max_tokens=budget)
        except NoLLMBackend as e:
            raise
        except RateLimited:
            # The rate window is account-wide, so the remaining runs of THIS question would
            # each spend another ~120s hitting it. Stop and use whatever samples we have —
            # 3 runs x 120s x every question would blow the CI timeout, and the next cron
            # cycle re-attempts this question once the window has cleared anyway.
            if verbose:
                print(f"    run {i + 1}/{runs}: rate-limited, stopping this question")
            break
        except Exception as e:  # noqa: BLE001 - one bad sample shouldn't kill the ensemble
            if verbose:
                print(f"    run {i + 1}/{runs}: error {type(e).__name__}")
            continue

        if qtype == "binary":
            parsed = parse_binary_probability(text)
        elif qtype == "multiple_choice":
            parsed = parse_option_probabilities(text, options)
        elif qtype in ("numeric", "discrete"):
            parsed = parse_percentiles(text)
            if parsed is None and text.strip():
                parsed = salvage_percentiles(prompt, text)
                if parsed is not None and verbose:
                    print(f"    run {i + 1}/{runs}: ladder salvaged from truncated reasoning")
        else:
            parsed = None

        if parsed is None:
            if verbose:
                print(f"    run {i + 1}/{runs}: unparseable")
            continue
        samples.append(parsed)
        rationales.append(text)
        if verbose:
            if qtype == "binary":
                shown = f"{parsed:.1%}"
            elif qtype in ("numeric", "discrete"):
                shown = "  ".join(f"p{int(p)}={parsed[p]:g}" for p in sorted(parsed))
            else:
                shown = json.dumps({k: round(v, 3) for k, v in parsed.items()})
            print(f"    run {i + 1}/{runs}: {shown}")

    if not samples:
        return None, qtype, rationales

    if qtype == "binary":
        return statistics.median(samples), qtype, rationales

    if qtype in ("numeric", "discrete"):
        # Aggregate percentile-by-percentile rather than by averaging finished CDFs. Averaging
        # curves from runs that disagree flattens the distribution into something nobody
        # believed; taking the median value at each percentile keeps the ensemble's sharpness.
        # Aggregate over each percentile the runs agree to answer, rather than the strict
        # intersection: one run that skips a rung must not cost us the whole question.
        quorum = max(1, len(samples) // 2)
        merged = _sorted_values({
            p: statistics.median(votes)
            for p in REQUESTED_PERCENTILES
            if len(votes := [s[p] for s in samples if p in s]) >= quorum
        })
        if len(merged) < 3:
            return None, qtype, rationales
        cdf = build_cdf(merged, question)
        if cdf is None:
            if verbose:
                print("    no valid CDF could be built from those percentiles")
            return None, qtype, rationales
        if verbose:
            print("    ensemble: " + "  ".join(f"p{int(p)}={merged[p]:g}" for p in sorted(merged)))
        return {"cdf": cdf, "percentiles": merged}, qtype, rationales

    # multiple choice: average each option across runs, then renormalise
    aggregated = {o: statistics.mean([s[o] for s in samples]) for o in options}
    total = sum(aggregated.values())
    return {o: v / total for o, v in aggregated.items()}, qtype, rationales


def create_forecast_payload(forecast, question_type: str) -> dict:
    if question_type == "binary":
        return {"probability_yes": forecast, "probability_yes_per_category": None, "continuous_cdf": None}
    if question_type == "multiple_choice":
        return {"probability_yes": None, "probability_yes_per_category": forecast, "continuous_cdf": None}
    cdf = forecast["cdf"] if isinstance(forecast, dict) and "cdf" in forecast else forecast
    return {"probability_yes": None, "probability_yes_per_category": None, "continuous_cdf": cdf}


# --------------------------------------------------------------------------- metaculus io


PAGE_SIZE = 100  # the server's hard cap: ask for 200 or 500 and it still returns 100


def list_open_questions(tournament, limit: int = 500) -> list[dict]:
    """Every open question in a tournament, following the API's pagination.

    This used to issue a single 50-row request and drop the `next` link on the floor. A
    MiniBench round is ~60 questions and a FutureEval season is 300-500, so with
    order_by=-hotness we would forecast the same hottest 50 and never see the rest — no
    error, no warning, just a season's worth of questions we never answered. Raising the
    number alone would not have fixed it either, since the server silently caps a page at
    100 however much you ask for.
    """
    params = {
        "limit": min(PAGE_SIZE, limit),
        "offset": 0,
        "order_by": "-hotness",
        "forecast_type": "binary,multiple_choice,numeric,discrete",
        "tournaments": tournament,
        "statuses": "open",
        "include_description": "true",
    }
    url = f"{API_BASE_URL}/posts/?{urllib.parse.urlencode(params)}"
    posts: list[dict] = []
    seen: set = set()
    while url and len(posts) < limit:
        data = _request(url, headers=metaculus_headers())
        batch = data.get("results") or []
        if not batch:
            # The API hands back a `next` link even for the page past the end, so an empty
            # batch — not a null link — is what actually means "that was all of them".
            url = None
            break
        for post in batch:
            # -hotness can reshuffle between requests, so the same post can appear on two
            # pages. Dedupe by id rather than trusting the paging to be stable.
            if post.get("id") not in seen:
                seen.add(post.get("id"))
                posts.append(post)
        url = data.get("next")
    if url and len(posts) >= limit:
        # Never truncate coverage silently — that is the bug this function just had.
        print(f"    NOTE: stopped at --limit {limit} with more pages still available")
    return posts[:limit]


def get_post_details(post_id: int) -> dict:
    return _request(f"{API_BASE_URL}/posts/{post_id}/", headers=metaculus_headers())


def already_forecast(post_details: dict) -> bool:
    try:
        return post_details["question"]["my_forecasts"]["latest"]["forecast_values"] is not None
    except Exception:  # noqa: BLE001 - absence of the key just means "not yet"
        return False


def submit_forecast(question_id: int, payload: dict) -> None:
    _request(
        f"{API_BASE_URL}/questions/forecast/",
        method="POST",
        headers=metaculus_headers(),
        payload=[{"question": question_id, "source": "api", **payload}],
    )


def post_comment(post_id: int, text: str) -> None:
    _request(
        f"{API_BASE_URL}/comments/create/",
        method="POST",
        headers=metaculus_headers(),
        payload={"text": text, "parent": None, "included_forecast": True,
                 "is_private": True, "on_post": post_id},
    )


# --------------------------------------------------------------------------- ledger


def load_ledger() -> dict:
    if LEDGER_PATH.exists():
        try:
            return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"forecasts": {}}


def save_ledger(ledger: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(json.dumps(ledger, indent=1, sort_keys=True), encoding="utf-8")


def ledger_has_submission(ledger: dict, question_id) -> bool:
    """Did we already SUBMIT this question, according to our own committed ledger?

    Only `submitted: true` counts — dry runs are recorded too, and treating one as done
    would skip a question we never actually answered.
    """
    entry = (ledger.get("forecasts") or {}).get(str(question_id))
    return bool(entry and entry.get("submitted"))


# --------------------------------------------------------------------------- modes


SELFTEST_QUESTION = {
    "id": 0,
    "type": "binary",
    "title": "Will the price of Bitcoin be above $1,000,000 on 2026-12-31?",
    "resolution_criteria": "Resolves YES if the CoinGecko BTC/USD close on 2026-12-31 exceeds $1,000,000.",
    "fine_print": "",
    "description": "Bitcoin has never exceeded $200,000.",
    "scheduled_close_time": "2026-12-31T00:00:00Z",
    "scheduled_resolve_time": "2026-12-31T00:00:00Z",
}

SELFTEST_MC_QUESTION = {
    "id": 0,
    "type": "multiple_choice",
    "title": "Which of these will be the warmest month of 2026 in the northern hemisphere?",
    "options": ["June", "July", "August"],
    "resolution_criteria": "Resolves to the month with the highest average NH land temperature.",
    "fine_print": "",
    "description": "",
    "scheduled_close_time": "2026-12-31T00:00:00Z",
    "scheduled_resolve_time": "2027-01-31T00:00:00Z",
}


def run_selftest(runs: int, research_enabled: bool = True) -> int:
    """End-to-end check of everything except the token-gated calls.

    We cannot test our own tournament submissions without burning a real forecast, so this
    proves the LLM backend, the prompt, the parser, the ensemble and the payload shape.
    """
    print("=== SELFTEST (no METACULUS_TOKEN required) ===\n")
    backend = "OpenRouter" if os.getenv("OPENROUTER_API_KEY") else (
        "HuggingFace" if (os.getenv("HF_INFERENCE_TOKEN") or os.getenv("HF_TOKEN")) else "NONE")
    print(f"LLM backend: {backend}")
    if backend == "NONE":
        # Do not bail. Three of the five checks are entirely offline — the parsers, the news
        # fetch, and CDF construction, which is the hardest part of this bot to get right.
        # A fresh clone with no API key still deserves to be able to verify them; returning 1
        # here made the test useless in exactly the situation where it is most wanted.
        print("  no LLM backend configured — the two model-dependent checks will be SKIPPED, "
              "the offline ones still run")
    failures = 0

    print("\n[1/5] parser unit checks")
    cases = [
        ("blah blah\nProbability: 37%", 0.37),
        ("Probability: 0.5%", 0.01),      # clamped up
        ("Probability: 99.9%", 0.99),     # clamped down
        ("no number here", None),
    ]
    for text, expected in cases:
        got = parse_binary_probability(text)
        ok = (got is None and expected is None) or (
            got is not None and expected is not None and abs(got - expected) < 1e-9)
        print(f"   {'ok  ' if ok else 'FAIL'} {text[:28]!r} -> {got}")
        failures += 0 if ok else 1

    mc = parse_option_probabilities("June: 20%\nJuly: 50%\nAugust: 30%", ["June", "July", "August"])
    ok = mc is not None and abs(sum(mc.values()) - 1.0) < 1e-6 and abs(mc["July"] - 0.5) < 0.01
    print(f"   {'ok  ' if ok else 'FAIL'} multiple-choice parse -> {mc}")
    failures += 0 if ok else 1

    print("\n[2/5] news research (keyless Google News RSS)")
    research = build_research(SELFTEST_QUESTION, enabled=research_enabled)
    if research_enabled and not research:
        print("   WARN: no headlines returned — the bot will still forecast, unresearched")
    else:
        n = max(0, len(research.splitlines()) - 1)
        print(f"   ok   {n} headlines")
        for line in research.splitlines()[1:4]:
            print(f"        {line[:100]}")

    print("\n[3/5] binary forecast (a question whose answer should be very low)")
    # A missing or dead backend must not abort the selftest, because the checks AFTER this
    # one are the offline ones — CDF construction is the hardest part of this bot to get
    # right and the part a fresh clone most needs to be able to verify. Losing them to an
    # unset API key means the one situation where you most want the test is the one where
    # it tells you nothing.
    try:
        forecast, qtype, _ = forecast_question(SELFTEST_QUESTION, runs=runs, research=research)
    except NoLLMBackend as e:
        print(f"   SKIPPED: {e}")
        print("   (the offline checks below still run and still mean something)")
        forecast, qtype = None, "binary"
        skipped_llm = True
    else:
        skipped_llm = False
    if forecast is None:
        if not skipped_llm:
            print("   FAIL: no usable samples")
            failures += 1
    else:
        print(f"   ensemble median: {forecast:.1%}")
        payload = create_forecast_payload(forecast, qtype)
        print(f"   payload: {json.dumps(payload)}")
        if forecast > 0.25:
            print("   WARN: implausibly high for this question — check the model/prompt")
        else:
            print("   ok   plausible")

    print("\n[4/5] CDF construction (offline — this is what a bad forecast gets rejected for)")
    grids = {"201-point numeric": [i * 0.5 for i in range(201)],
             "17-point discrete": [float(i) - 0.5 for i in range(17)]}
    shapes = {
        "ordinary":        {5: 20, 10: 25, 20: 35, 40: 45, 60: 55, 80: 70, 90: 80, 95: 88},
        "entirely below":  {5: -90, 10: -80, 20: -70, 40: -60, 60: -50, 80: -40, 90: -30, 95: -20},
        "entirely above":  {5: 300, 10: 310, 20: 320, 40: 330, 60: 340, 80: 350, 90: 360, 95: 370},
        "a single spike":  {5: 50, 10: 50, 20: 50, 40: 50, 60: 50, 80: 50, 90: 50, 95: 50},
    }
    cdf_failures = 0
    for grid_name, grid in grids.items():
        for open_lo in (True, False):
            for open_hi in (True, False):
                question = {"type": "numeric", "open_lower_bound": open_lo,
                            "open_upper_bound": open_hi,
                            "scaling": {"continuous_range": grid, "range_min": grid[0],
                                        "range_max": grid[-1]}}
                for shape_name, percentiles in shapes.items():
                    cdf = build_cdf(percentiles, question)
                    problem = "build_cdf returned None" if cdf is None else validate_cdf(cdf, question)
                    if problem:
                        print(f"   FAIL {grid_name} open=({open_lo},{open_hi}) {shape_name}: {problem}")
                        cdf_failures += 1
    print(f"   {'ok  ' if not cdf_failures else 'FAIL'} {len(grids) * 4 * len(shapes)} "
          f"combinations, {cdf_failures} invalid")
    failures += cdf_failures

    off_grid = parse_percentiles("Percentile 1: 0\nPercentile 2: 0\nPercentile 4: 0\n"
                                 "Percentile 6: 0\nPercentile 8: 0\nPercentile 9: 5")
    print(f"   {'ok  ' if off_grid is None else 'FAIL'} a relabelled percentile ladder is rejected")
    failures += 0 if off_grid is None else 1

    dressed = parse_percentiles("**Percentile 5:** $1,200\n- Percentile 10: $1,300\n"
                                "| Percentile 20 | 1400 |\nPercentile 40 = 1500\n"
                                "Percentile 60: 1600")
    ok = dressed is not None and len(dressed) == 5 and dressed[5.0] == 1200
    print(f"   {'ok  ' if ok else 'FAIL'} bold/bulleted/tabulated/currency percentiles parse -> "
          f"{len(dressed) if dressed else 0} points")
    failures += 0 if ok else 1

    print("\n[5/5] multiple-choice forecast")
    try:
        mc_forecast, mc_type, _ = forecast_question(SELFTEST_MC_QUESTION, runs=max(1, runs - 1))
    except NoLLMBackend as e:
        print(f"   SKIPPED: {e}")
        mc_forecast, mc_type = None, "multiple_choice"
        skipped_llm = True
    if mc_forecast is None:
        if not skipped_llm:
            print("   FAIL: no usable samples")
            failures += 1
    else:
        print(f"   ensemble: {json.dumps({k: round(v, 3) for k, v in mc_forecast.items()})}")
        print(f"   sums to {sum(mc_forecast.values()):.4f}")
        print(f"   payload: {json.dumps(create_forecast_payload(mc_forecast, mc_type))}")

    print(f"\n=== SELFTEST {'PASS' if failures == 0 else f'FAIL ({failures})'} ===")
    # A selftest is what you run to find out what is broken, so it must not report two
    # anonymous "no usable samples" when the cause is known and external. Every LLM-dependent
    # step fails together when the backend is out of credit; the offline steps (parsers, CDF
    # construction) are unaffected and their result above still stands.
    if failures:
        dead = backend_depleted()
        if dead:
            print(f"\n  \033[33mThe failures above are the BACKEND, not this code\033[0m: "
                  f"{dead}.\n"
                  "  Steps 1, 2 and 4 are offline and their results are still valid.\n"
                  "  Unblock: a Metaculus proxy credit grant, or set "
                  "OPENROUTER_API_KEY / METACULUS_PROXY_MODEL.")
    return 1 if failures else 0


def _group_posts_dropped(tournament) -> int:
    """How many open posts this tournament has that `list_open_questions` cannot see.

    Same request without `forecast_type`, counting only posts that carry subquestions
    instead of a top-level question. Returns 0 on any failure — a diagnostic must never
    stop a run that could otherwise forecast.
    """
    try:
        params = {"limit": 100, "offset": 0, "tournaments": tournament, "statuses": "open"}
        d = _request(f"{API_BASE_URL}/posts/?{urllib.parse.urlencode(params)}",
                     headers=metaculus_headers())
        return sum(1 for p in (d.get("results") or [])
                   if (p.get("group_of_questions") or {}).get("questions"))
    except Exception:  # noqa: BLE001
        return 0


def backend_depleted() -> str | None:
    """Is the LLM backend out of credit for the whole billing period? Returns a reason.

    A 402 was documented here as a ~90 second rate window. On 2026-08-11 that stopped being
    true: `whoami-v2` reports `canPay: false` with `periodEnd` 2026-09-01, and all three
    router models 402 with "You have depleted your monthly included credits". That is a
    three-week wall, not a window — and the tournament runs on a */30 cron, so without this
    check the job re-discovers it 48 times a day, forecasts nothing, and bills a rounded-up
    Actions minute each time out of a 2,000 minute budget.

    Only asks HuggingFace: with OPENROUTER_API_KEY set the HF path is never reached.
    """
    if os.getenv("OPENROUTER_API_KEY"):
        return None
    token = os.getenv("HF_INFERENCE_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        return None  # no backend at all is NoLLMBackend's job to report, not this one
    try:
        d = _request("https://huggingface.co/api/whoami-v2",
                     headers={"Authorization": f"Bearer {token}"})
    except Exception:  # noqa: BLE001 - a failed probe must never block a run that could work
        return None
    if d.get("canPay") is False:
        end = d.get("periodEnd")
        when = (datetime.fromtimestamp(end, tz=timezone.utc).strftime("%Y-%m-%d")
                if isinstance(end, (int, float)) else "the end of the period")
        # canPay=false alone is normal for a free account with credit left, so confirm with
        # one real call before declaring the lane dead.
        try:
            call_llm("Say OK", max_tokens=8)
            return None
        except RateLimited:
            return (f"HuggingFace monthly included credits are depleted and the account "
                    f"cannot pay; they reset {when}")
        except Exception:  # noqa: BLE001
            return None
    return None


def run_tournament(tournament, runs: int, submit: bool, limit: int, comment: bool,
                   research_enabled: bool = True) -> int:
    ledger = load_ledger()
    print(f"=== tournament={tournament} submit={submit} runs={runs} ===")

    dead = backend_depleted()
    if dead:
        print(f"\n\033[31mBACKEND UNAVAILABLE\033[0m: {dead}.\n"
              "  Every question would abstain, so this run stops here instead of burning an\n"
              "  Actions minute per cron tick to rediscover it.\n"
              "  Unblock, cheapest first:\n"
              "    1. Metaculus runs an LLM proxy at llm-proxy.metaculus.com that ALREADY\n"
              "       accepts our METACULUS_TOKEN (probed: it answers 400 'no allowance for\n"
              "       model', not 401). Credits are granted on request by Metaculus\n"
              "       with a bot description — an email, not a browser gate.\n"
              "    2. OPENROUTER_API_KEY, which takes priority over HuggingFace here.\n"
              "  Meanwhile forecasts can still be placed by hand through the same validation:\n"
              "    scripts/metaculus_bot/coverage.py --open   # what is unanswered\n"
              "    scripts/metaculus_bot/submit_manual.py     # send them\n")
        return 2

    posts = list_open_questions(tournament, limit=limit)
    print(f"{len(posts)} open posts\n")

    # `list_open_questions` filters on `forecast_type`, which drops posts whose questions live
    # under `group_of_questions` — they have no top-level question to match. Market Pulse is
    # entirely group posts, so this loop sees ZERO for a live $7,500 tournament and exits
    # clean having done nothing. Never let that be silent: say what was dropped and where the
    # coverage actually comes from.
    #
    # Group forecasting is deliberately NOT implemented here yet. The LLM backend is depleted
    # until 2026-09-01, so new code on this path could not be executed even once before being
    # trusted with money; the queue path below is tested and is what holds Market Pulse.
    dropped = _group_posts_dropped(tournament)
    if dropped:
        print(f"    \033[33mNOTE: {dropped} open post(s) in this tournament are GROUP posts "
              f"and are not visible to this loop\033[0m")
        print("    They are covered by scripts/metaculus_bot/coverage.py (which reads posts "
              "without the forecast_type filter) feeding state/metaculus-queued.json via "
              "submit_manual.py, which the workflow submits every run.")

    counters = {"forecast": 0, "skipped_done": 0, "skipped_numeric": 0, "failed": 0}

    for post in posts:
        question = post.get("question")
        if not question or question.get("status") != "open":
            continue
        post_id, question_id = post["id"], question["id"]
        qtype = question.get("type", "binary")
        title = question.get("title", "")
        print(f"[{question_id}] {title[:88]}  ({qtype})")

        if qtype not in ("binary", "multiple_choice") and not NUMERIC_SUPPORTED:
            print("    skip: numeric/discrete not supported yet (a malformed CDF is worse "
                  "than abstaining)")
            counters["skipped_numeric"] += 1
            continue

        # Fast path off our own committed ledger. Proving "already forecast" through the API
        # costs a get_post_details call AND the 1s pacing sleep below, for every already-done
        # question, on every run. Measured: that was ~50s of the 140s an average run took,
        # and the run does it 18 times a day to re-learn something it wrote down itself.
        #
        # Safe because the ledger only records actual submissions, and because it is not the
        # only check: coverage.py re-derives coverage from the API without consulting it, so
        # drift shows up in the blitz as an UNFORECAST count rather than hiding here.
        if ledger_has_submission(ledger, question_id):
            print("    skip: already forecast (ledger)")
            counters["skipped_done"] += 1
            continue

        # Pace ourselves, but only in front of an actual API call. A season is 300-500
        # questions and each costs several API calls; going flat out earns a Cloudflare 429
        # and loses questions outright, which is far more expensive than a second per
        # question.
        if counters["forecast"] or counters["skipped_done"]:
            time.sleep(1)

        try:
            details = get_post_details(post_id)
        except urllib.error.HTTPError as e:  # noqa: PERF203 - keep going through the queue
            # The bare type name here read "HTTPError" and hid the reason for two runs. It
            # was a Cloudflare 429, which is a pacing problem we can fix, not a dead post.
            print(f"    skip: could not read post (HTTP {e.code})")
            counters["failed"] += 1
            continue
        except Exception as e:  # noqa: BLE001 - keep going through the queue
            print(f"    skip: could not read post ({type(e).__name__}: {e})")
            counters["failed"] += 1
            continue

        if already_forecast(details):
            print("    skip: already forecast")
            counters["skipped_done"] += 1
            continue

        full_question = details.get("question") or question
        full_question.setdefault("type", qtype)
        research = build_research(full_question, enabled=research_enabled)
        if research:
            print(f"    research: {len(research.splitlines()) - 1} headlines")
        forecast, qtype, rationales = forecast_question(full_question, runs=runs, research=research)
        if forecast is None:
            print("    FAILED: no usable samples")
            counters["failed"] += 1
            continue

        payload = create_forecast_payload(forecast, qtype)
        if payload["continuous_cdf"] is not None:
            # Last line of defence: never put a malformed CDF on the wire, whatever built it.
            problem = validate_cdf(payload["continuous_cdf"], full_question)
            if problem:
                print(f"    ABSTAINED: CDF rejected by our own check ({problem})")
                counters["failed"] += 1
                continue
        if submit:
            try:
                submit_forecast(question_id, payload)
                print("    submitted")
                if comment and rationales:
                    try:
                        post_comment(post_id, rationales[-1][:20000])
                    except Exception as e:  # noqa: BLE001 - a failed comment is cosmetic
                        print(f"    (comment failed: {type(e).__name__})")
            except Exception as e:  # noqa: BLE001 - report and continue
                print(f"    SUBMIT FAILED: {e}")
                counters["failed"] += 1
                continue
        else:
            print(f"    dry-run payload: {json.dumps(payload)[:160]}")

        counters["forecast"] += 1
        ledger["forecasts"][str(question_id)] = {
            "title": title,
            "type": qtype,
            # A 201-point CDF would bury the ledger; the percentiles it was built from say
            # the same thing in a form a human can read.
            "forecast": forecast["percentiles"] if isinstance(forecast, dict)
            and "percentiles" in forecast else forecast,
            "tournament": str(tournament),
            "submitted": bool(submit),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        save_ledger(ledger)

    print(f"\n=== {json.dumps(counters)} ===")
    # A question we could not forecast is a question we scored nothing on, so it must not
    # leave the run green. Reads already retry, so what reaches here has failed repeatedly.
    if counters["failed"]:
        print(f"{counters['failed']} question(s) could not be forecast")
    return 1 if counters["failed"] else 0


def main() -> int:
    # The Windows console defaults to cp932 here, which cannot encode the dashes and
    # arrows in our own log lines.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    load_env()
    parser = argparse.ArgumentParser(description="Metaculus forecasting bot")
    parser.add_argument("--tournament", default="minibench",
                        help="summer | minibench | test, or a raw tournament id/slug")
    parser.add_argument("--runs", type=int, default=3, help="ensemble size per question")
    # 500 covers a full FutureEval season; the old default of 50 silently capped us at
    # roughly a sixth of one, and at less than a MiniBench round.
    parser.add_argument("--limit", type=int, default=500,
                        help="max posts to pull across all pages")
    parser.add_argument("--submit", action="store_true", help="actually submit forecasts")
    parser.add_argument("--dry-run", action="store_true", help="fetch and forecast, never submit")
    parser.add_argument("--comment", action="store_true", help="post the rationale as a private comment")
    parser.add_argument("--selftest", action="store_true", help="run without a Metaculus token")
    parser.add_argument("--no-research", action="store_true", help="skip the news lookup")
    args = parser.parse_args()

    if args.selftest:
        return run_selftest(runs=max(2, args.runs), research_enabled=not args.no_research)

    tournament = TOURNAMENTS.get(args.tournament, args.tournament)
    submit = args.submit and not args.dry_run
    if not submit:
        print("(dry run — pass --submit to send forecasts)\n")
    return run_tournament(tournament, runs=args.runs, submit=submit,
                          limit=args.limit, comment=args.comment,
                          research_enabled=not args.no_research)


if __name__ == "__main__":
    sys.exit(main())
