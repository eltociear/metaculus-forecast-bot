#!/usr/bin/env python3
"""Answer MiniBench questions from the data source that RESOLVES them, not from an LLM.

MiniBench is a back-to-back series of two-week $1,000 tournaments, ~60 questions each, and
Metaculus states the questions are "automatically created and resolved from public data
sources (e.g. FRED, Google Trends, Metaculus, Stocks)". Ours bear that out — one round asked
for the Central Park high temperature, whether Central Park records measurable precipitation,
whether an NWS heat alert is active for Manhattan, the closing price of Bitcoin, total crypto
market cap, and whether a named Atlantic cyclone is active. All on a 1-7 day horizon.

Every one of those has a free, keyless, authoritative API — and it is the SAME source the
question resolves against. Handing them to a language model with ten news headlines is the
worst available method, and our own output showed it: for a one-day-ahead Lockheed Martin
close the bot produced p5=550 / p95=620, centred near 585.

⚠ The first version of this note called that "+/-6% on an instrument with ~1%/day volatility,
four times too wide". That was an assumption, and measuring killed it: LMT's realised 60-day
volatility is **2.10%/day**, so a correct one-day 90% band is about +/-3.4% and the model's
width was under twice too wide, not four times. What the measurement DID expose is worse and
was invisible while the width story was in the way — LMT last closed at **604.79**, so the
model's centre sat ~3.3% BELOW spot and its p95 barely reached the current price. On a
one-day horizon the centre is the whole forecast; a distribution in the wrong place cannot be
rescued by having the right width. That is why every price family here anchors on spot.

What this module does NOT do is pretend to know things it cannot. It returns None for anything
outside the families it can source, and the LLM path handles those unchanged.

    python scripts/metaculus_bot/quant.py          # show every source live
"""
import json
import math
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

UA = {"User-Agent": "eltociear-forecast-bot (github.com/eltociear)"}
CENTRAL_PARK = (40.7789, -73.9692)          # the Belvedere Castle station the questions use
REQUESTED_PERCENTILES = (5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 90.0, 95.0)

# Standard normal quantiles for the percentiles Metaculus asks us for.
_Z = {5.0: -1.6449, 10.0: -1.2816, 20.0: -0.8416, 40.0: -0.2533,
      60.0: 0.2533, 80.0: 0.8416, 90.0: 1.2816, 95.0: 1.6449}


def _get(url, timeout=30):
    return json.loads(urllib.request.urlopen(
        urllib.request.Request(url, headers=UA), timeout=timeout).read().decode("utf-8", "replace"))


# --------------------------------------------------------------------------- distributions


def normal_percentiles(center: float, sigma: float, lo=None, hi=None) -> dict:
    """{percentile: value} for a normal. Sigma is the honest input: everything else is scoring.

    Clamped to the question's bounds when given, because a percentile outside the scale is
    rejected by build_cdf and would cost the whole question rather than a few points.
    """
    out = {}
    for p in REQUESTED_PERCENTILES:
        v = center + _Z[p] * sigma
        if lo is not None:
            v = max(v, lo)
        if hi is not None:
            v = min(v, hi)
        out[p] = v
    return out


def lognormal_percentiles(spot: float, sigma_frac: float) -> dict:
    """Percentiles for a price after a random walk of fractional volatility `sigma_frac`.

    A price is the textbook case where the median forecast IS the spot price: over one day the
    drift is negligible beside the noise, so anchoring anywhere else is a bet, not a forecast.
    The LLM path anchored Bitcoin's median about 3% above spot and picked round numbers at
    1,000-unit increments, which is what guessing looks like.
    """
    return {p: spot * math.exp(_Z[p] * sigma_frac) for p in REQUESTED_PERCENTILES}


# --------------------------------------------------------------------------- weather (NWS)


def nws_central_park() -> dict:
    """Official NWS forecast for Central Park: per-date max temp (F) and precip probability.

    api.weather.gov is free, keyless, and is the source these questions resolve against.
    """
    pt = _get(f"https://api.weather.gov/points/{CENTRAL_PARK[0]},{CENTRAL_PARK[1]}")["properties"]
    grid = _get(pt["forecastGridData"])["properties"]
    out = {}

    # maxTemperature is degC on a per-day validTime; the questions are in F.
    for v in (grid.get("maxTemperature") or {}).get("values", []):
        day = v["validTime"].split("T")[0]
        out.setdefault(day, {})["max_temp_f"] = v["value"] * 9 / 5 + 32

    # probabilityOfPrecipitation is sub-daily; a day "records measurable precipitation" if it
    # happens in ANY period, so take the day's MAXIMUM, not its mean. Averaging would halve a
    # real afternoon thunderstorm risk into a coin flip.
    for v in (grid.get("probabilityOfPrecipitation") or {}).get("values", []):
        day = v["validTime"].split("T")[0]
        cur = out.setdefault(day, {}).get("precip_prob_pct")
        val = v["value"]
        if val is not None:
            out[day]["precip_prob_pct"] = val if cur is None else max(cur, val)
    return out


def nws_active_alerts(area="NY", contains=None) -> list:
    d = _get(f"https://api.weather.gov/alerts/active?area={area}")
    feats = d.get("features") or []
    if contains:
        feats = [f for f in feats
                 if contains.lower() in json.dumps(f.get("properties") or {}).lower()]
    return [(f.get("properties") or {}).get("event") for f in feats]


def temp_sigma_f(lead_days: float) -> float:
    """Spread for an NWS max-temp forecast, in F, by lead time.

    NWS day-1 max-temp error runs about 2-3F and grows with lead. These are deliberately on
    the generous side: on a continuous question an over-tight distribution that misses is
    punished far harder than a slightly wide one that contains the answer.
    """
    return min(2.0 + 1.1 * max(0.0, lead_days), 9.0)


# --------------------------------------------------------------------------- markets


def crypto_spot() -> dict:
    p = _get("https://api.coingecko.com/api/v3/simple/price"
             "?ids=bitcoin,ethereum&vs_currencies=usd")
    g = _get("https://api.coingecko.com/api/v3/global")["data"]
    return {"btc_usd": p["bitcoin"]["usd"], "eth_usd": p["ethereum"]["usd"],
            "total_mcap_usd": g["total_market_cap"]["usd"]}


def realised_vol(series: list) -> float:
    """Daily volatility from a close series — measured, not assumed."""
    rets = [math.log(b / a) for a, b in zip(series, series[1:]) if a > 0 and b > 0]
    if len(rets) < 5:
        return 0.0
    mu = sum(rets) / len(rets)
    return math.sqrt(sum((r - mu) ** 2 for r in rets) / (len(rets) - 1))


def btc_history(days=60) -> list:
    d = _get(f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
             f"?vs_currency=usd&days={days}&interval=daily")
    return [p[1] for p in d.get("prices", [])]


def equity_closes(symbol: str, rng="3mo") -> list:
    """Daily closes for a listed symbol, keyless.

    Stooq — the obvious choice — serves a JavaScript wall to a plain client, so its CSV never
    arrives. Yahoo's chart endpoint answers without a key and carries the meta we need
    (currency, exchange) to notice if a ticker resolves somewhere unexpected.
    """
    d = _get(f"https://query1.finance.yahoo.com/v8/finance/chart/"
             f"{urllib.parse.quote(symbol)}?range={rng}&interval=1d")
    res = (d.get("chart") or {}).get("result") or []
    if not res:
        return []
    q = (res[0].get("indicators") or {}).get("quote") or [{}]
    return [c for c in (q[0].get("close") or []) if c]


# --------------------------------------------------------------------------- hurricanes


def nhc_active(basin="atlantic") -> list:
    """Named storms currently active. Atlantic ids start 'al', E-Pacific 'ep'."""
    d = _get("https://www.nhc.noaa.gov/CurrentStorms.json")
    pre = {"atlantic": "al", "epacific": "ep"}.get(basin, "al")
    return [s for s in (d.get("activeStorms") or [])
            if str(s.get("id", "")).lower().startswith(pre)]


# --------------------------------------------------------------------------- router


DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2}),?\s+(\d{4})\b", re.I)
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}


def target_date(text: str):
    m = DATE_RE.search(text or "")
    if not m:
        return None
    try:
        return datetime(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2))).date()
    except ValueError:
        return None


def quant_forecast(question: dict):
    """(kind, value, why) for a question we can source, else None.

    Matching is deliberately narrow. A wrong match silently replaces a mediocre forecast with
    a confident wrong one, which is worse than leaving it to the LLM — so every branch needs
    BOTH its subject keywords AND a target date it can resolve against, and anything that does
    not match exactly falls through untouched.

    Only families where the fetched data ANSWERS the question are here. Forward-looking
    questions that need a model on top of the data (will a heat alert be ACTIVE in six days,
    will a named cyclone EXIST in six days) are deliberately excluded: today's alert list says
    nothing about next week, and pretending otherwise is the error this module exists to stop.
    """
    title = (question.get("title") or "")
    low = title.lower()
    qtype = question.get("type")
    day = target_date(title)
    if not day:
        return None
    lead = (day - datetime.now(timezone.utc).date()).days
    if lead < 0 or lead > 7:
        return None                      # outside the NWS/vol horizon we can honestly cover

    key = day.isoformat()

    # --- Central Park high temperature -------------------------------------------------
    if qtype in ("numeric", "discrete") and "central park" in low and (
            "high temperature" in low or "maximum temperature" in low):
        cp = nws_central_park()
        rec = cp.get(key) or {}
        t = rec.get("max_temp_f")
        if t is None:
            return None
        sigma = temp_sigma_f(lead)
        return ("percentiles", normal_percentiles(t, sigma),
                f"NWS gridpoint maxTemperature for {key} = {t:.1f}F, sigma {sigma:.1f}F at "
                f"lead {lead}d")

    # --- Central Park measurable precipitation -----------------------------------------
    if qtype == "binary" and "central park" in low and "precipitation" in low:
        cp = nws_central_park()
        pop = (cp.get(key) or {}).get("precip_prob_pct")
        if pop is None:
            return None
        # NWS probability of precipitation IS the probability of >=0.01in at a point, which is
        # exactly what "measurable precipitation" means. It is a calibrated number from the
        # office that resolves the question; it needs no adjustment.
        p = min(0.97, max(0.03, pop / 100.0))
        return ("probability", p, f"NWS probabilityOfPrecipitation for {key} = {pop}%")

    # --- Bitcoin close ------------------------------------------------------------------
    if qtype in ("numeric", "discrete") and "bitcoin" in low and (
            "clos" in low or "price" in low):
        spot = crypto_spot()["btc_usd"]
        vol = realised_vol(btc_history(60))
        if not vol:
            return None
        sigma = vol * math.sqrt(max(1, lead))
        return ("percentiles", lognormal_percentiles(spot, sigma),
                f"BTC spot ${spot:,} anchored, realised 60d vol {vol*100:.2f}%/day, "
                f"sqrt({max(1,lead)}d) -> sigma {sigma*100:.2f}%")

    # --- listed equity close --------------------------------------------------------------
    # Company -> ticker, spelled out rather than guessed. A ticker inferred from prose is how
    # you end up confidently forecasting the wrong instrument, which is worse than abstaining;
    # anything not on this list falls through to the model.
    EQUITIES = {
        "lockheed martin": "LMT", "boeing": "BA", "nvidia": "NVDA", "apple": "AAPL",
        "microsoft": "MSFT", "tesla": "TSLA", "amazon": "AMZN", "alphabet": "GOOGL",
        "meta platforms": "META", "palantir": "PLTR", "rtx": "RTX",
        "northrop grumman": "NOC", "general dynamics": "GD",
    }
    if qtype in ("numeric", "discrete") and ("stock price" in low or "share price" in low
                                             or "close at" in low or "closing price" in low):
        for company, sym in EQUITIES.items():
            if company not in low:
                continue
            closes = equity_closes(sym)
            vol = realised_vol(closes)
            if not closes or not vol:
                return None
            spot = closes[-1]
            # Trading days, not calendar days: a Friday question resolving Monday is ONE
            # session of risk, not three. Overstating the horizon inflates the spread by
            # sqrt(3) and hands away points on exactly the weekend questions.
            sessions = max(1, sum(1 for i in range(1, lead + 1)
                                  if (datetime.now(timezone.utc).date()
                                      + timedelta(days=i)).weekday() < 5))
            sigma = vol * math.sqrt(sessions)
            return ("percentiles", lognormal_percentiles(spot, sigma),
                    f"{sym} last close ${spot:,.2f} anchored, realised 60d vol "
                    f"{vol*100:.2f}%/day over {sessions} trading session(s) -> "
                    f"sigma {sigma*100:.2f}%")

    return None


# --------------------------------------------------------------------------- self-test


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=== NWS Central Park (resolves the temp / precip questions) ===")
    cp = nws_central_park()
    for day in sorted(cp)[:7]:
        r = cp[day]
        t = r.get("max_temp_f")
        print(f"  {day}  max {t:.1f}F" if t is not None else f"  {day}  max   ?  ",
              f" precip {r.get('precip_prob_pct')}%")
    today = datetime.now(timezone.utc).date()
    for day in sorted(cp)[:3]:
        t = cp[day].get("max_temp_f")
        if t is None:
            continue
        lead = (datetime.fromisoformat(day).date() - today).days
        s = temp_sigma_f(lead)
        pc = normal_percentiles(t, s)
        print(f"    -> {day} lead {lead}d sigma {s:.1f}F: "
              + "  ".join(f"p{int(p)}={v:.0f}" for p, v in sorted(pc.items())))

    print("\n=== NWS active alerts (NY) ===")
    print(" ", nws_active_alerts("NY") or "none active")

    print("\n=== markets ===")
    c = crypto_spot()
    print(f"  BTC ${c['btc_usd']:,}   ETH ${c['eth_usd']:,}   "
          f"total mcap ${c['total_mcap_usd']/1e12:.3f}T")
    hist = btc_history(60)
    vol = realised_vol(hist)
    print(f"  BTC realised daily vol (60d): {vol:.4f} = {vol*100:.2f}%/day")
    for horizon in (1, 2):
        pc = lognormal_percentiles(c["btc_usd"], vol * math.sqrt(horizon))
        print(f"    {horizon}d: " + "  ".join(f"p{int(p)}={v:,.0f}" for p, v in sorted(pc.items())))

    print("\n=== NHC Atlantic ===")
    st = nhc_active("atlantic")
    print("  active named Atlantic storms:",
          [f"{s.get('name')} ({s.get('classification')})" for s in st] or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
