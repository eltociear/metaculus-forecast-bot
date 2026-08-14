#!/usr/bin/env python3
"""Score our own forecasts against how the questions actually resolved.

Nothing in this repo has ever measured whether the bot is any GOOD. Coverage was the lever
while questions were being missed, and coverage is now full — so the only thing standing
between us and prize money is accuracy, and accuracy was unmeasured. Every "improvement" to
the prompt or the ensemble so far has been shipped on taste.

This is the instrument. For every ledger entry it fetches the question, keeps the ones that
have RESOLVED, and scores three forecasters on the same set:

  us         — what we submitted
  crowd      — Metaculus' community prediction at resolution, the benchmark to beat
  always 0.5 — the no-skill floor

Brier score, lower is better. Binary only: multiple-choice and numeric need their own rules
and mixing them into one number would hide which half is failing.

    python scripts/metaculus_bot/score.py            # score everything resolved
    python scripts/metaculus_bot/score.py --verbose  # one line per question
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast import (API_BASE_URL, _request, load_env,  # noqa: E402
                      load_ledger, metaculus_headers)

# forecast.py calls this from its own main(), so importing it does NOT populate the token.
load_env()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def resolved_binary(q):
    """(outcome, community_p) for a resolved binary question, else None.

    `resolution` is the string "yes"/"no" on binary questions. Annulled and ambiguous
    resolutions are dropped rather than scored — they have no truth to score against, and
    counting them as 0 would quietly punish us for questions nobody could answer.
    """
    if q.get("type") != "binary":
        return None
    res = (q.get("resolution") or "").lower()
    if res not in ("yes", "no"):
        return None
    outcome = 1.0 if res == "yes" else 0.0
    community = None
    try:
        cp = (q.get("aggregations") or {}).get("recency_weighted", {})
        latest = cp.get("latest") or {}
        vals = latest.get("forecast_values") or latest.get("centers")
        if vals:
            community = float(vals[-1]) if len(vals) > 1 else float(vals[0])
    except Exception:  # noqa: BLE001 - a missing crowd number must not drop OUR score
        community = None
    return outcome, community


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    ledger = load_ledger().get("forecasts", {})
    binaries = {k: v for k, v in ledger.items() if v.get("type") == "binary"}
    print(f"ledger: {len(ledger)} forecasts, {len(binaries)} binary\n")

    rows, unresolved, unreadable = [], 0, 0
    for qid, entry in binaries.items():
        try:
            d = _request(f"{API_BASE_URL}/posts/{qid}/", headers=metaculus_headers())
        except Exception:  # noqa: BLE001
            # a post id that is not a post id: some ledger keys are question ids
            try:
                d = _request(f"{API_BASE_URL}/questions/{qid}/", headers=metaculus_headers())
            except Exception:  # noqa: BLE001
                unreadable += 1
                continue
        q = d.get("question") or d
        got = resolved_binary(q)
        if not got:
            unresolved += 1
            continue
        outcome, community = got
        p = entry.get("forecast")
        if not isinstance(p, (int, float)):
            unreadable += 1
            continue
        rows.append({"qid": qid, "title": (entry.get("title") or q.get("title") or "")[:60],
                     "p": float(p), "outcome": outcome, "community": community,
                     "tournament": entry.get("tournament")})

    print(f"resolved & scorable: {len(rows)}   still open: {unresolved}   unreadable: {unreadable}\n")
    if not rows:
        print("nothing resolved yet — this is a result, not a bug. Re-run as questions close.")
        return 0

    def brier(pairs):
        return sum((p - o) ** 2 for p, o in pairs) / len(pairs)

    ours = brier([(r["p"], r["outcome"]) for r in rows])
    naive = brier([(0.5, r["outcome"]) for r in rows])
    with_crowd = [r for r in rows if r["community"] is not None]
    crowd = brier([(r["community"], r["outcome"]) for r in with_crowd]) if with_crowd else None

    if a.verbose:
        print(f"{'ours':>6} {'crowd':>6} {'out':>4}  {'brier':>6}  question")
        for r in sorted(rows, key=lambda r: -((r["p"] - r["outcome"]) ** 2)):
            c = f"{r['community']:.2f}" if r["community"] is not None else "  -"
            print(f"{r['p']:>6.2f} {c:>6} {r['outcome']:>4.0f}  "
                  f"{(r['p']-r['outcome'])**2:>6.3f}  {r['title']}")
        print()

    print(f"{'forecaster':<22} {'Brier':>8}   (lower is better)")
    print("-" * 42)
    print(f"{'US':<22} {ours:>8.4f}")
    if crowd is not None:
        print(f"{'Metaculus community':<22} {crowd:>8.4f}   over {len(with_crowd)} of {len(rows)}")
    print(f"{'always 0.5 (no skill)':<22} {naive:>8.4f}")
    print()
    base = sum(r["outcome"] for r in rows) / len(rows)
    print(f"base rate YES in this set: {base:.1%}  (n={len(rows)})")
    if crowd is not None:
        verdict = "BEATING" if ours < crowd else "LOSING TO"
        print(f"\nwe are {verdict} the crowd by {abs(ours-crowd):.4f} Brier")
    if ours >= naive:
        print("\n⚠ WE ARE NO BETTER THAN GUESSING 0.5. Accuracy work is the whole job.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
