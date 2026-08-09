"""Win-rate tracker for the human vs the bot.

Records game results one at a time (or in batches) and reports the human's
win rate with a 95% Wilson confidence interval, plus advice on whether enough
games have been played.

Usage:
  python track_winrate.py add w        # record one game: w|l|d (draw)
  python track_winrate.py add wwldll   # record a batch string
  python track_winrate.py add --seat ai_first w   # optional per-seat tag
  python track_winrate.py show
  python track_winrate.py reset

Draws are counted as half a win by default (common game convention).
"""
import argparse
import math
import os
import sys

_BASE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(_BASE, "runs", "default", "winrate_log.json")


def _wilson(wins, games, z=1.96):
    """95% Wilson score interval for a binomial proportion."""
    if games == 0:
        return (0.0, 0.0, 0.0)
    p = wins / games
    denom = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denom
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denom
    return (center, max(0.0, center - half), min(1.0, center + half))


def _load():
    if not os.path.exists(LOG):
        return []
    import json
    with open(LOG, encoding="utf-8") as fh:
        return json.load(fh)


def _save(records):
    import json
    with open(LOG, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=1)


def cmd_add(args):
    records = _load()
    for ch in args.result:
        if ch not in "wld":
            raise SystemExit(f"bad result char: {ch!r} (use w/l/d)")
        records.append({"seat": args.seat, "result": ch})
    _save(records)
    print(f"recorded {len(args.result)} game(s); total {len(records)}")
    cmd_show(args)


def cmd_show(args):
    records = _load()
    if not records:
        print("no games recorded yet")
        return
    n = len(records)
    wins = sum(1 for r in records if r["result"] == "w")
    draws = sum(1 for r in records if r["result"] == "d")
    losses = n - wins - draws
    decided = n - draws
    half_wins = wins + 0.5 * draws   # draws count as half
    center, lo, hi = _wilson(half_wins, n)
    print(f"games={n}  W{wins} L{losses} D{draws}  "
          f"(decided {decided})")
    print(f"human win rate (draw=0.5): {100*center:.1f}%  "
          f"95% CI [{100*lo:.1f}%, {100*hi:.1f}%]")
    if decided:
        c2, lo2, hi2 = _wilson(wins, decided)
        print(f"win rate on decided games only: {100*c2:.1f}%  "
              f"[{100*lo2:.1f}%, {100*hi2:.1f}%]")
    # seat split
    by_seat = {}
    for r in records:
        by_seat.setdefault(r["seat"] or "none", []).append(r["result"])
    for seat, rs in by_seat.items():
        w = sum(1 for x in rs if x == "d" or x == "w")
        w_ = sum(1 for x in rs if x == "w")
        d_ = sum(1 for x in rs if x == "d")
        c, _, _ = _wilson(w_ + 0.5 * d_, len(rs))
        print(f"  seat {seat:12s}: games={len(rs)} "
              f"winrate={100*c:.0f}% (W{w_} D{d_})")
    # blocks of 25
    print("per-block (25):")
    for b in range(0, n, 25):
        blk = records[b:b + 25]
        w = sum(1 for r in blk if r["result"] == "w")
        d = sum(1 for r in blk if r["result"] == "d")
        c, _, _ = _wilson(w + 0.5 * d, len(blk))
        print(f"  games {b+1:3d}-{b+len(blk):3d}: W{w} D{d} -> {100*c:.0f}%")
    # stop advice
    half = (hi - lo) / 2
    if n < 50:
        print("advice: too few games; keep going (target >= ~100 for a real number)")
    elif half <= 0.07:
        print(f"advice: interval is tight (±{100*half:.0f}%); the number is stable enough.")
    elif lo > 0.75:
        print("advice: win rate is clearly above 75% (CI excludes 50%); direction is decided.")
    elif hi < 0.25:
        print("advice: win rate is clearly below 25%; direction is decided.")
    else:
        print(f"advice: interval still wide (±{100*half:.0f}%); "
              f"target ~200 games for ±7% unless the direction is decided.")


def cmd_reset(args):
    if os.path.exists(LOG):
        os.remove(LOG)
    print("log reset")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd")
    pa = sub.add_parser("add")
    pa.add_argument("--run", default="default",
                    help="per-run log under runs/<run>/winrate_log.json")
    pa.add_argument("result", help="string of w/l/d, e.g. wwldl")
    pa.add_argument("--seat", default="mixed",
                    help="tag like ai_first or human_first")
    ps = sub.add_parser("show")
    ps.add_argument("--run", default="default")
    pr = sub.add_parser("reset")
    pr.add_argument("--run", default="default")
    args = p.parse_args()
    global LOG
    LOG = os.path.join(_BASE, "runs", args.run, "winrate_log.json")
    if args.cmd == "add":
        cmd_add(args)
    elif args.cmd == "show":
        cmd_show(args)
    elif args.cmd == "reset":
        cmd_reset(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
