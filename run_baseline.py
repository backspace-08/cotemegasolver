"""Benchmark runner: bot win rate vs the bundled policies, optionally parallel.

Usage:
  python run_baseline.py --temp 0.0 --seeds 100 --limit 100 --workers 8
"""
import sys
import time

sys.path.insert(0, "src")

from cote_megaverse.benchmark import benchmark_policies


def fmt(seat):
    return f"W{seat['wins']} L{seat['losses']} D{seat['draws']}"


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--limit", type=int, default=60)
    p.add_argument("--workers", type=int, default=None,
                   help="parallel processes (e.g. number of cores)")
    p.add_argument("--out", default=None, help="write report to this file")
    args = p.parse_args()

    t0 = time.time()
    r = benchmark_policies(seeds=range(args.seeds), depth=2,
                           max_half_turns=args.limit, temperature=args.temp,
                           workers=args.workers)
    dt = time.time() - t0

    lines = [f"temperature={args.temp} seeds={args.seeds} depth=2 "
             f"limit={args.limit} workers={args.workers} elapsed={dt:.0f}s\n"]
    lines.append(f"{'policy':14s} {'ai_first':16s} {'human_first':16s} "
                 f"{'combined':14s} missed_lethal\n")
    for pol, v in r.items():
        lines.append(f"{pol:14s} {fmt(v['ai_first']):16s} "
                     f"{fmt(v['human_first']):16s} "
                     f"{fmt({'wins': v['wins'], 'losses': v['losses'], 'draws': v['draws']}):14s} "
                     f"{v['missed_guaranteed_lethal']}\n")
    report = "".join(lines)
    print(report, flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report)


if __name__ == "__main__":
    main()
