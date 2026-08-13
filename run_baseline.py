"""Seeded AI-vs-human-policy baseline (parallel, file-based main).

Usage:
  python run_baseline.py --seeds 100 --limit 100 --workers 16 --out after_bench.txt

Do NOT invoke via `python -c "..."`: on Windows multiprocessing uses spawn and
needs a file-based ``__main__``.
"""
import argparse
import json
import sys

from cote_megaverse.benchmark import benchmark_policies

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--limit", type=int, default=100,
                   help="max half-turns per game")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", default="after_bench.txt")
    args = p.parse_args()
    result = benchmark_policies(
        seeds=range(args.seeds), depth=args.depth,
        max_half_turns=args.limit, workers=args.workers)
    summary = {policy: {k: v for k, v in data.items() if k != "matches"}
               for policy, data in result.items()}
    text = json.dumps(summary, indent=2, default=str)
    print(text)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print("wrote %s" % args.out)

if __name__ == "__main__":
    sys.path.insert(0, "src")
    main()
