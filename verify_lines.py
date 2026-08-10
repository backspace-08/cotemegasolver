"""Scripted verification of claimed exploit lines, parameterised and parallel.

Usage:
    python verify_lines.py <lo> <hi> <workers> <policy> [policy ...]

Plays every seed in [lo, hi) on BOTH seats for each policy, so the count per
policy is 2 * (hi - lo). W/L/D are reported from the HUMAN side, because the
human is the one executing the candidate exploit line.

Windows needs spawn, hence the file plus __main__ guard.
"""

import sys
import time
from multiprocessing import Pool

sys.path.insert(0, "src")

from cote_megaverse.benchmark import run_match  # noqa: E402

LIMIT = 100


def _job(task):
    """Run one match. Never raises: a failure is reported, never dropped."""
    policy, seed, ai_starts = task
    try:
        report = run_match(seed=seed, policy=policy, depth=2,
                           max_half_turns=LIMIT, ai_starts=ai_starts)
        return policy, seed, ai_starts, report["winner"], None
    except Exception as exc:                      # noqa: BLE001
        return policy, seed, ai_starts, "error", f"{type(exc).__name__}: {exc}"


def wilson(successes, total, z=1.96):
    """95% Wilson score interval as a percentage pair."""
    if total == 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    margin = (z / denom) * ((phat * (1 - phat) / total
                             + z * z / (4 * total * total)) ** 0.5)
    return 100 * max(0.0, centre - margin), 100 * min(1.0, centre + margin)


def main():
    lo, hi, workers = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    policies = tuple(sys.argv[4:])
    seeds = range(lo, hi)
    per_policy = 2 * len(seeds)
    tasks = [(policy, seed, seat)
             for policy in policies for seed in seeds for seat in (False, True)]
    print(f"policies={policies} seeds={lo}-{hi - 1} "
          f"per_policy={per_policy} total={len(tasks)} "
          f"workers={workers} limit={LIMIT}", flush=True)

    t0 = time.time()
    with Pool(workers) as pool:
        results = pool.map(_job, tasks)
    elapsed = time.time() - t0

    errors = [r for r in results if r[3] == "error"]
    print(f"\ncompleted {len(results)}/{len(tasks)} matches in {elapsed:.0f}s"
          f"  ({elapsed / max(1, len(tasks)):.2f}s per match)\n")
    print(f"{'policy':10s} {'human W/L/D':16s} {'done':8s} "
          f"{'human_wr':10s} 95% Wilson")
    for policy in policies:
        rows = [r for r in results if r[0] == policy]
        w = sum(r[3] == "human" for r in rows)
        l = sum(r[3] == "ai" for r in rows)
        d = sum(r[3] == "draw" for r in rows)
        e = sum(r[3] == "error" for r in rows)
        done = w + l + d
        wr = 100 * w / done if done else 0.0
        low, high = wilson(w, done)
        print(f"{policy:10s} W{w:2d} L{l:2d} D{d:2d}      "
              f"{done:2d}/{per_policy:<4d} {wr:7.1f}%   "
              f"[{low:.1f}%; {high:.1f}%]"
              + (f"  errors={e}" if e else ""), flush=True)

    if errors:
        print("\nFAILED MATCHES (counted above, not dropped):")
        for policy, seed, seat, _winner, error in errors:
            print(f"  {policy} seed={seed} ai_first={seat}: {error}")

    print(f"\nelapsed={elapsed:.0f}s workers={workers}")


if __name__ == "__main__":
    main()
