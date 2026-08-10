"""Re-measure the `burster` control on its ORIGINAL seeds (0-24).

The verification run put `burster` at 15% human on seeds 200-219, against an
earlier 8% on seeds 0-24. Two explanations: seed variation, or the bot changed.
Replaying the original seeds separates them: if 0-24 still gives ~8%, the
difference is seeds; if it moved, the bot or harness changed.

Usage: python verify_control.py [workers]
"""

import sys
import time
from multiprocessing import Pool

sys.path.insert(0, "src")

from cote_megaverse.benchmark import run_match  # noqa: E402


def _job(task):
    policy, seed, ai_starts = task
    try:
        report = run_match(seed=seed, policy=policy, depth=2,
                           max_half_turns=100, ai_starts=ai_starts)
        return policy, report["winner"], None
    except Exception as exc:                      # noqa: BLE001
        return policy, "error", f"{type(exc).__name__}: {exc}"


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    tasks = [(policy, seed, seat)
             for policy in ("burster", "greedy")
             for seed in range(25) for seat in (False, True)]
    t0 = time.time()
    with Pool(workers) as pool:
        results = pool.map(_job, tasks)
    elapsed = time.time() - t0

    for policy in ("burster", "greedy"):
        rows = [r for r in results if r[0] == policy]
        w = sum(r[1] == "human" for r in rows)
        l = sum(r[1] == "ai" for r in rows)
        d = sum(r[1] == "draw" for r in rows)
        e = sum(r[1] == "error" for r in rows)
        done = w + l + d
        print(f"{policy:9s} seeds 0-24: human W{w} L{l} D{d} / {done}"
              f"  human_wr={100*w/max(1,done):5.1f}%"
              + (f"  errors={e}" if e else ""), flush=True)
    print(f"completed {len(results)} matches in {elapsed:.0f}s "
          f"workers={workers}")


if __name__ == "__main__":
    main()
