"""Measure the real per-turn cost of the compact protocol.

Plays one full game via the CLI exactly as an LLM would (one subprocess per
turn, intent chosen from the state's own hints) and reports round-trips, bytes,
and latency. This is the baseline any optimisation must beat.

Usage: python measure_protocol.py [seed] [run]
"""
import json
import subprocess
import sys
import time

SEED = sys.argv[1] if len(sys.argv) > 1 else "500"
RUN = sys.argv[2] if len(sys.argv) > 2 else "measure"


def call(args):
    t0 = time.time()
    proc = subprocess.run([sys.executable, "play.py", *args],
                          capture_output=True, text=True)
    latency = (time.time() - t0) * 1000
    out = (proc.stdout or proc.stderr).strip()
    return out, latency


def pick_intent(state):
    """Simple legal intent: attack with the whole budget when we know it.

    `bud` is only present when it is our turn; otherwise "-" advances the bot.
    """
    budget = state.get("you", {}).get("bud")
    return "-" if budget is None else str(budget)


def main():
    subprocess.run([sys.executable, "-c",
                    "import shutil,os;shutil.rmtree(os.path.join('runs',%r),"
                    "ignore_errors=True)" % RUN], check=False)
    calls = 0
    total = 0
    latencies = []

    out, lat = call(["new", "--run", RUN, "--seed", SEED,
                     "--human_first", "--compact"])
    calls += 1
    total += len(out)
    latencies.append(lat)
    state = json.loads(out)["state"]

    for _ in range(60):
        intent = pick_intent(state)
        out, lat = call(["move", "--run", RUN, intent])
        calls += 1
        total += len(out)
        latencies.append(lat)
        try:
            payload = json.loads(out)
        except json.JSONDecodeError:
            print("NON-JSON:", out[:200])
            break
        if "done" in payload or "result" in payload:
            print("finished:", payload)
            break
        state = payload

    print(f"round_trips={calls}")
    print(f"total_chars={total}  avg_chars_per_call={total // max(1, calls)}")
    print(f"total_latency_ms={sum(latencies):.0f}  "
          f"avg_latency_ms={sum(latencies) / max(1, len(latencies)):.0f}")


if __name__ == "__main__":
    main()
