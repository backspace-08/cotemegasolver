"""End-to-end smoke test of the compact protocol.

Plays one full game through the CLI the way an LLM player does and asserts the
two properties the protocol optimisation is supposed to guarantee:

1. Every non-terminal response is ACTIONABLE: it is our turn and a budget is
   present, so the player never has to spend a round-trip on a "-" call.
2. The fairness contract holds: the bot's bank and held shields never appear
   in any payload.

Also prints the transcript size so a regression in verbosity is visible.
Usage: python smoke_compact.py [seed]
"""
import json
import shutil
import os
import subprocess
import sys

SEED = sys.argv[1] if len(sys.argv) > 1 else "504"
RUN = "smoke_compact"


def call(*args):
    proc = subprocess.run([sys.executable, "play.py", *args],
                          capture_output=True, text=True)
    return (proc.stdout or proc.stderr).strip()


def check_no_leak(raw, payload):
    bot = payload.get("bot") or (payload.get("state") or {}).get("bot") or {}
    assert "bank" not in bot, f"LEAK: bot bank exposed in {raw[:120]}"
    assert "sh" not in bot, f"LEAK: bot shields exposed in {raw[:120]}"


def main():
    shutil.rmtree(os.path.join("runs", RUN), ignore_errors=True)
    calls = 0
    total = 0
    dash_needed = 0

    raw = call("new", "--run", RUN, "--seed", SEED, "--human_first", "--compact")
    calls, total = calls + 1, total + len(raw)
    payload = json.loads(raw)
    check_no_leak(raw, payload)
    state = payload["state"]

    for _ in range(120):
        assert state["turn"] == "YOU", \
            f"non-actionable response: turn={state['turn']}"
        budget = state.get("you", {}).get("bud")
        assert budget is not None, "actionable response without a budget"
        if budget is None:
            dash_needed += 1

        # A plain legal intent: attack with everything the budget allows.
        raw = call("move", "--run", RUN, str(budget))
        calls, total = calls + 1, total + len(raw)
        payload = json.loads(raw)
        if "done" in payload:
            print(f"game over: {payload}")
            break
        check_no_leak(raw, payload)
        state = payload

    result = call("end", "--run", RUN)
    calls, total = calls + 1, total + len(result)
    print(f"end: {result}")

    print(f"\nround_trips={calls}  dead_dash_calls_needed={dash_needed}")
    print(f"transcript_chars={total}  avg_per_call={total // max(1, calls)}")
    assert dash_needed == 0, "protocol still requires dead '-' calls"
    print("OK: every response actionable, no fairness leak")
    shutil.rmtree(os.path.join("runs", RUN), ignore_errors=True)


if __name__ == "__main__":
    main()
