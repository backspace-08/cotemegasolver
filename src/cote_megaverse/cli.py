"""Command-line position analyzer for the new rules engine."""

import argparse
import json

from .benchmark import analyze_position
from .rules import Type, initial


def team(value):
    return tuple(Type[item.strip().upper()] for item in value.split(","))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default="A,B,C", type=team)
    parser.add_argument("--opponent", default="B,C,C", type=team)
    parser.add_argument("--depth", default=3, type=int)
    args = parser.parse_args()
    state = initial(args.team, args.opponent)
    report = analyze_position(state, depth=args.depth)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
