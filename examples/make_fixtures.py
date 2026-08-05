"""Regenerate the example data. Seeded, so the files are reproducible.

python examples/make_fixtures.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write(name: str, records: list[dict]) -> None:
    path = HERE / name
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    print(f"wrote {path.name} ({len(records)} records)")


def main() -> None:
    rng = random.Random(42)

    # Two model runs on the same 200 items. Item difficulty drives shared
    # successes (correlated outcomes), and model B is genuinely a bit better.
    run_a, run_b = [], []
    for i in range(200):
        difficulty = rng.random()
        a_pass = rng.random() < (0.95 - 0.55 * difficulty)  # ~ 67% overall
        b_bonus = 0.08 if rng.random() < 0.9 else -0.02
        b_pass = rng.random() < min(0.98, 0.95 - 0.55 * difficulty + b_bonus)
        run_a.append({"id": f"item-{i:03d}", "score": int(a_pass)})
        run_b.append({"id": f"item-{i:03d}", "score": int(b_pass)})
    write("run_a.jsonl", run_a)
    write("run_b.jsonl", run_b)

    # One judge scoring 40 items 3 times each on a 1-10 scale, with noise.
    reps = []
    for i in range(40):
        true_score = rng.uniform(3.0, 9.5)
        for _ in range(3):
            noisy = round(min(10.0, max(1.0, rng.gauss(true_score, 0.8))))
            reps.append({"id": f"item-{i:03d}", "score": noisy})
    write("judge_repeats.jsonl", reps)


if __name__ == "__main__":
    main()
