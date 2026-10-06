"""Optionally rerun the OG 'test longer' control at 1, 1.5, 2 and 3 times its stop.

The main report reuses its archived aggregate. This independent experiment uses
the three frozen N=64 rate-pair designs with a 20,000-query cap per trial.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from simulation import ROOT, ScenarioEnvironment, most_likely
from git_bayesect import Bisector

MULTIPLIERS = (1.0, 1.5, 2.0, 3.0)
CELLS = ("e1_n64_0.2", "e1_n64_0.3", "e1_n64_0.5")


def trial(job: dict[str, Any]) -> dict[str, Any]:
    env = ScenarioEnvironment(job)
    bisector = Bisector(np.ones(job["n_arms"]))
    stop = None
    answers: dict[float, int] = {}
    for t in range(1, job["budget"] + 1):
        arm = bisector.select()
        bisector.record(arm, bool(env.sample(arm)))
        posterior = bisector.distribution
        if stop is None and posterior.max() >= 0.95:
            stop = t
        if stop is not None:
            for c in MULTIPLIERS:
                if c not in answers and t >= c * stop:
                    answers[c] = most_likely(posterior)
            if len(answers) == len(MULTIPLIERS):
                break
    return {
        "trial": job["trial"],
        "seed": job["seed"],
        "culprit": job["culprit"],
        "first_stop": stop,
        "answers": answers,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--trial-ids", nargs="+", type=int)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    design_path = ROOT / "evidence/results/full/design.json.gz"
    design = json.loads(gzip.decompress(design_path.read_bytes()))
    selected = [c for c in design if c["name"] in CELLS]
    jobs = {
        c["name"]: [
            {**j, "budget": 20000}
            for j in c["jobs"]
            if args.trial_ids is None or j["trial"] in args.trial_ids
        ]
        for c in selected
    }
    if args.workers < 1 or any(not group for group in jobs.values()):
        p.error("Nonpositive workers or no selected trials")
    if args.dry_run:
        print(f"{sum(map(len, jobs.values()))} OG trials in {len(jobs)} cells")
        return
    if args.out.exists() and any(args.out.iterdir()):
        p.error("Use an empty output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    lines = [
        "N=64, confidence 0.95: share of all trials answered wrongly",
        "rates          at stop 1.5x longer   2x longer   3x longer",
    ]
    if args.trial_ids is not None:
        lines.insert(0, "DEVELOPMENT PARTIAL RUN")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for cell, group in jobs.items():
            rows = list(pool.map(trial, group, chunksize=4))
            results[cell] = rows
            # A missing answer at the cap counts in the denominator, not as wrong.
            wrong = {
                c: float(
                    np.mean([c in r["answers"] and r["answers"][c] != r["culprit"] for r in rows])
                )
                for c in MULTIPLIERS
            }
            rate = group[0]["p_new"]
            lines.append(f"0.1->{rate:<5g} " + " ".join(f"{wrong[c]:>11.3f}" for c in MULTIPLIERS))
    (args.out / "uniform_extension.txt").write_text("\n".join(lines) + "\n")
    sources = [
        Path(__file__),
        Path(__file__).with_name("simulation.py"),
        ROOT / "git_bayesect.py",
        design_path,
    ]
    (args.out / "trials.json").write_text(
        json.dumps(
            {
                "partial": args.trial_ids is not None,
                "budget": 20000,
                "confidence": 0.95,
                "cells": results,
                "source_sha256": {
                    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sources
                },
            },
            indent=2,
        )
        + "\n"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
