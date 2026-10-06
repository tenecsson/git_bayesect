"""Time production statistical kernels; excludes Git, printing and state-file IO."""

from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from git_bayesect import Bisector, SafeExperiment, SafeStopping


def measure(n: int, mode: str, repeat: int, warmup: int, steps: int) -> dict[str, Any]:
    b = Bisector([1] * n)
    safe = SafeStopping(n, mode.startswith("checked")) if mode != "default" else None
    action_rng = np.random.default_rng(4333 + repeat)
    outcome_rng = np.random.default_rng(789 + repeat)
    elapsed = []
    for t in range(warmup + steps):
        start = time.perf_counter_ns()
        experiment = SafeExperiment.choose(b, action_rng) if safe is not None else None
        i = b.select() if experiment is None else experiment.index
        y = bool(outcome_rng.random() < (0.3 if i <= n // 2 else 0.1))
        b.record(i, y)
        if safe is not None:
            safe.record(b, i, y, experiment, check=mode != "checked4" or (t + 1) % 4 == 0)
        _ = b.distribution
        end = time.perf_counter_ns()
        if t >= warmup:
            elapsed.append((end - start) / 1000)
    return {
        "N": n,
        "mode": mode,
        "repeat": repeat,
        "mean_us": float(np.mean(elapsed)),
        "median_us": float(np.median(elapsed)),
        "p95_us": float(np.quantile(elapsed, 0.95)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    started = time.perf_counter()
    records = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for n in [64, 256, 1024, 4096]:
            for repeat in range(4):
                for mode in ["default", "safe", "checked1", "checked4"]:
                    record = measure(n, mode, repeat, 100, 300)
                    if repeat:
                        records.append(record)
    means = [
        {
            "N": n,
            **{
                mode: float(
                    np.mean([r["mean_us"] for r in records if r["N"] == n and r["mode"] == mode])
                )
                for mode in ["default", "safe", "checked1", "checked4"]
            },
        }
        for n in [64, 256, 1024, 4096]
    ]
    report = {
        "protocol": {
            "warmup": 100,
            "measured_steps": 300,
            "discarded_repeats": 1,
            "measured_repeats": 3,
            "scope": "complete statistical step including posterior materialization, excluding Git/state IO",
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "production_sha256": hashlib.sha256((ROOT / "git_bayesect.py").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
        "means_us": means,
        "repeats": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(means))


if __name__ == "__main__":
    main()
