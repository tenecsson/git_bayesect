"""Replay saved observations through the production policy; generates no new trials."""

from pathlib import Path
import argparse, gzip, hashlib, json, sys, time
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from git_bayesect import Bisector, SafeExperiment, SafeStopping
from simulation import ScenarioEnvironment

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
paths = sorted((ROOT / "evidence/fixtures").glob("*.json.gz"))
assert len(paths) == 63, "Expected one path per cell and three outcome edge cases"
started = time.monotonic()
results = []
for path in paths:
    data = json.loads(gzip.decompress(path.read_bytes()))
    job = data["job"]
    environment = ScenarioEnvironment(job)
    assert list(environment.targets) == data["targets"]
    assert environment.boosted == data["prior_boost_index"]
    n = job["n_arms"]
    prior = np.ones(n)
    if data["prior_boost_index"] is not None:
        prior[data["prior_boost_index"]] = job["prior_boost"]
    b = Bisector(prior, *job["beta_priors"])
    g = SafeStopping(n, True)
    rng = np.random.default_rng([job["seed"], 4333])
    thresholds = -np.log(data["alpha_grid"])
    first: dict[str, list[Any]] = {
        r: [None] * len(thresholds) for r in ["singleton", "range", "check"]
    }
    for t, (arm, y) in enumerate(
        zip(data["observations"]["arms"], data["observations"]["outcomes"]), 1
    ):
        experiment = SafeExperiment.choose(b, rng)
        assert experiment.index == arm, (path.name, t, experiment.index, arm)
        assert environment.sample(arm) == y, (path.name, t, "environment outcome")
        b.record(arm, bool(y))
        g.record(b, arm, bool(y), experiment, check=t % job["check_every"] == 0)
        for k, threshold in enumerate(thresholds):
            if all(first[r][k] is not None for r in first):
                continue
            alive = np.flatnonzero(g.max_log_evidence < threshold)
            for rule in first:
                if first[rule][k] is not None:
                    continue
                answer = None
                status = None
                if rule == "check" and g.max_log_misfit >= threshold:
                    status = "check_abstain"
                elif not len(alive):
                    status = "empty"
                elif len(alive) == 1 or (
                    rule == "range" and alive[-1] - alive[0] + 1 <= job["range_width"]
                ):
                    answer = [int(alive[0]), int(alive[-1])] if rule == "range" else int(alive[0])
                    correct = (
                        any(answer[0] <= x <= answer[1] for x in data["targets"])
                        if isinstance(answer, list)
                        else answer in data["targets"]
                    )
                    status = "correct" if correct else "wrong"
                if status is not None:
                    first[rule][k] = {"step": t, "status": status, "answer": answer}
    for rule, endpoints in first.items():
        for got, saved in zip(endpoints, data["stops"][rule]):
            if got is None:
                assert saved["status"] == "censored", (path, rule, saved)
            else:
                for key in ["step", "status", "answer"]:
                    assert got[key] == saved[key], (path, rule, key, got, saved)
    wealth_error = float(np.max(abs(g.log_evidence - data["final"]["log_wealth"])))
    peak_error = float(np.max(abs(g.max_log_evidence - data["final"]["max_log_wealth"])))
    assert wealth_error < 1e-9 and peak_error < 1e-9, (path, wealth_error, peak_error)
    assert abs(g.max_log_misfit - data["final"]["safe_max_log_misfit"]) < 1e-8
    results.append(
        {
            "cell": data["cell"],
            "trial": data["trial"],
            "observations": data["steps_run"],
            "wealth_max_abs_error": wealth_error,
            "peak_max_abs_error": peak_error,
            "source_trial_sha256": hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest(),
        }
    )
    print(data["cell"], data["trial"], data["steps_run"], flush=True)
report = {
    "status": "passed",
    "scope": "Every announced arm and regenerated environment outcome, final wealth and peaks, and first singleton/range/check endpoints at every recorded alpha; one full path per historical cell plus wrong, empty and sentinel examples.",
    "trials": results,
    "total_observations": sum(x["observations"] for x in results),
    "elapsed_seconds": time.monotonic() - started,
    "production_sha256": hashlib.sha256((ROOT / "git_bayesect.py").read_bytes()).hexdigest(),
    "validation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "new_statistical_trials": 0,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n")
