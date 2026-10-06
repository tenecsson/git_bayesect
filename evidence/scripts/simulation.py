"""Reproduce the documented environments and stopping endpoints using production code.

The scenario generator and endpoint accounting preserve the original suite. Saved
OG controls are read from trial_endpoints.csv.gz, not rerun.
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import numpy.typing as npt
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from git_bayesect import Bisector, SafeExperiment, SafeStopping

FloatArray = npt.NDArray[np.float64]


def most_likely(posterior: FloatArray) -> int:
    """git bayesect's reported commit: the posterior argmax, ties broken toward the middle."""
    ties = np.flatnonzero(np.isclose(posterior, posterior.max(), rtol=1e-12, atol=1e-15))
    return int(ties[np.argmin(np.abs(ties - len(posterior) // 2))])


class ScenarioEnvironment:
    """Flaky-test simulator for one trial, one reproducible outcome stream per commit.

    The kth run of a commit consumes the same uniforms however runs of other commits are
    interleaved, so selectors and rules are compared on common random numbers.
    """

    def __init__(self, job: dict[str, Any]) -> None:
        n, culprit = job["n_arms"], job["culprit"]
        p_new, p_old = job["p_new"], job["p_old"]
        master = random.Random(job["seed"])
        self._streams = [random.Random(master.randbytes(16)) for _ in range(n)]
        shape = np.random.default_rng(
            [job["seed"], 7]
        )  # scenario randomness, separate from outcomes
        idx = np.arange(n)
        mid = 0.5 * (p_new + p_old)
        self.rates: FloatArray = np.where(idx <= culprit, p_new, p_old).astype(float)
        self.targets: tuple[int, ...] = (culprit,)
        self._shift_at: int | None = None
        self._shift = 0.0
        self._repeat = 0.0
        scenario = job["scenario"]
        if scenario == "jitter":
            self.rates = expit(logit(self.rates) + shape.normal(0.0, job["jitter_sigma"], n))
        elif scenario == "drift":
            self._shift_at, self._shift = job["drift_at"], job["drift_shift"]
        elif scenario in ("two_step", "partial_fix"):
            while True:
                b1, b2 = sorted(int(v) for v in shape.choice(np.arange(n - 1), 2, replace=False))
                if b2 - b1 >= 2:
                    break
            if scenario == "two_step":
                self.rates = np.select([idx <= b1, idx <= b2], [p_new, mid], p_old)
                self.targets = (b1, b2)
            else:
                self.rates = np.select([idx <= b1, idx <= b2], [mid, p_new], p_old)
                self.targets = (b2,)
        elif scenario == "broken":
            if culprit <= n - 3:
                self.rates[int(shape.integers(culprit + 1, n - 1))] = 1.0
        elif scenario == "bursty":
            self._repeat = job["burst_rho"]
        elif scenario == "ramp":
            width = job["ramp_width"]
            start = min(culprit, n - 1 - width)
            ramp = p_new + (p_old - p_new) * (idx - start) / width
            self.rates = np.clip(ramp, min(p_new, p_old), max(p_new, p_old))
            self.targets = tuple(range(start, start + width))
        elif scenario != "step":
            raise ValueError(f"unknown scenario {scenario}")
        # a user prior boosting one commit that is not an acceptable answer (E6)
        others = [i for i in range(n) if i not in self.targets]
        self.boosted = int(shape.choice(others)) if job["prior_boost"] != 1.0 else None
        self._t = 0
        self._last: list[int | None] = [None] * n

    def sample(self, arm: int) -> int:
        rate = self.rates[arm]
        if self._shift_at is not None and self._t >= self._shift_at:
            rate = expit(logit(rate) + self._shift)
        self._t += 1
        stream = self._streams[arm]
        if self._repeat:
            repeat = stream.random() < self._repeat
            fresh = int(stream.random() < rate)
            last = self._last[arm]
            x = last if repeat and last is not None else fresh
        else:
            x = int(stream.random() < rate)
        self._last[arm] = x
        return x


def answer_status(answer: Any, targets: tuple[int, ...]) -> tuple[str, bool]:
    if answer is None:
        return "empty", False
    if isinstance(answer, (tuple, list)):
        correct = any(int(answer[0]) <= b <= int(answer[1]) for b in targets)
    else:
        correct = int(answer) in targets
    return ("correct" if correct else "wrong"), correct


def trial(job: dict[str, Any], alphas: tuple[float, ...]) -> dict[str, Any]:
    """Run the production safe method once and evaluate all thresholds and rules on its path."""
    started = time.perf_counter()
    cpu_started = time.process_time()
    n = int(job["n_arms"])
    env = ScenarioEnvironment(job)
    targets = tuple(env.targets)
    prior = np.ones(n)
    if env.boosted is not None:
        prior[env.boosted] = job["prior_boost"]
    a_new, b_new, a_old, b_old = job["beta_priors"]
    b = Bisector(prior, alpha_new=a_new, beta_new=b_new, alpha_old=a_old, beta_old=b_old)
    safe = SafeStopping(n, check_assumptions=True)
    counts = np.zeros(n, dtype=int)
    rng = np.random.default_rng([int(job["seed"]), 4333])
    logs = np.zeros(n)
    peaks = np.full(n, -np.inf)
    thresholds = -np.log(np.asarray(alphas, dtype=float))
    rules = ("singleton", "range", "check", "posterior")
    stops: dict[str, list[dict[str, Any] | None]] = {name: [None] * len(alphas) for name in rules}
    open_ = {name: np.ones(len(alphas), dtype=bool) for name in rules}
    arms: list[int] = []
    outcomes: list[int] = []
    t = 0
    last_wrong_map = 0
    while t < int(job["budget"]):
        # Allocation and forecast are fixed before this round's environment draw.
        experiment = SafeExperiment.choose(b, rng)
        arm = experiment.index
        y = int(env.sample(arm))
        counts[arm] += 1
        b.record(arm, bool(y))
        t += 1
        safe.record(b, arm, bool(y), experiment, check=t % int(job["check_every"]) == 0)
        logs, peaks = safe.log_evidence, safe.max_log_evidence
        arms.append(arm)
        outcomes.append(y)

        posterior = b.distribution
        map_id = most_likely(posterior)
        if map_id not in targets:
            last_wrong_map = t
        # RIPr survivor set is defined by permanent maxima, not current wealth.
        alive = peaks < thresholds[:, None]
        any_alive = alive.any(axis=1)
        singleton = alive.sum(axis=1) == 1
        none_alive = ~any_alive
        for rule, fired in (
            ("singleton", singleton | none_alive),
            (
                "range",
                none_alive
                | (
                    any_alive
                    & (
                        (n - 1 - np.argmax(alive[:, ::-1], axis=1) - np.argmax(alive, axis=1) + 1)
                        <= int(job["range_width"])
                    )
                ),
            ),
        ):
            new = open_[rule] & fired
            for ix in np.flatnonzero(new):
                if rule == "singleton" and singleton[ix]:
                    ans: Any = int(np.flatnonzero(alive[ix])[0])
                elif rule == "range" and any_alive[ix]:
                    ans = (int(np.argmax(alive[ix])), int(n - 1 - np.argmax(alive[ix, ::-1])))
                else:
                    ans = None
                status, correct = answer_status(ans, targets)
                stops[rule][ix] = dict(
                    step=t,
                    answer=ans,
                    status=status,
                    correct=correct,
                    true_rejected=not bool(alive[ix, list(targets)].any()),
                    targets_rejected=[b for b in targets if not alive[ix, b]],
                    targets_alive=[b for b in targets if alive[ix, b]],
                )
            open_[rule] &= ~fired

        # The original SafeStopping model-check is updated once, every four rounds.
        misfit = safe.max_log_misfit >= thresholds
        count = alive.sum(axis=1)
        check_fired = (count <= 1) | misfit
        new = open_["check"] & check_fired
        for ix in np.flatnonzero(new):
            if misfit[ix]:
                stops["check"][ix] = dict(
                    step=t,
                    answer=None,
                    status="check_abstain",
                    correct=False,
                    true_rejected=not bool(alive[ix, list(targets)].any()),
                    targets_rejected=[b for b in targets if not alive[ix, b]],
                    targets_alive=[b for b in targets if alive[ix, b]],
                    reason="assumption_check",
                )
            else:
                ans = int(np.flatnonzero(alive[ix])[0]) if count[ix] == 1 else None
                status, correct = answer_status(ans, targets)
                stops["check"][ix] = dict(
                    step=t,
                    answer=ans,
                    status=status,
                    correct=correct,
                    true_rejected=not bool(alive[ix, list(targets)].any()),
                    targets_rejected=[b for b in targets if not alive[ix, b]],
                    targets_alive=[b for b in targets if alive[ix, b]],
                    reason="riPr_empty" if ans is None else "singleton",
                )
        open_["check"] &= ~check_fired

        posterior_stop = posterior[map_id] >= 1.0 - np.asarray(alphas)
        new = open_["posterior"] & posterior_stop
        for ix in np.flatnonzero(new):
            status, correct = answer_status(map_id, targets)
            stops["posterior"][ix] = dict(
                step=t,
                answer=int(map_id),
                status=status,
                correct=correct,
                true_rejected=None,
                targets_rejected=[b for b in targets if not alive[ix, b]],
                targets_alive=[b for b in targets if alive[ix, b]],
            )
        open_["posterior"] &= ~posterior_stop

        if job["full_budget"] is False and not any(mask.any() for mask in open_.values()):
            break

    for rule in rules:
        for endpoint_ix, record in enumerate(stops[rule]):
            if record is None:
                alive = peaks < thresholds[endpoint_ix]
                ans = None
                status = "censored"
                stops[rule][endpoint_ix] = dict(
                    step=None,
                    answer=ans,
                    status=status,
                    correct=False,
                    true_rejected=(
                        not bool(alive[list(targets)].any()) if rule != "posterior" else None
                    ),
                    targets_rejected=[b for b in targets if not alive[b]],
                    targets_alive=[b for b in targets if alive[b]],
                )
    final_set = np.flatnonzero(peaks < -math.log(0.05))
    return {
        "trial": int(job["trial"]),
        "seed": int(job["seed"]),
        "n_arms": n,
        "culprit": int(job["culprit"]),
        "targets": list(targets),
        "p_new": float(job["p_new"]),
        "p_old": float(job["p_old"]),
        "budget": int(job["budget"]),
        "steps_run": t,
        "elapsed_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu_started,
        "last_unacceptable_map_step": last_wrong_map,
        "posterior_settle_step": (
            last_wrong_map + 1 if most_likely(b.distribution) in targets else None
        ),
        "partition": job.get("partition", "suite"),
        "scenario": job["scenario"],
        "prior_boost_index": env.boosted,
        "observations": {"arms": arms, "outcomes": outcomes},
        "stops": stops,
        "final": {
            "counts": counts.tolist(),
            "log_wealth": logs.tolist(),
            "max_log_wealth": peaks.tolist(),
            "alive_at_0.05": final_set.astype(int).tolist(),
            "posterior_map": int(most_likely(b.distribution)),
            "safe_max_log_misfit": float(safe.max_log_misfit),
            "true_rejected_at_0.05": int(int(job["culprit"]) not in final_set),
            "targets_rejected_at_0.05": [x for x in targets if x not in final_set],
            "targets_alive_at_0.05": [x for x in targets if x in final_set],
            "target_intersects_set_at_0.05": any(x in final_set for x in targets),
        },
    }
