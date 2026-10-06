"""Rebuild the complete evidence report from packaged endpoints or fresh RIPr runs.

OG controls are always reused. The report never imports research code or reads
published summaries, except when --verify-published explicitly checks its output.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Sequence
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import shutil
from typing import Any

import numpy as np
from scipy.special import expit, xlogy

from analysis import (
    PATH_RULES,
    RULES,
    cost_of,
    get_baseline,
    make_cell_summary,
    outcome_summary,
    paired_summary,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evidence/results/full"
SLOPE_CELLS = ("e1_n64_0.2", "e1_n64_0.3", "e1_n64_0.5")
SLOPE_METHODS = ("champion_singleton", "champion_posterior", "saved_bayes")
Record = dict[str, Any]
Controls = dict[tuple[int, float], Record]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[Record]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def integers(value: str) -> list[int]:
    return [int(x) for x in value.split("|") if x]


def parse_cell(
    raw: list[dict[str, str]], cell: Record, extras: dict[tuple[str, int], Record]
) -> tuple[list[Record], Controls]:
    """Restore report records, checking every trial/alpha against the frozen jobs."""
    jobs = {j["trial"]: j for j in cell["jobs"]}
    trials: dict[int, Record] = {}
    controls: Controls = {}
    alphas = cell["alpha_grid"]
    for row in raw:
        tid, alpha = int(row["trial"]), float(row["alpha"])
        if tid not in jobs or alpha not in alphas or (tid, alpha) in controls:
            raise ValueError(f"Unknown or duplicate endpoint: {cell['name']}/{tid}/{alpha}")
        job = jobs[tid]
        for field, key in (
            ("seed", "seed"),
            ("nominal_culprit", "culprit"),
            ("N", "n_arms"),
            ("budget", "budget"),
        ):
            if int(row[field]) != job[key]:
                raise ValueError(f"Endpoint/job mismatch: {cell['name']}/{tid}/{field}")
        for key in ("p_new", "p_old"):
            if float(row[key]) != job[key]:
                raise ValueError(f"Endpoint/job rate mismatch: {cell['name']}/{tid}/{key}")
        if (row["full_budget"] == "True") != job["full_budget"]:
            raise ValueError("Endpoint/job horizon mismatch")
        extra = extras[cell["name"], tid]
        if extra["seed"] != job["seed"]:
            raise ValueError("Supplemental input/job seed mismatch")
        info = {
            "cell": cell["name"],
            "trial": tid,
            "seed": job["seed"],
            "job": job,
            "alpha_grid": alphas,
            "culprit": job["culprit"],
            "n_arms": job["n_arms"],
            "p_new": job["p_new"],
            "p_old": job["p_old"],
            "budget": job["budget"],
            "targets": integers(row["targets"]),
            "steps_run": int(row["steps_run"]),
            "posterior_settle_step": (
                int(row["posterior_settle_step"]) if row["posterior_settle_step"] else None
            ),
            "last_unacceptable_map_step": int(row["last_unacceptable_map_step"]),
            "final": extra["final"],
        }
        r = trials.setdefault(tid, {**info, "stops": {rule: {} for rule in PATH_RULES}})
        if any(r[k] != v for k, v in info.items()):
            raise ValueError("Inconsistent trial metadata across thresholds")
        for rule in PATH_RULES:
            prefix = "champion_" + rule + "_"
            status = row[prefix + "status"]
            step = int(row[prefix + "step"]) if row[prefix + "step"] else None
            if (status == "censored") != (step is None):
                raise ValueError("Endpoint status and censoring disagree")
            answer = ast.literal_eval(row[prefix + "answer"]) if row[prefix + "answer"] else None
            r["stops"][rule][alpha] = {
                "step": step,
                "answer": answer,
                "status": status,
                "correct": row[prefix + "correct"] == "True",
                "true_rejected": (
                    (row[prefix + "true_rejected"] == "True")
                    if row[prefix + "true_rejected"]
                    else None
                ),
                "targets_rejected": integers(row[prefix + "targets_rejected"]),
                "targets_alive": integers(row[prefix + "targets_alive"]),
            }
        controls[tid, alpha] = {
            "status": row["saved_bayes_status"],
            "stop_step": int(row["saved_bayes_cost"]),
            "answer": row["saved_bayes_answer"],
            "correct": row["saved_bayes_correct"],
            **extra["saved_bayes"],
        }
    if set(trials) != set(jobs) or len(controls) != len(jobs) * len(alphas):
        raise ValueError(f"Incomplete retained cell: {cell['name']}")
    for r in trials.values():
        r["stops"] = {rule: [r["stops"][rule][a] for a in alphas] for rule in PATH_RULES}
    return sorted(trials.values(), key=lambda r: r["trial"]), controls


def endpoint_row(r: Record, alpha: float, ix: int, controls: Controls) -> Record:
    out = {
        "cell": r["cell"],
        "trial": r["trial"],
        "seed": r["seed"],
        "nominal_culprit": r["culprit"],
        "targets": "|".join(map(str, r["targets"])),
        "N": r["n_arms"],
        "p_new": r["p_new"],
        "p_old": r["p_old"],
        "budget": r["budget"],
        "steps_run": r["steps_run"],
        "full_budget": r["job"]["full_budget"],
        "alpha": alpha,
        "posterior_settle_step": r["posterior_settle_step"],
        "last_unacceptable_map_step": r["last_unacceptable_map_step"],
    }
    for rule in PATH_RULES:
        rec = r["stops"][rule][ix]
        for key in (
            "step",
            "answer",
            "status",
            "correct",
            "true_rejected",
            "targets_rejected",
            "targets_alive",
        ):
            value = rec[key]
            out[f"champion_{rule}_{key}"] = (
                "|".join(map(str, value)) if key.startswith("targets_") else value
            )
        if rule == "range":
            answer = rec["answer"]
            out["range_width_at_stop"] = (
                answer[1] - answer[0] + 1 if isinstance(answer, list) else None
            )
    b = controls[r["trial"], alpha]
    out.update(
        saved_bayes_status=b["status"],
        saved_bayes_cost=b["stop_step"],
        saved_bayes_answer=b["answer"],
        saved_bayes_correct=b["correct"],
    )
    return out


def records_at(rows: list[Record], controls: Controls, method: str, alpha: float) -> list[Record]:
    if method == "saved_bayes":
        return [
            dict(
                status=(b := controls[r["trial"], alpha])["status"],
                step=None if b["status"] == "censored" else b["stop_step"],
            )
            for r in rows
        ]
    rule = method.removeprefix("champion_")
    return [r["stops"][rule][r["alpha_grid"].index(alpha)] for r in rows]


def prior_bins(rows: list[Record], controls: Controls) -> list[Record]:
    bins = (
        (0, 0.05, "[0,.05)"),
        (0.05, 0.1, "[.05,.1)"),
        (0.1, 0.2, "[.1,.2)"),
        (0.2, 0.4, "[.2,.4)"),
        (0.4, 1.0, "[.4,1]"),
    )
    groups = []
    for lo, hi, label in bins:
        groups.append(
            (
                "non-sentinel; p_old draw used",
                label,
                [
                    r
                    for r in rows
                    if r["culprit"] != r["n_arms"] - 1
                    and lo <= abs(r["p_new"] - r["p_old"])
                    and (
                        abs(r["p_new"] - r["p_old"]) <= hi
                        if hi == 1
                        else abs(r["p_new"] - r["p_old"]) < hi
                    )
                ],
            )
        )
    groups.append(
        (
            "constant-rate sentinel; p_old draw unused",
            "sentinel",
            [r for r in rows if r["culprit"] == r["n_arms"] - 1],
        )
    )
    result = []
    for population, label, group in groups:
        if not group:
            continue
        budgets = [r["budget"] for r in group]
        for method in ("champion_singleton", "champion_check", "champion_posterior", "saved_bayes"):
            result.append(
                {
                    "population": population,
                    "gap_bin": label,
                    "rule": method,
                    **outcome_summary(records_at(group, controls, method, 0.05), budgets),
                }
            )
        cc = records_at(group, controls, "champion_singleton", 0.05)
        og = records_at(group, controls, "saved_bayes", 0.05)
        diffs = [cost_of(c, b) - cost_of(o, b) for c, o, b in zip(cc, og, budgets)]
        result.append(
            {
                "population": population,
                "gap_bin": label,
                "rule": "champion_singleton_vs_saved",
                "paired_control": "OG",
                **paired_summary(diffs, [r["culprit"] for r in group], "original_design_position"),
            }
        )
    return result


def capacity(p: float, q: float) -> tuple[float, float]:
    def entropy(x: float) -> float:
        return float(-xlogy(x, x) - xlogy(1 - x, 1 - x))

    if abs(p - q) < 1e-10:
        return 0.0, 0.5
    m = float(expit(-(entropy(p) - entropy(q)) / (p - q)))
    weight = float(np.clip((m - q) / (p - q), 0, 1))
    return entropy(m) - weight * entropy(p) - (1 - weight) * entropy(q), weight


def slopes(cell: str, rows: list[Record], controls: Controls) -> tuple[Record, Record]:
    rate, weight = capacity(rows[0]["p_new"], rows[0]["p_old"])
    differences: dict[str, dict[int, int]] = {}
    for method in SLOPE_METHODS:
        low, high = [records_at(rows, controls, method, a) for a in (1e-5, 1e-6)]
        differences[method] = {
            r["trial"]: int(b["step"]) - int(a["step"])
            for r, a, b in zip(rows, low, high)
            if a["status"] != "censored" and b["status"] != "censored"
        }
    common = set.intersection(*(set(d) for d in differences.values()))

    def mean(values: Sequence[float]) -> float | None:
        return float(np.mean(values)) if values else None

    cc = list(differences["champion_singleton"].values())
    small = {
        "cell": cell,
        "n_trials": len(rows),
        "paired_uncensored_trials": len(cc),
        "censored_at_either_alpha": len(rows) - len(cc),
        "mean_extra_samples_per_nat_uncensored": float(np.mean(cc) / math.log(10)) if cc else None,
        "mean_extra_samples_per_log10_alpha_uncensored": mean(cc),
        "oracle_capacity_nats_per_sample": rate,
        "oracle_samples_per_nat": 1 / rate,
        "interpretation": "Uncensored paired subset only; capped observations are listed, not treated as exact endpoints.",
    }
    shared = {
        "cell": cell,
        "p_old": rows[0]["p_old"],
        "p_new": rows[0]["p_new"],
        "oracle_rate": rate,
        "oracle_samples_per_nat": 1 / rate,
        "oracle_pair_lambda": weight,
        "common_uncensored_trials": len(common),
        "methods": {
            method: {
                "n": len(rows),
                "uncensored_n": len(d),
                "samples_per_nat_uncensored": mean([v / math.log(10) for v in d.values()]),
                "samples_per_nat_common_uncensored": mean(
                    [d[t] / math.log(10) for t in sorted(common)]
                ),
            }
            for method, d in differences.items()
        },
    }
    return small, shared


def compare_values(actual: Any, expected: Any, path: str = "") -> int:
    """Compare all numerical/structural results; explanatory wording may change."""
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"Keys differ at {path}: {set(actual) ^ set(expected)}")
        return sum(compare_values(actual[k], v, path + "/" + k) for k, v in expected.items())
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"Lengths differ at {path}")
        return sum(
            compare_values(a, b, f"{path}/{i}") for i, (a, b) in enumerate(zip(actual, expected))
        )
    if isinstance(expected, str):
        if any(word in path for word in ("interpretation", "definition", "sha256")):
            return 0
        # CSV nested cells contain Python literals, so compare their numeric contents.
        try:
            a, b = ast.literal_eval(actual), ast.literal_eval(expected)
        except (ValueError, SyntaxError):
            pass
        else:
            return compare_values(a, b, path)
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(f"Numeric mismatch at {path}: {actual!r} != {expected!r}")
    elif actual != expected:
        raise ValueError(f"Mismatch at {path}: {actual!r} != {expected!r}")
    return 1


def verify_published(out: Path) -> int:
    checked = 0
    for name in ("summary.json", "common-slope-audit.json"):
        actual, expected = [json.loads((p / name).read_text()) for p in (out, DATA)]
        if name == "summary.json":
            actual, expected = actual["cells"], expected["cells"]
        else:
            actual, expected = actual["cells"], expected["cells"]
        checked += compare_values(actual, expected, name)
    for name in ("paired_costs.csv", "prior_gap_bins.csv", "small_alpha_slopes.csv"):
        with (out / name).open(newline="") as a, (DATA / name).open(newline="") as b:
            checked += compare_values(list(csv.DictReader(a)), list(csv.DictReader(b)), name)
    if (out / "report.txt").read_text().strip() != (DATA / "report.txt").read_text().strip():
        raise ValueError("Human-readable report differs")
    return checked


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--run", type=Path, help="Fresh run_evidence.py output; otherwise use retained endpoints"
    )
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Explicitly label a selected or unfinished run as partial",
    )
    p.add_argument(
        "--verify-published",
        action="store_true",
        help="Compare every statistic with the accepted full report",
    )
    args = p.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        p.error("Use an empty output directory")
    design = json.loads(gzip.decompress((DATA / "design.json.gz").read_bytes()))
    supplement = json.loads(gzip.decompress((DATA / "analysis-inputs.json.gz").read_bytes()))
    extras = {(r["cell"], r["trial"]): r for r in supplement["trials"]}
    definitions = {c["name"]: c for c in design}
    expected_scope = {(c["name"], j["trial"]) for c in design for j in c["jobs"]}
    if set(extras) != expected_scope or len(extras) != len(supplement["trials"]):
        raise ValueError("Supplemental inputs do not cover exactly the frozen jobs")
    sources = [
        Path(__file__),
        Path(__file__).with_name("analysis.py"),
        DATA / "design.json.gz",
        DATA / "analysis-inputs.json.gz",
        DATA / "trial_endpoints.csv.gz",
        DATA / "archived_uniform_extension.txt",
    ]
    scope = expected_scope
    missing: list[tuple[str, int]] = []
    if args.run:
        manifest_path = args.run / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        sources.append(manifest_path)
        for relative, expected in manifest["sources"].items():
            source = ROOT / relative
            if sha(source) != expected:
                raise ValueError(f"Run source changed: {relative}")
            sources.append(source)
        scope = {tuple(pair) for pair in manifest["scope"]}
        if not scope or len(scope) != len(manifest["scope"]) or not scope <= expected_scope:
            raise ValueError("Invalid run scope")
        for cell, tid in sorted(scope):
            path = args.run / cell / f"trial-{tid:05d}.json"
            if path.exists():
                sources.append(path)
            else:
                missing.append((cell, tid))
    partial = scope != expected_scope or bool(missing)
    if partial and not args.allow_partial:
        p.error("Incomplete suite; use --allow-partial for development")
    if partial and args.verify_published:
        p.error("A partial run cannot verify the full published report")
    snapshot = {str(path.resolve()): sha(path) for path in sources}
    out.mkdir(parents=True, exist_ok=True)
    all_cells: dict[str, Any] = {}
    paired_rows: list[Record] = []
    gap_rows: list[Record] = []
    slope_rows: list[Record] = []
    common_rows: list[Record] = []
    writer = None
    seen: set[str] = set()
    with (
        gzip.open(DATA / "trial_endpoints.csv.gz", "rt", newline="") as handle,
        gzip.open(out / "trial_endpoints.csv.gz", "wt", newline="") as destination,
    ):
        reader = csv.DictReader(handle)
        for name, group in itertools.groupby(reader, key=lambda r: r["cell"]):
            if name in seen or name not in definitions:
                raise ValueError(f"Noncontiguous or unexpected cell: {name}")
            seen.add(name)
            raw = list(group)
            cell = definitions[name]
            archived_rows, controls = parse_cell(raw, cell, extras)
            if args.run:
                rows = []
                for retained in archived_rows:
                    tid = retained["trial"]
                    if (name, tid) not in scope or (name, tid) in missing:
                        continue
                    path = args.run / name / f"trial-{tid:05d}.json"
                    r = json.loads(path.read_text())
                    for key in (
                        "cell",
                        "trial",
                        "job",
                        "alpha_grid",
                        "seed",
                        "culprit",
                        "n_arms",
                        "p_new",
                        "p_old",
                        "budget",
                        "targets",
                    ):
                        if r[key] != retained[key]:
                            raise ValueError(f"Run/control identity mismatch: {path}/{key}")
                    r.pop("observations", None)
                    rows.append(r)
            else:
                rows = archived_rows
            if not rows:
                continue
            for r in rows:
                for ix, alpha in enumerate(cell["alpha_grid"]):
                    row = endpoint_row(r, alpha, ix, controls)
                    if writer is None:
                        writer = csv.DictWriter(destination, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
            summary = make_cell_summary(
                supplement["cell_configs"][name], cell["jobs"], rows, controls
            )
            all_cells[name] = summary
            for alpha in cell["alpha_grid"]:
                paired_rows.append(
                    {
                        "cell": name,
                        "champion_rule": "champion_singleton",
                        "saved_rule": "saved_bayes",
                        "baseline": "OG",
                        "alpha": alpha,
                        **summary["paired_costs"]["OG"][str(alpha)],
                    }
                )
            if name == "e2_n64_prior":
                gap_rows = prior_bins(rows, controls)
            if name in SLOPE_CELLS:
                small, common = slopes(name, rows, controls)
                slope_rows.append(small)
                common_rows.append(common)
    if seen != set(definitions) or not all_cells:
        raise ValueError("Missing retained cells or no completed selected trials")
    result = {
        "partial": partial,
        "partial_label": "DEVELOPMENT PARTIAL RUN" if partial else None,
        "standard_suite_complete": not partial,
        "selected_cells": len(all_cells),
        "selected_trials": sum(c["n"] for c in all_cells.values()),
        "scope_missing_trial_files": len(missing),
        "cells": all_cells,
        "archived_uniform_extension": {
            "path": "archived_uniform_extension.txt",
            "interpretation": "Retained aggregate OG controls; no trial-level paired observations.",
        },
    }
    write_json(out / "summary.json", result)
    write_csv(out / "paired_costs.csv", paired_rows)
    write_csv(out / "prior_gap_bins.csv", gap_rows)
    write_csv(out / "small_alpha_slopes.csv", slope_rows)
    write_json(
        out / "common-slope-audit.json",
        {
            "status": "computed",
            "partial": partial,
            "definition": "Difference between first stops at alpha=1e-6 and 1e-5, divided by log(10). Common subset is uncensored for the three listed methods at both endpoints.",
            "source_sha256": sha(out / "trial_endpoints.csv.gz"),
            "cells": common_rows,
        },
    )
    shutil.copyfile(DATA / "archived_uniform_extension.txt", out / "archived_uniform_extension.txt")
    lines = [
        "Randomized safe stopping and OG git bayesect",
        f"{len(all_cells)} cells; {result['selected_trials']:,} trials; alpha = 0.05",
        "Mean costs charge the full budget to censored trials. Full threshold grids are in summary.json.",
        "",
    ]
    if partial:
        lines.insert(0, "DEVELOPMENT PARTIAL RUN")
    for name, cell in sorted(all_cells.items()):
        lines.append(f"{name} (n={cell['n']})")
        for method in RULES:
            stat = cell["rules"][method]["0.05"]
            lines.append(
                "  "
                + method
                + f": mean={stat['mean_capped_cost']:.4f}; "
                + ", ".join(f"{k}={v}" for k, v in stat["counts"].items())
            )
        lines.append("")
    (out / "report.txt").write_text("\n".join(lines).rstrip() + "\n")
    checked = verify_published(out) if args.verify_published else None
    if any(sha(Path(path)) != expected for path, expected in snapshot.items()):
        raise RuntimeError("Report inputs changed during aggregation")
    write_json(
        out / "provenance.json",
        {
            "partial": partial,
            "source_and_input_sha256": snapshot,
            "artifact_sha256": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
            "python": platform.python_version(),
            "numpy": np.__version__,
            "baseline_trials_rerun": False,
            "published_values_verified": checked,
        },
    )
    print(
        json.dumps(
            {
                "out": str(out),
                "cells": len(all_cells),
                "trials": result["selected_trials"],
                "partial": partial,
                "published_values_verified": checked,
            }
        )
    )


if __name__ == "__main__":
    main()
