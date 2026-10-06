"""Recompute every endpoint count and mean from the published compressed trial table."""

from __future__ import annotations
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RULES = tuple("champion_" + x for x in ["singleton", "range", "check", "posterior"]) + tuple(
    "saved_" + x for x in ["bayes"]
)
STATUSES = ["correct", "wrong", "empty", "check_abstain", "censored"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source = ROOT / "evidence/results/full"
    summary = json.loads((source / "summary.json").read_text())
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    row_count = 0
    with gzip.open(source / "trial_endpoints.csv.gz", "rt", newline="") as f:
        for row in csv.DictReader(f):
            row_count += 1
            alpha = str(float(row["alpha"]))
            for rule in RULES:
                key = (row["cell"], rule, alpha)
                g = groups.setdefault(
                    key, {"n": 0, "counts": dict.fromkeys(STATUSES, 0), "cost": 0}
                )
                status = row[rule + "_status"]
                if rule.startswith("saved_"):
                    cost = int(row[rule + "_cost"])
                else:
                    cost = int(row[rule + "_step"] or row["budget"])
                g["n"] += 1
                g["counts"][status] += 1
                g["cost"] += cost
            if float(alpha) == 0.05 and row["cell"] in ["e1_n64_0.2", "e1_n64_0.3", "e1_n64_0.5"]:
                status = row["saved_bayes_status"]
                if status in ["correct", "wrong"]:
                    pair = pairs.setdefault(
                        (row["cell"], status), {"n": 0, "default": 0, "safe": 0}
                    )
                    pair["n"] += 1
                    pair["default"] += int(row["saved_bayes_cost"])
                    pair["safe"] += int(row["champion_singleton_step"] or row["budget"])
    output = []
    for (cell, rule, alpha), g in sorted(groups.items()):
        expected = summary["cells"][cell]["rules"][rule][alpha]
        assert g["n"] == expected["n"], (cell, rule, alpha, "n")
        assert g["counts"] == expected["counts"], (cell, rule, alpha, "counts")
        mean = g["cost"] / g["n"]
        assert math.isclose(mean, expected["mean_capped_cost"], abs_tol=1e-9), (
            cell,
            rule,
            alpha,
            "mean",
        )
        output.append(
            {
                "cell": cell,
                "rule": rule,
                "alpha": float(alpha),
                "n": g["n"],
                "counts": g["counts"],
                "mean_capped_cost": mean,
            }
        )
    expected_groups = sum(len(c["alphas"]) * len(RULES) for c in summary["cells"].values())
    assert len(groups) == expected_groups
    paired = []
    for (cell, status), g in sorted(pairs.items()):
        paired.append(
            {
                "cell": cell,
                "default_outcome": status,
                "n": g["n"],
                "default_mean": g["default"] / g["n"],
                "safe_mean": g["safe"] / g["n"],
                "ratio_of_means": g["safe"] / g["default"],
            }
        )
    report = {
        "status": "passed",
        "scope": "All published endpoint status counts and capped means",
        "rows": row_count,
        "aggregates": len(groups),
        "paired_costs": paired,
        "tables": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validated {len(groups)} aggregates from {row_count} rows")
    print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
