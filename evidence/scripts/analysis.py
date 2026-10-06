"""Statistics shared by the saved-endpoint and fresh-run evidence reports."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import math
from typing import Any
import numpy as np

PATH_RULES = ("singleton", "range", "check", "posterior")
RULES = tuple("champion_" + r for r in PATH_RULES) + ("saved_bayes",)
OUTCOME_STATUSES = ("correct", "wrong", "empty", "check_abstain", "censored")


def get_baseline(
    mapping: dict[tuple[int, float], dict[str, Any]], trial_id: int, rule: str, alpha: float
) -> dict[str, Any]:
    if rule != "bayes":
        raise ValueError("Only OG controls are included")
    return mapping[trial_id, alpha]


def status_of_baseline(row: dict[str, Any]) -> str:
    return str(row["status"])


def cost_of(record: dict[str, Any], budget: int) -> int:
    return int(budget if record["step"] is None else record["step"])


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, mid - half), min(1.0, mid + half)]


def outcome_summary(records: list[dict[str, Any]], budgets: list[int]) -> dict[str, Any]:
    n = len(records)
    counts = {s: sum(r["status"] == s for r in records) for s in OUTCOME_STATUSES}
    answered = counts["correct"] + counts["wrong"]
    costs = np.asarray([cost_of(r, b) for r, b in zip(records, budgets)], dtype=float)
    wrong = counts["wrong"]
    known_failure = wrong + counts["empty"]
    known_rejection = [
        r.get("true_rejected") for r in records if r.get("true_rejected") is not None
    ]
    target_alive = [r.get("targets_alive") for r in records if "targets_alive" in r]
    result = {
        "n": n,
        "counts": counts,
        "unconditional_wrong": wrong / n if n else None,
        "unconditional_wrong_wilson95": wilson(wrong, n),
        "known_wrong_or_empty_count": known_failure,
        "unconditional_wrong_or_empty": (known_failure / n if n else None),
        "unconditional_wrong_or_empty_wilson95": wilson(known_failure, n),
        "conditional_answer_error": wrong / answered if answered else None,
        "conditional_answer_error_wilson95": wilson(wrong, answered),
        "answered": answered,
        "mean_capped_cost": float(costs.mean()) if n else None,
        "median_capped_cost": float(np.quantile(costs, 0.5)) if n else None,
        "p80_capped_cost": float(np.quantile(costs, 0.8)) if n else None,
        "p90_capped_cost": float(np.quantile(costs, 0.9)) if n else None,
        "p95_capped_cost": float(np.quantile(costs, 0.95)) if n else None,
    }
    result["known_candidate_set_miscoverage_n"] = len(known_rejection)
    result["candidate_set_miscoverage_n"] = sum(bool(x) for x in known_rejection)
    result["candidate_set_miscoverage_fraction_known"] = (
        result["candidate_set_miscoverage_n"] / len(known_rejection) if known_rejection else None
    )
    result["any_target_retained_n"] = sum(bool(x) for x in target_alive)
    result["any_target_retained_fraction"] = (
        result["any_target_retained_n"] / len(target_alive) if target_alive else None
    )
    return result


def cost_by_status(records: list[dict[str, Any]], budgets: list[int]) -> dict[str, Any]:
    result = {}
    for status in OUTCOME_STATUSES:
        vals = [cost_of(r, b) for r, b in zip(records, budgets) if r["status"] == status]
        result[status] = {
            "n": len(vals),
            "mean_capped_cost": float(np.mean(vals)) if vals else None,
            "median_capped_cost": float(np.median(vals)) if vals else None,
        }
    return result


def paired_summary(diffs: Sequence[float], strata: list[int], stratum_label: str) -> dict[str, Any]:
    """Within-design-stratum paired mean and standard error, or descriptive only."""
    n = len(diffs)
    if not n:
        return {"n": 0, "mean_cost_difference": None, "interval_method": "none"}
    vals = np.asarray(diffs, dtype=float)
    result: dict[str, Any] = {"n": n, "mean_cost_difference": float(vals.mean())}
    groups: dict[int, list[float]] = defaultdict(list)
    for x, stratum in zip(diffs, strata):
        groups[int(stratum)].append(float(x))
    if any(len(g) < 2 for g in groups.values()):
        result.update(
            interval_method="descriptive_single_rep_per_boundary",
            standard_error=None,
            pointwise95=None,
        )
        return result
    variance = sum((len(g) / n) ** 2 * float(np.var(g, ddof=1)) / len(g) for g in groups.values())
    se = math.sqrt(variance)
    result.update(
        interval_method=f"{stratum_label}_stratified_paired_normal_95",
        standard_error=se,
        pointwise95=[
            float(vals.mean() - 1.959963984540054 * se),
            float(vals.mean() + 1.959963984540054 * se),
        ],
    )
    return result


def settling_summary(
    entries: list[dict[str, Any]], tolerated_unresolved: float = 0.05
) -> dict[str, Any]:
    """Finite-horizon multiplier bound; unresolved paths remain right-censored."""
    n = len(entries)
    stopped = [x for x in entries if x.get("stop_step") is not None]
    ratios = [
        float(x["settle_step"]) / float(x["stop_step"])
        for x in stopped
        if x.get("settle_step") is not None
    ]
    unresolved = len(stopped) - len(ratios)
    horizon_bounds = [
        float(x["observed_horizon"]) / float(x["stop_step"])
        for x in stopped
        if x.get("settle_step") is None and x.get("observed_horizon") is not None
    ]
    rank = int(math.ceil((1 - tolerated_unresolved) * len(stopped))) if stopped else 0
    c_min = max(1.0, float(sorted(ratios)[rank - 1])) if rank and len(ratios) >= rank else None
    return {
        "n_trials": n,
        "n_without_posterior_first_stop": n - len(stopped),
        "n_with_posterior_first_stop": len(stopped),
        "settled_within_observed_horizon": len(ratios),
        "unresolved_or_right_censored": unresolved,
        "unresolved_fraction_among_first_stops": unresolved / len(stopped) if stopped else None,
        "minimum_observed_multiplier_for_95pct_settled": c_min,
        "largest_right_censor_lower_bound_multiplier": (
            max(horizon_bounds) if horizon_bounds else None
        ),
        "finite_horizon_interpretation": "A null means the observed trajectory does not establish a finite multiplier; unresolved paths are right-censored at their recorded horizon.",
    }


def make_cell_summary(
    run: dict[str, Any],
    jobs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    base_map: dict[tuple[int, float], dict[str, Any]],
) -> dict[str, Any]:
    job_map = {int(j["trial"]): j for j in jobs}
    alphas = rows[0]["alpha_grid"] if rows else []
    cell_out: dict[str, Any] = {
        "config": {
            k: run[k] for k in ("n", "gap", "scenario", "reps", "budget", "full_budget", "prior")
        },
        "n": len(rows),
        "alphas": alphas,
        "rules": {},
        "terminal": {},
        "paired_costs": {},
        "og_status_split_at_0.05": {},
        "hindsight": {},
        "threshold_for_5pct_error": {},
        "settling_at_0.05": {},
        "settling_multiplier_by_alpha": {},
        "range_at_0.05": {},
        "cost_by_returned_status_at_0.05": {},
    }
    if not rows:
        return cell_out
    budgets = [int(r["budget"]) for r in rows]

    # Keep original saved rules separate from rules evaluated on the champion path.
    for summary_key in RULES:
        is_champion = summary_key.startswith("champion_")
        raw_rule = summary_key.removeprefix("champion_").removeprefix("saved_")
        cell_out["rules"][summary_key] = {}
        for ix, alpha in enumerate(alphas):
            if is_champion:
                records = [r["stops"][raw_rule][ix] for r in rows]
            else:
                saved = [get_baseline(base_map, int(r["trial"]), raw_rule, alpha) for r in rows]
                records = [
                    dict(
                        step=(None if status_of_baseline(x) == "censored" else int(x["stop_step"])),
                        status=status_of_baseline(x),
                    )
                    for x in saved
                ]
            cell_out["rules"][summary_key][str(alpha)] = outcome_summary(records, budgets)

    if 0.05 in alphas:
        ix = alphas.index(0.05)
        for rule in PATH_RULES:
            per_target = {}
            for target in sorted({b for rr in rows for b in rr["targets"]}):
                total = [r for r in rows if target in r["targets"]]
                reject_n = sum(target in r["stops"][rule][ix]["targets_rejected"] for r in total)
                per_target[str(target)] = {
                    "n": len(total),
                    "rejected": reject_n,
                    "rejection_fraction": reject_n / len(total) if total else None,
                }
            cell_out["rules"][f"champion_{rule}"]["per_target_at_0.05"] = per_target

    # Matched costs use identical seed, boundary, targets, rates, N and budget.
    paired: dict[str, Any] = defaultdict(dict)
    for ix, alpha in enumerate(alphas):
        for champ_rule, old_rule, label in (("singleton", "bayes", "OG"),):
            diffs, strata = [], []
            for r in rows:
                job = job_map[int(r["trial"])]
                b = get_baseline(base_map, int(r["trial"]), old_rule, alpha)
                champ_cost = cost_of(r["stops"][champ_rule][ix], int(r["budget"]))
                old_cost = int(
                    job["budget"] if status_of_baseline(b) == "censored" else b["stop_step"]
                )
                diffs.append(champ_cost - old_cost)
                strata.append(int(r["culprit"]))
            stratum = (
                "original_design_position"
                if run["scenario"] in ("partial_fix", "two_step")
                else "culprit"
            )
            paired[label][str(alpha)] = paired_summary(diffs, strata, stratum)
    cell_out["paired_costs"] = dict(paired)

    if 0.05 in alphas:
        ix = alphas.index(0.05)
        for rule in ("singleton", "posterior"):
            cell_out["cost_by_returned_status_at_0.05"][f"champion_{rule}"] = cost_by_status(
                [r["stops"][rule][ix] for r in rows], budgets
            )
        champ_costs = []
        og_costs = []
        for r in rows:
            b = get_baseline(base_map, int(r["trial"]), "bayes", 0.05)
            champ_costs.append(cost_of(r["stops"]["singleton"][ix], int(r["budget"])))
            og_costs.append(
                int(r["budget"] if status_of_baseline(b) == "censored" else b["stop_step"])
            )
        split = {}
        for old_status in ("correct", "wrong", "censored"):
            ids = []
            for i, r in enumerate(rows):
                b = get_baseline(base_map, int(r["trial"]), "bayes", 0.05)
                if status_of_baseline(b) == old_status:
                    ids.append(i)
            cc = np.asarray([champ_costs[i] for i in ids], float)
            oo = np.asarray([og_costs[i] for i in ids], float)
            split[old_status] = {
                "n": len(ids),
                "mean_champion_cost": float(cc.mean()) if len(cc) else None,
                "mean_og_cost": float(oo.mean()) if len(oo) else None,
                "mean_champion_to_og_cost_ratio": float(np.mean(cc / oo)) if len(cc) else None,
                "ratio_of_mean_costs": float(cc.mean() / oo.mean()) if len(cc) else None,
            }
        cell_out["og_status_split_at_0.05"] = split

        champ_settle = []
        og_settle = []
        for r in rows:
            cp = r["stops"]["posterior"][ix]
            champ_settle.append(
                {
                    "stop_step": None if cp["step"] is None else cp["step"],
                    "settle_step": r.get("posterior_settle_step"),
                    "observed_horizon": r["steps_run"],
                    "status": cp["status"],
                }
            )
            b = get_baseline(base_map, int(r["trial"]), "bayes", 0.05)
            sv = b.get("settle_step", "")
            settle = int(sv) if sv.strip().isdigit() and int(sv) > 0 else None
            og_settle.append(
                {
                    "stop_step": (
                        None if status_of_baseline(b) == "censored" else int(b["stop_step"])
                    ),
                    "settle_step": settle,
                    "observed_horizon": int(b["steps_run"]),
                    "status": status_of_baseline(b),
                }
            )
        cell_out["settling_at_0.05"] = {
            "champion_same_path_posterior": settling_summary(champ_settle),
            "saved_original_bayes": settling_summary(og_settle),
            "champion_first_stop_correct_vs_wrong": {
                status: settling_summary([x for x in champ_settle if x["status"] == status])
                for status in ("correct", "wrong")
            },
            "saved_original_first_stop_correct_vs_wrong": {
                status: settling_summary([x for x in og_settle if x["status"] == status])
                for status in ("correct", "wrong")
            },
        }
        for alpha in alphas:
            a_ix = alphas.index(alpha)
            c_entries = []
            o_entries = []
            for r in rows:
                cp = r["stops"]["posterior"][a_ix]
                c_entries.append(
                    {
                        "stop_step": None if cp["step"] is None else cp["step"],
                        "settle_step": r.get("posterior_settle_step"),
                        "observed_horizon": r["steps_run"],
                        "status": cp["status"],
                    }
                )
                b = get_baseline(base_map, int(r["trial"]), "bayes", alpha)
                sv = b.get("settle_step", "")
                settled = int(sv) if sv.strip().isdigit() and int(sv) > 0 else None
                o_entries.append(
                    {
                        "stop_step": (
                            None if status_of_baseline(b) == "censored" else int(b["stop_step"])
                        ),
                        "settle_step": settled,
                        "observed_horizon": int(b["steps_run"]),
                        "status": status_of_baseline(b),
                    }
                )
            cell_out["settling_multiplier_by_alpha"][str(alpha)] = {
                "champion_same_path_posterior": settling_summary(c_entries),
                "saved_original_bayes": settling_summary(o_entries),
            }

        widths = [
            r["stops"]["range"][ix]["answer"]
            for r in rows
            if isinstance(r["stops"]["range"][ix]["answer"], list)
        ]
        range_cost = [cost_of(r["stops"]["range"][ix], int(r["budget"])) for r in rows]
        singleton_cost = [cost_of(r["stops"]["singleton"][ix], int(r["budget"])) for r in rows]
        cell_out["range_at_0.05"] = {
            "answered_ranges": len(widths),
            "mean_width": float(np.mean([hi - lo + 1 for lo, hi in widths])) if widths else None,
            "mean_capped_cost": float(np.mean(range_cost)),
            "mean_capped_cost_saving_vs_singleton": float(
                np.mean(np.asarray(singleton_cost) - range_cost)
            ),
        }

        target_error = cell_out["rules"]["champion_singleton"]["0.05"]["unconditional_wrong"]
        champion_mean = cell_out["rules"]["champion_singleton"]["0.05"]["mean_capped_cost"]
        qualifying = []
        for alpha in alphas:
            entry = cell_out["rules"]["saved_bayes"][str(alpha)]
            error = entry["unconditional_wrong"]
            if error is not None and target_error is not None and error <= target_error:
                qualifying.append((entry["mean_capped_cost"], alpha, error))
        best = min(qualifying) if qualifying else None
        cell_out["hindsight"]["saved_bayes_cheapest_meeting_champion_error"] = {
            "target_champion_singleton_error_at_0.05": target_error,
            "alpha": best[1] if best else None,
            "saved_og_error": best[2] if best else None,
            "saved_og_mean_capped_cost": best[0] if best else None,
            "champion_singleton_mean_capped_cost": champion_mean,
            "saved_og_to_champion_cost_ratio": (
                best[0] / champion_mean if best and champion_mean else None
            ),
            "no_tested_og_alpha_meets_target": best is None,
            "descriptive_hindsight_only": True,
        }

    for rule in RULES:
        candidate = []
        candidate_raw = []
        candidate_lower = []
        candidate_upper = []
        for alpha in alphas:
            entry = cell_out["rules"][rule][str(alpha)]
            if entry["unconditional_wrong"] is not None and entry["unconditional_wrong"] <= 0.05:
                candidate.append(alpha)
            if (
                entry["unconditional_wrong_or_empty"] is not None
                and entry["unconditional_wrong_or_empty"] <= 0.05
            ):
                candidate_raw.append(alpha)
            if (
                entry["unconditional_wrong_or_empty"] is not None
                and entry["unconditional_wrong_or_empty"] <= 0.05
            ):
                candidate_lower.append(alpha)
            if (
                entry["unconditional_wrong_or_empty"] is not None
                and entry["unconditional_wrong_or_empty"] <= 0.05
            ):
                candidate_upper.append(alpha)
        cell_out["threshold_for_5pct_error"][rule] = {
            "loosest_grid_alpha_with_unconditional_error_at_most_0.05": (
                max(candidate) if candidate else None
            ),
            "none_on_grid": not candidate,
            "loosest_grid_alpha_with_wrong_or_empty_at_most_0.05": (
                max(candidate_raw) if candidate_raw else None
            ),
            "none_on_grid_for_exact_wrong_or_empty": not candidate_raw,
            "loosest_alpha_lower_bound_at_most_0.05": (
                max(candidate_lower) if candidate_lower else None
            ),
            "loosest_alpha_upper_bound_at_most_0.05": (
                max(candidate_upper) if candidate_upper else None
            ),
            "threshold_interpretation": "Descriptive thresholds on the tested grid, not a calibration guarantee.",
        }

    observed_budget = [
        bool(job_map[int(r["trial"])]["full_budget"] or r["steps_run"] == r["budget"]) for r in rows
    ]
    valid_empty = [len(r["final"]["alive_at_0.05"]) == 0 for r in rows]
    saved_final = [get_baseline(base_map, int(r["trial"]), "bayes", 0.05) for r in rows]
    g_rejected = [r["final"]["safe_max_log_misfit"] >= -math.log(0.05) for r in rows]
    checked = [r["stops"]["check"][alphas.index(0.05)] for r in rows] if 0.05 in alphas else []
    cell_out["terminal"] = {
        "n_at_budget": sum(observed_budget),
        "n_stopped_before_budget": len(rows) - sum(observed_budget),
        "champion_riPr_empty_survivor_set_count_at_recorded_horizon": sum(valid_empty),
        "champion_riPr_empty_survivor_set_fraction_at_recorded_horizon": sum(valid_empty)
        / len(rows),
        "n_targets_retained_in_0.05_set": sum(
            bool(r["final"]["target_intersects_set_at_0.05"]) for r in rows
        ),
        "n_targets_all_rejected_at_0.05": sum(
            not r["final"]["target_intersects_set_at_0.05"] for r in rows
        ),
        "mean_survivor_count_at_recorded_horizon": float(
            np.mean([len(r["final"]["alive_at_0.05"]) for r in rows])
        ),
        "posterior_map_acceptable_at_recorded_horizon": sum(
            r["final"]["posterior_map"] in r["targets"] for r in rows
        ),
        "assumption_check_rejected_at_recorded_horizon": sum(
            r["final"]["safe_max_log_misfit"] >= -math.log(0.05) for r in rows
        ),
        "champion_check_answer_with_later_G_crossing_at_recorded_horizon": {
            "n_final_G_crossed": sum(g_rejected),
            "correct_answers_before_later_G_crossing": sum(
                g and rec["status"] == "correct" for g, rec in zip(g_rejected, checked)
            ),
            "wrong_answers_before_later_G_crossing": sum(
                g and rec["status"] == "wrong" for g, rec in zip(g_rejected, checked)
            ),
            "check_abstentions_at_first_stop": sum(
                rec["status"] == "check_abstain" for rec in checked
            ),
            "interpretation": "A path can continue after its .05 checked rule fires for other thresholds; these counts show terminal G rejection after a checked answer, not a rule that answered despite an earlier G crossing.",
        },
        "saved_original_at_recorded_horizon": {
            "n_trials": len(saved_final),
            "final_map_acceptable_count": sum(x["final_map_ok"] == "1" for x in saved_final),
        },
        "interpretation": "Champion and saved CSV diagnostics are recorded-horizon endpoints. They are fixed-budget only for original full_budget jobs or paths that reached the cap; empty survivor sets differ from rejecting every target.",
    }
    return cell_out
