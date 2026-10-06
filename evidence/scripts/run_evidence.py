"""Run or resume production RIPr simulations from the frozen 60-cell design."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from simulation import ROOT, trial


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.partial")
    tmp.write_text(json.dumps(value, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def run_one(task: tuple[str, dict[str, Any], list[float], str]) -> str:
    cell, job, alphas, output = task
    result = trial(job, tuple(alphas))
    result.update(cell=cell, job=job, alpha_grid=alphas)
    atomic_json(Path(output), result)
    return output


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cells", nargs="*")
    p.add_argument("--trial-ids", nargs="*", type=int)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    design_path = ROOT / "evidence/results/full/design.json.gz"
    design = json.loads(gzip.decompress(design_path.read_bytes()))
    if args.cells and set(args.cells) - {c["name"] for c in design}:
        p.error("Unknown cell name")
    tasks = [
        (c["name"], j, c["alpha_grid"], str(args.out / c["name"] / f"trial-{j['trial']:05d}.json"))
        for c in design
        if not args.cells or c["name"] in args.cells
        for j in c["jobs"]
        if args.trial_ids is None or j["trial"] in args.trial_ids
    ]
    if not tasks or args.workers < 1:
        p.error("No selected trials, or nonpositive worker count")
    if args.dry_run:
        print(f"{len(tasks)} trials in {len({t[0] for t in tasks})} cells")
        return
    sources = [
        ROOT / "git_bayesect.py",
        Path(__file__),
        Path(__file__).with_name("simulation.py"),
        design_path,
    ]
    identity = {
        "sources": {
            str(s.relative_to(ROOT)): hashlib.sha256(s.read_bytes()).hexdigest() for s in sources
        },
        "scope": [(cell, job["trial"]) for cell, job, _, _ in tasks],
    }
    identity = json.loads(json.dumps(identity))
    manifest = args.out / "manifest.json"
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text()) != identity:
            p.error("Output exists or sources/scope changed; choose a fresh output directory")
    else:
        if args.out.exists() and any(args.out.iterdir()):
            p.error("Output is nonempty without an identity manifest")
        atomic_json(manifest, identity)
    remaining = []
    for task in tasks:
        path = Path(task[3])
        if path.exists():
            completed = json.loads(path.read_text())
            if completed["job"] != task[1] or completed["alpha_grid"] != task[2]:
                raise ValueError(f"Resume mismatch: {path}")
        else:
            remaining.append(task)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, output in enumerate(pool.map(run_one, remaining), 1):
            print(f"{i}/{len(remaining)} {output}", flush=True)

    cells: dict[str, Any] = {}
    for cell, job, alphas, output in tasks:
        result = json.loads(Path(output).read_text())
        group = cells.setdefault(cell, {"n": 0, "rules": {}})
        group["n"] += 1
        for rule, endpoints in result["stops"].items():
            by_alpha = group["rules"].setdefault("champion_" + rule, {})
            for alpha, endpoint in zip(alphas, endpoints):
                record = by_alpha.setdefault(str(alpha), {"n": 0, "counts": {}, "cost": 0})
                record["n"] += 1
                status = endpoint["status"]
                record["counts"][status] = record["counts"].get(status, 0) + 1
                record["cost"] += endpoint["step"] or job["budget"]
    for group in cells.values():
        for by_alpha in group["rules"].values():
            for record in by_alpha.values():
                record["mean_capped_cost"] = record.pop("cost") / record["n"]
    atomic_json(
        args.out / "summary.json",
        {
            "complete_selected_scope": True,
            "standard_suite_complete": len(tasks) == sum(len(c["jobs"]) for c in design),
            "selected_trials": len(tasks),
            "cells": cells,
        },
    )


if __name__ == "__main__":
    main()
