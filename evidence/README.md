# Safe stopping evidence

Start with the [post](post.md), then the [technical companion](technical-companion.md)
for the construction, assumptions and full results. This branch contains the
implementation, the frozen experiment design, git bayesect's saved runs, and
all scripts and inputs needed for the commands below. No other branch is
needed.

## Setup

Run these commands from the repository root. Python 3.12 and the pinned evidence
dependencies match the tested environment (Python 3.12.8, NumPy 2.2.3,
SciPy 1.17.1, Matplotlib 3.10.8):

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r evidence/requirements.txt
export MPLCONFIGDIR=/tmp/git-bayesect-matplotlib
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

The examples write under `evidence/runs/`, which git ignores; the scripts
create it as needed. It is an output location, not a prerequisite.
Use a fresh report/output directory each time. The simulation runner supports
resuming an existing run as described below.

## Rebuild the published results without simulations

```sh
python evidence/scripts/report.py \
  --out evidence/runs/report --verify-published
python evidence/scripts/figures.py \
  --summary evidence/runs/report/summary.json \
  --out evidence/runs/figures
```

This recomputes all 60 cells and 27,287 trials from the 459,272 retained
trial/threshold rows, including confidence intervals, cost quantiles, paired
costs, the split by whether git bayesect was right, prior-gap bins, settling diagnostics, range
results, and both small-alpha slope analyses. `--verify-published` checks the
recomputed statistics against the accepted JSON/CSV results and text report.
The figures read the newly generated summary. Allow a few minutes for analysis;
no bisections are simulated.

The report writes:

| File | Contents |
|---|---|
| `summary.json`, `report.txt` | Full statistics and the readable 95% confidence summary |
| `trial_endpoints.csv.gz` | All method/threshold endpoints, including git bayesect's saved runs |
| `paired_costs.csv` | Paired sample differences with design-stratified intervals |
| `prior_gap_bins.csv` | Prior-drawn rate gaps; the constant-rate sentinel is separate |
| `small_alpha_slopes.csv` | Safe stopping's small-alpha slopes on its uncensored pairs |
| `common-slope-audit.json` | Slopes on the common uncensored subset for all three displayed methods |
| `archived_uniform_extension.txt` | git bayesect's archived “test longer” control |
| `provenance.json` | Input/source/output hashes, versions, scope and verification count |

The “test longer” aggregate is copied, rather than reconstructed from the
endpoint table: it came from a separate experiment. Its reproduction script is
included below. Acceptance and curation files under `results/full/` record
historical checks; new verification runs write their own outputs.

For a smaller independent check of all endpoint counts and means:

```sh
python evidence/scripts/tables.py --output evidence/runs/endpoint-check.json
```

## Run the full safe-stopping suite again

The frozen design includes every seed, rate pair, boundary, budget, prior,
misspecification parameter and threshold grid. First inspect the scope:

```sh
python evidence/scripts/run_evidence.py --out evidence/runs/run --dry-run
# Expected: 27287 trials in 60 cells
```

Then simulate and analyze:

```sh
python evidence/scripts/run_evidence.py --out evidence/runs/run --workers 4
python evidence/scripts/report.py \
  --run evidence/runs/run --out evidence/runs/fresh-report
python evidence/scripts/figures.py \
  --summary evidence/runs/fresh-report/summary.json \
  --out evidence/runs/fresh-figures
```

This reruns safe stopping's choice of commits and evaluates the singleton, range,
assumption-check and same-path posterior stops at every threshold. The report
pairs the fresh endpoints with **git bayesect's saved runs** by frozen job
identity; git bayesect is not rerun.
`run_evidence.py` writes individual trajectories and a basic progress summary;
`report.py` produces the full analysis. Add `--verify-published` to the report
command to check agreement with the accepted numerical results as well.
Floating-point/library differences can affect adaptive paths, so keep the
pinned environment when checking exact reproduction.

The complete simulation is an hours-scale job and writes several GB of raw
trajectories. Adjust `--workers` for the machine. To resume after interruption,
use the same output, selected cells/trials and source files:

```sh
python evidence/scripts/run_evidence.py \
  --out evidence/runs/run --workers 4 --resume
```

Each completed trial is written atomically. Resume checks code/design hashes
and job metadata; changing sources or scope requires a fresh output directory.
Full reports reject incomplete runs by default.

## Small end-to-end check

This runs one trial, then the same reporting and plotting pipeline:

```sh
python evidence/scripts/run_evidence.py \
  --out evidence/runs/smoke-run \
  --cells e1_n64_0.3 --trial-ids 0 --workers 1
python evidence/scripts/report.py \
  --run evidence/runs/smoke-run \
  --out evidence/runs/smoke-report --allow-partial
python evidence/scripts/figures.py \
  --summary evidence/runs/smoke-report/summary.json \
  --out evidence/runs/smoke-figures
```

Partial reports and figures are labeled accordingly. Missing figure panels
stay blank. To select cells, use names from `results/full/summary.json`;
for example `e1_n64_0.2`, `e1_n64_0.3`, `e2_n64_prior`, or
`e3_n64_0.3_drift`. `--allow-partial` also permits analysis of the completed
trials in an interrupted run.

## Replay, timing and the separate "keep testing" experiment

```sh
python evidence/scripts/replay.py --output evidence/runs/replay.json
python evidence/scripts/timing.py --output evidence/runs/timing.json
```

Replay checks 63 saved paths (104,054 observations): one per cell plus
wrong-answer, empty-set and constant-rate examples. It verifies the chosen
commits, regenerated outcomes, final scores and their running maxima, and the
first stopping points. Timing measures the complete statistical update at N=64, 256, 1024
and 4096, excluding Git/state I/O. Both usually take minutes; timing varies by
machine and load.

Only the following optional command reruns the separate git bayesect control
that keeps testing for 1, 1.5, 2 or 3 times its first 95% posterior stop:

```sh
python evidence/scripts/uniform_extension.py \
  --out evidence/runs/og-extension --workers 4
```

It uses the frozen N=64 0.1→0.2, 0.1→0.3 and 0.1→0.5 seeds, with 504 trials
per pair and a 20,000-query cap. It writes `uniform_extension.txt` and
`trials.json`; compare the former with
`results/full/archived_uniform_extension.txt`. Trials that reach the cap without
an answer remain in the denominator. Use `--dry-run` to inspect the scope or
`--trial-ids 0` for a small check. The main reproduction commands reuse the saved
aggregate and do not require this experiment.

## Files in this branch

- `results/full/` contains accepted results, the frozen `design.json.gz`, saved
  endpoints, audits and timing. `analysis-inputs.json.gz` adds per-trial final
  safe-stopping diagnostics and git bayesect's settling data, which the
  endpoint table lacks. `replay-final-code.json` replays the saved paths through
  the final `safe` commit.
- `fixtures/` contains the 63 compressed replay trajectories.
- `scripts/` contains the simulation, analysis, figure, replay, timing and
  "keep testing" code. All imports resolve within this checkout or the dependencies
  in `requirements.txt`.
- `figures/` contains the figures in PNG and SVG form: two in the post, two in
  the technical companion.

[Curation provenance](results/full/curation.json) records original source hashes
and the selection of retained inputs.
[The package manifest](results/full/package-manifest.json) hashes the published
files. The implementation is also available on `safe`; this `safe-evidence`
branch includes it.
