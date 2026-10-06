# Safe stopping for git bayesect: technical companion

This document backs every claim in the [post](post.md): the model and
notation, the stopping rules and their validity, the simulation design, full
results, a map from claims to evidence, and how to reproduce everything.

All numbers come from a study of 60 cells and 27,287 simulated bisections.
The summaries are in [results/full/](results/full/): `report.txt`,
`summary.json`, `paired_costs.csv`, `prior_gap_bins.csv`, `timing.json`, and
every trial's endpoints in `trial_endpoints.csv.gz`. git bayesect's own
results ("the posterior stop" below) are saved runs from an earlier study
with the same jobs: seeds, rates, culprits and budgets. They were reused, not
rerun. Safe stopping's runs replay exactly through the code on branch `safe`
(§7, §9).

## 1. Model and notation

We use git bayesect's own notation. There are N commits from the *old*
commit to the *new* one, inclusive, indexed by distance back from the new
one: i = 0 (new), 1, …, N−1 (old). Hypothesis **B** ∈ {0, …, N−1} says

```
commit i ~ Bernoulli(p_new)  if i <= B      (the new behaviour: B is the culprit)
commit i ~ Bernoulli(p_old)  if i >  B      (the old behaviour)
```

with both rates unknown and no assumption about which is larger. B = N−1
means even the old commit already has the new behaviour. A rate pair is
written **p_old → p_new**. In fixed-rate simulations the true culprit is
uniform over 0..N−2.

For each B, git bayesect computes the marginal likelihood under Beta priors
on the two rates (Jeffreys Beta(½, ½) by default):

```
m_B = B(y_new + ½, n_new − y_new + ½) / B(½, ½)  ×  B(y_old + ½, n_old − y_old + ½) / B(½, ½)
```

Here (n_new, y_new) are the runs and failures on commits i ≤ B, and (n_old,
y_old) those on i > B. With commit prior π_B (uniform by default), the
posterior is post_B = π_B m_B / Z, where Z = Σ_B π_B m_B. The posterior stop
chooses commits with git bayesect's expected-entropy-reduction `select()`.
**Safe stopping changes both the choice of commit and the stopping rule.**

## 2. Stopping rules

| name here | CLI | rule |
|---|---|---|
| posterior stop | default | stop when max_B post_B ≥ confidence; answer the argmax (ties toward the middle, as upstream) |
| stricter threshold / keep testing | — | the posterior stop at a lower α′, or continued for c× as many runs |
| safe stop | `--safe` | randomized choice of commit; stop when exactly one hypothesis survives (§2.1) |
| safe range stop | `--max-range w` | the same; stop when the survivors span ≤ w adjacent commits; answer the range |
| assumption check | `--check-assumptions` | safe stop, but abstain as soon as the check in §2.2 fires |

α = 1 − confidence throughout.

### 2.1 The safe stop

**Predictions.** Before each test, take each hypothesis's posterior-mean
rates and average them over the posterior. This is git bayesect's predictive
probability that commit i fails:

```
a_B = (y_new + ½) / (n_new + 1)        b_B = (y_old + ½) / (n_old + 1)
q_i = Σ_{B ≥ i} post_B a_B  +  Σ_{B < i} post_B b_B
```

Custom Beta priors replace the halves, here and in the posterior.

**Choice of commit.** Let j be the posterior mode and s the commit
`select()` would test. Safe stopping draws the next commit I from

```
w = post_j · (λ δ_j + (1 − λ) δ_{j+1})  +  (1 − post_j) · δ_s
```

λ is the split between j and j+1 that makes one test most informative
about whether their rates differ, at the rates a_j and b_j:

```
h(p) = −p log p − (1 − p) log(1 − p)
r* = expit(−[h(a_j) − h(b_j)] / [a_j − b_j]),    λ = clip((r* − b_j) / (a_j − b_j), 0, 1)
```

with λ = ½ when a_j ≈ b_j, and the two endpoints when j is the
constant-rate hypothesis N−1. Early on, w is mostly git bayesect's own
choice; once one culprit dominates, it is mostly that culprit and its
neighbour. On test numbers that are perfect squares, w is instead a single
commit, cycling from the ends inward, so every commit keeps being tested
occasionally.

**Evidence.** For each hypothesis B, average the predictions on each side of
B, weighting by the same probabilities w:

```
r_{B,new} = Σ_{i ≤ B} w_i q_i / Σ_{i ≤ B} w_i        r_{B,old} = Σ_{i > B} w_i q_i / Σ_{i > B} w_i
e_B = f(Y; q_I) / f(Y; r_B(I)),    f(y; p) = p^y (1 − p)^(1 − y)
E_B(t) = Π_{s ≤ t} e_B(s)
```

Here r_B(I) is the average for the side of B that contains the tested
commit I. The factor bets that the tested commit behaves as predicted rather
than as its side's average: commits that hypothesis B says are alike, the
predictions may say are different. Reject B once max_{s ≤ t} E_B(s) ≥ 1/α.
The surviving set is {B : max_{s ≤ t} E_B(s) < 1/α}:
- stop when it has one element;
- with `--max-range w`, stop when its span is ≤ w;
- report "no single commit explains these results" if it is empty.

**Validity.** Fix the true culprit B\*. Suppose that, given everything
observed so far, every commit i ≤ B\* fails with one probability p_new,t and
every commit i > B\* with another, p_old,t. The two rates may change from
test to test and depend on earlier results; the predictions may be wrong.
For one side S of B\*, with W_S = Σ_{i∈S} w_i and r_S its averaged
prediction,

```
Σ_{i∈S} w_i [p_S q_i / r_S + (1 − p_S)(1 − q_i) / (1 − r_S)]  =  p_S W_S + (1 − p_S) W_S  =  W_S.
```

Summing over the two sides gives E[e_{B\*} | past] = 1, so E_{B\*}(t) is a
nonnegative martingale with mean 1. By Ville's inequality,
P(sup_t E_{B\*}(t) ≥ 1/α) ≤ α. The method answers wrongly, or reports an
empty set, only if it has rejected B\*; a range answer is wrong only if it
excludes B\*. So:

> P(wrong answer or empty set) ≤ α, for every culprit, any N, and any
> failure rates that are shared by the commits on each side of the culprit
> at each moment, however often you check.

No Bonferroni correction over hypotheses is needed. The cancellation above
averages over the random choice of commit, so the guarantee needs the
commit to be drawn from w, not picked after looking. A forced round has a
single commit on its side of every B, so its factor is 1.

**Missing results.** A test that is cancelled after its commit is drawn
(an interrupted run, a result recorded for a different commit, or `undo`)
cannot simply be forgotten: keeping only the draws you like biases the
evidence. Such a draw gets the factor min(e_B(0), e_B(1)). That is no larger
than the factor for either outcome, so E_B stays a supermartingale and the
bound holds.

**Compute.** Cumulative sums give every side average at once, so a step
costs O(N):

```python
left_mass, left_moment = cumsum(w), cumsum(w * q)
right_mass, right_moment = suffix_sum(w), suffix_sum(w * q)    # sums over i > B
r = where(arange(N) < I, right_moment / right_mass, left_moment / left_mass)
log_E += log_f(Y, q[I]) - log_f(Y, r)
alive = maximum(max_log_E, log_E) < log(1 / alpha)
```

The production sampler puts mass on at most three commits per test, so
saving each announced choice takes O(1) space.

### 2.2 The assumption check

Safe stopping also eliminates hypotheses when the model is wrong, but the
signal that *no* hypothesis fits usually arrives after it has already
stopped. The assumption check adds

```
G_t = Z_check(t) / max_B L̂_B(t)
```

where L̂_B is the likelihood at B's best-fitting rates. G_t is an e-process
for "some single change with independent runs and fixed rates holds",
because max_B L̂_B is at least the true likelihood. The rule abstains as soon
as max_{s ≤ t} G_s ≥ 1/α. Under that model, P(wrong answer) ≤ α and
P(empty set or abstention) ≤ 2α. Z_check averages two O(N) Jeffreys-mixture
alternatives:

- **own rate per commit:** Π_i m(y_i, n_i). This targets broken commits and
  per-commit variation.
- **reruns follow a Markov chain:** separate transition rates on each side
  of an unknown culprit, mixed over B. It pays for only a handful of
  parameters. This targets correlated reruns.

The check's model is narrower than safe stopping's: it does not allow the
rates to change over time. Evaluating G only at some steps is still valid.
The simulations evaluate it every 4th run; the CLI evaluates it at every
run.

### 2.3 What the guarantee covers

Write μ_i for commit i's actual failure probability given the past, and
μ̄_S for its w-weighted average over side S. Then exactly

```
E[e_B | past] = 1 + Σ_{S ∈ {new, old}} [Σ_{i∈S} w_i (μ_i − μ̄_S)(q_i − r_S)] / [r_S (1 − r_S)].
```

- **A shift shared by every commit on a side cancels.** If the CI runner
  gets busier, both rates change, git bayesect's model is wrong, and the
  bound still holds. The `drift` scenarios test exactly this (§4.7).
- **Differences between commits on the same side do not cancel** when the
  predictions line up with them. A broken commit, per-commit variation, and
  reruns that repeat a commit's last result all make two commits on one side
  fail at different rates.

## 3. Simulation design

- **Environment.** Each commit has its own seeded random stream, so the k-th
  run of a commit gives the same outcome whichever rule is used. Safe
  stopping tests different commits from git bayesect, so the two see
  different sequences. Every trial has an independent seed.
- **Trials.** Environments × every culprit position: 32 × 15 = 480 trials
  at N=16, 8 × 63 = 504 at N=64, 2 × 255 = 510 at N=256, 1 × 1023 at
  N=1024. Some cells use fewer (315 for 0.1 → 0.15; 252 per misspecified
  cell at N=64). One context cell draws the rates and the culprit from git
  bayesect's own prior (512 trials).
- **One trajectory, every threshold.** Safe stopping's choice of commit
  doesn't depend on α, so each trial records when each rule would stop at
  every α on a grid from 0.5 to 10⁻⁶ (or 10⁻³), and what it would answer.
  The main misspecification cells (N=64, 0.1 → 0.3) always run to their
  3,000-run budget; other trials end once every rule has stopped.
- **Outcomes.** An answer is *right* if it is an acceptable answer (a range
  must contain one), *wrong* otherwise. An *empty* survivor set,
  *abstained* and *didn't stop* (no stop within the budget) are reported
  separately. Mean run counts charge censored trials their full budget.

**Misspecification scenarios** (all keep commit N−1 old-ish and commit 0
new-ish):

| scenario | what happens in the repo | acceptable answer |
|---|---|---|
| `drift` | CI load changes mid-bisection: every commit's log-odds + Δ from run 150 (Δ = 0.5; sweep 0.25 / 1.0) | the culprit |
| `partial_fix` | regression p_old → p_new at B₂; a newer commit B₁ half-fixes it | B₂ |
| `two_step` | two regressions: p_old → midpoint → p_new | either culprit |
| `ramp` | the rate climbs over w commits instead of jumping (w = 4; sweep 2 / 8) | any commit in the ramp |
| `jitter` | per-commit log-odds noise N(0, σ²) (σ = 0.25; sweep 0.1 / 0.5) | the culprit |
| `bursty` | each rerun of a commit repeats its previous outcome with probability ρ (0.5; sweep 0.25 / 0.75) | the culprit |
| `broken` | one commit older than the culprit always fails | the culprit |

Two further cells answer with ranges of up to 8 commits: a 4-commit ramp and
two regressions (504 trials each, 4,000-run budget).

## 4. Results

### 4.1 Calibration of the posterior stop (model holds)

| rates p_old → p_new | N | claimed 80%: wrong | claimed 95%: wrong [95% Wilson CI] | safe stop at α = 5%: wrong or empty |
|---|---|---|---|---|
| 0.1 → 0.5 | 16 / 64 / 256 / 1024 | 21.5 / 19.0 / 20.0 / 20.7% | 3.3 / 4.2 / 4.9 / 4.9% (each ±~1.5) | 1.5 / 1.2 / 1.0 / 1.9% |
| 0.1 → 0.3 | 16 / 64 / 256 / 1024 | 20.2 / 23.8 / 25.9 / 25.5% | 5.6 / 5.6 / 5.9 / 6.5% (N=1024: [5.2, 8.2]) | 0.8 / 1.6 / 2.0 / 1.8% |
| 0.1 → 0.2 | 16 / 64 / 256 | 29.2 / 29.6 / 28.4% | 5.6 / **9.5** [7.3, 12.4] / 7.8 [5.8, 10.5]% | 1.2 / 2.8 / 0.4% |
| 0.1 → 0.15 | 64 | 30.5% | 7.3% [4.9, 10.7] | 2.9% |
| 0.02 → 0.1 | 16 / 64 / 256 | 17.7 / 22.0 / 21.8% | 3.8 / 5.6 / 6.1% | 1.2 / 1.2 / 2.7% |

- The posterior stop is roughly calibrated for large gaps and overclaims up
  to about 2× (at 95%) and 1.5× (at 80%) for small ones.
- N-dependence is weak and inconsistent: a slight upward drift at
  0.1 → 0.3 and 0.02 → 0.1, but none at 0.1 → 0.2.
- Safe stopping stays below its 5% bound in every cell: 0.4–2.9% wrong or
  empty. Empty sets are rare (0–3 per cell).
- The post's Figure 1 plots the N=64 rows across thresholds.

**Averaged over the prior.** With rates and culprit drawn from git
bayesect's own prior (N=64, 512 trials), it was wrong 3.4% of the time at a
claimed 95%, among trials that stopped (16 of 474): calibrated on average,
as a posterior must be. Binned by |p_new − p_old|, it was wrong 14% of the
time below 0.05 (3 of 22) and 1.7% above 0.4 (4 of 237). The average hides
the hard instances. Safe stopping made 4 wrong answers and 1 empty set in
512; 42 trials didn't stop within 5,000 runs, including all 8 whose culprit
was the constant-rate hypothesis.

### 4.2 Wrong answers are early stops

At N=64, 0.1 → 0.2, claimed 95%:

| trials | n | median stop | mean stop | mean run at which the answer settles on the truth |
|---|---|---|---|---|
| right at stop | 455 | 999 | 1083 | 858 |
| wrong at stop | 48 | 337 | 518 | 1350 |

In all but 4 of 8,337 well-specified trials, the posterior stop's answer has
settled on the true culprit by the end of the run. Safe stopping's own
wrong answers are early too: at the same rates its 13 wrong answers average
699 runs, against 1,346 for its 490 right ones.

### 4.3 The posterior stop when the model is wrong

At N=64, 0.1 → 0.3, claimed 95% (252 trials, run to 3,000):

| scenario | wrong at stop | mean runs | answer acceptable at 3,000 runs |
|---|---|---|---|
| model holds (504 trials) | 5.6% | 336 | 99.8% (at the end of the run) |
| drift | 6.3% | 331 | 100% |
| ramp | 6.3% | 864 | 99.6% |
| partial_fix | 10.3% | 451 | 96.0% |
| two_step | 14.3% | 954 | 96.4% |
| jitter σ=0.25 | 20.2% | 357 | 94.0% |
| bursty ρ=0.5 | 73.8% | 192 | 96.0% |
| broken | 51.6% | 244 | 75.0% |

Its wrong stops under `bursty` come after a median of 108 runs.

Across gaps, N and severity (wrong at a claimed 95%):

| scenario | 0.1 → 0.5 (N=64) | 0.1 → 0.3 (N=64 / 256) | 0.1 → 0.2 (N=64) | severity sweep at 0.1 → 0.3, N=64 |
|---|---|---|---|---|
| drift | 4.4% | 6.3 / 7.6% | 9.5% | Δ 0.25 / 0.5 / 1.0: 4.8 / 6.3 / 6.2% |
| partial_fix | 10.9% | 10.3 / 7.8% | 10.3% | — |
| two_step | 12.7% | 14.3 / 15.1% | 13.5% | — |
| ramp | 3.2% | 6.3 / 7.1% | 9.5% | w 2 / 4 / 8: 6.9 / 6.3 / 5.8% |
| jitter | 6.5% | 20.2 / 21.0% | 40.5% | σ 0.1 / 0.25 / 0.5: 6.5 / 20.2 / 55.6% |
| bursty | 56.3% | 73.8 / 80.8% | 79.8% | ρ 0.25 / 0.5 / 0.75: 34.5 / 73.8 / 91.1% |
| broken | 34.7% | 51.6 / 25.1% | 67.9% | — |

### 4.4 Priors

At N=64, 0.1 → 0.3 (true rates p_new = 0.3, p_old = 0.1). The same priors go
into safe stopping's predictions and its choice of commit. Rate priors are
written Beta(new) / Beta(old); "strength" is α + β.

| priors | posterior stop: wrong at 95% [CI] / runs | safe stop: wrong / empty / runs |
|---|---|---|
| Jeffreys (default) | 5.6% [3.9, 7.9] / 336 | 1.4% / 0.2% / 419 |
| centred on the truth, strength 1: Beta(0.3, 0.7) / Beta(0.1, 0.9) | **11.1%** [8.7, 14.2] / 327 | 1.4% / 0.4% / 458 |
| centred on the truth, strength 4: Beta(1.2, 2.8) / Beta(0.4, 3.6) | 6.7% [4.9, 9.3] / 308 | 0.6% / 0.2% / 422 |
| centred on the truth, strength 20: Beta(6, 14) / Beta(2, 18) | 3.2% [2.0, 5.1] / 250 | 1.0% / 0% / 326 |
| expects ~60% / ~10%: Beta(1.2, 0.8) / Beta(0.2, 1.8) | 11.9% [9.4, 15.0] / 298 | 1.8% / 0.6% / 440 |
| expects ~90% / ~5%: Beta(0.9, 0.1) / Beta(0.05, 0.95) | **26.0%** [22.4, 30.0] / 264 | 1.8% / 0.4% / 485 |
| commit prior ×10 on a random wrong commit | 5.2% [3.5, 7.5] / 335 | 1.6% / 0.2% / 433 |

[Issue #24](https://github.com/hauntsaninja/git_bayesect/issues/24) upstream
observes that user-supplied rate priors can hurt calibration, even when they
are centred on the true rates. It sets two bars for exposing priors to
users: a prior should not make the reported confidence wrong, and it should
not make the bisection converge more slowly than the default Jeffreys prior.

- **The issue's observation holds.** A weak prior centred on the true rates
  doubles the posterior stop's error. A confident mistaken prior makes it
  five times the claim, and the posterior stop also stops *earlier* under
  it.
- **Safe stopping meets the first bar for any prior:** its 5% bound holds
  whatever the prior, which only shapes the predictions. In these settings
  it was wrong or empty at most 2.4% of the time.
- **On the second bar (no slower than Jeffreys):**
  - accurate priors make safe stopping faster (326 runs at strength 20,
    against 419);
  - mistaken priors cost it at most 16%.
- **Commit priors:** a misleading ×10 weight was harmless for both.

### 4.5 Stricter thresholds and "keep testing"

The largest α′ at which the posterior stop's realized error is ≤ 5% (the
grid goes down to 10⁻³ in misspecified cells):
- model holds: 0.016–0.051, depending on gap and N;
- drift and ramps: 0.016–0.051; two regressions: 0.010–0.016; partial fix:
  0.005–0.01;
- bursty ρ=0.25: 0.001;
- none ≥ 10⁻³ reaches 5% for: jitter σ=0.25 at gaps ≤ 0.3 or σ=0.5, bursty
  ρ ≥ 0.5, a broken commit, or a partial fix at 0.1 → 0.5.

With hindsight, the cheapest tested threshold at which the posterior stop is
wrong no more often than safe stopping lies between α′ = 0.005 and 0.016,
depending on the cell. There it costs 3–11% fewer runs than safe stopping,
but the right α′ depends on the unknown rates.

The multiplier c that leaves ≤ 5% of bisections unsettled when each keeps
testing until c times the posterior stop's test runs:
- model holds: 1.0–3.5;
- at N=64, 0.1 → 0.3: drift 1.8; ramp 5.2; partial fix 5.8; two
  regressions 11; bursty ρ=0.5: 78;
- more than the runs show under jitter and with a broken commit: too many
  answers (15 and 62 of 252) had still not settled at 3,000 runs.

### 4.6 Cost of the safe stop (model holds)

Mean runs to stop at α = 0.05 (claimed 95% for the posterior stop):

| rates | N | posterior stop | safe stop | safe range ≤ 4 (saving) | safe / posterior |
|---|---|---|---|---|---|
| 0.1 → 0.5 | 16 / 64 / 256 / 1024 | 86 / 103 / 125 / 144 | 108 / 130 / 149 / 170 | 106 / 128 / 148 / 168 (0.5–2.0%) | 1.26 / 1.26 / 1.20 / 1.18 |
| 0.1 → 0.3 | 16 / 64 / 256 / 1024 | 258 / 336 / 391 / 454 | 348 / 419 / 500 / 562 | 343 / 417 / 498 / 559 (0.3–1.5%) | 1.35 / 1.25 / 1.28 / 1.24 |
| 0.1 → 0.2 | 16 / 64 / 256 | 855 / 1051 / 1246 | 1126 / 1330 / 1598 | 1104 / 1314 / 1589 (0.6–1.9%) | 1.32 / 1.27 / 1.28 |
| 0.1 → 0.15 | 64 | 3876 | 4966 | 4948 (0.4%) | 1.28 |
| 0.02 → 0.1 | 16 / 64 / 256 | 540 / 671 / 821 | 679 / 835 / 1001 | 670 / 828 / 990 (0.9–1.4%) | 1.26 / 1.24 / 1.22 |

- **The extra runs go where the posterior stop was wrong.** At N=64,
  0.1 → 0.2, safe stopping averages 1,447 runs on the 48 trials where the
  posterior stop was wrong, against 1,316 on the 455 where it was right.
- **The cost is mostly an offset.** At N=64 the ratio falls from 1.25–1.27×
  at α = 5% to 1.11–1.14× at α = 10⁻⁶, and the cost per extra nat of
  evidence is close to the oracle rate (§4.10).
- **Part of the cost is over-coverage.** Safe stopping is wrong or empty
  0.4–2.9% of the time at a 5% bound. With hindsight, the posterior stop
  reaches the same error 3–11% more cheaply (§4.5).
- **Ranges save little here:** 0.3–2.0%.

**Where the extra runs go.** These are safe stopping's mean runs divided by
the posterior stop's, split by whether the posterior stop's answer was right
(α = 0.05):

| rates | N | on trials where it was right | on trials where it was wrong |
|---|---|---|---|
| 0.1 → 0.5 | 16 / 64 / 256 / 1024 | 1.23 / 1.23 / 1.17 / 1.18 | 3.13 / 2.36 / 2.28 / 1.69 |
| 0.1 → 0.3 | 16 / 64 / 256 / 1024 | 1.28 / 1.24 / 1.22 / 1.19 | 3.55 / 1.45 / 3.07 / 2.50 |
| 0.1 → 0.2 | 16 / 64 / 256 | 1.28 / 1.21 / 1.22 | 5.07 / 2.80 / 2.67 |
| 0.1 → 0.15 | 64 | 1.22 | 4.08 |
| 0.02 → 0.1 | 16 / 64 / 256 | 1.22 / 1.21 / 1.18 | 2.96 / 1.96 / 2.33 |

Across all 15 cells that is 1.17–1.28× where the posterior stop is right and
1.45–5.07× where it is wrong.

**A uniform extension is not equivalent.** Stop as git bayesect does, keep
testing until c times that stopping time, and answer with the most likely
commit then (N=64, same jobs; the archived control in
[archived_uniform_extension.txt](results/full/archived_uniform_extension.txt)).
Share answered wrongly:

| rates | at the stop | 1.5× the runs | 2× the runs | 3× the runs | safe stop: wrong or empty (average cost) |
|---|---|---|---|---|---|
| 0.1 → 0.2 | 9.5% | 8.3% | 6.7% | 4.8% | 2.8% (1.27×) |
| 0.1 → 0.3 | 5.6% | 4.2% | 3.0% | 2.2% | 1.6% (1.25×) |
| 0.1 → 0.5 | 4.2% | 3.4% | 2.4% | 1.8% | 1.2% (1.26×) |

At 1.5× the runs, more than safe stopping's average cost, the uniform
extension is wrong 2.6–3.0 times as often as safe stopping is wrong or empty.

![Mean test runs against actual error as the stopping threshold varies; git bayesect vs safe stopping, for three gaps](figures/fig2_cost_of_safety.png)

*Figure 2.* Mean runs against realized error as the threshold sweeps from 80%
to 99.9999% confidence, on the same jobs (N=64). Dots mark 95%.

Per-trial stopping times at N=64, 0.1 → 0.2, α = 0.05:

| | median | 80th pct | 90th pct | 95th pct |
|---|---|---|---|---|
| posterior stop | 956 | 1386 | 1760 | 2099 |
| safe stop | 1229 | 1853 | 2204 | 2564 |
| safe range stop | 1217 | 1841 | 2148 | 2519 |

### 4.7 Safe stopping when the model is wrong

At N=64, 0.1 → 0.3, α = 5% (Figure 3 in the post shows these rows except
`two_step` and `ramp`; Figure 4 below shows the ramps):

| scenario | posterior stop wrong | safe wrong / empty | safe range wrong | assumption check wrong / abstains | safe runs (didn't stop) | survivor set empty at 3,000 |
|---|---|---|---|---|---|---|
| model holds | 5.6% | 1.4% / 0.2% | 1.4% | 1.4% / 0% | 419 (0 of 504) | — |
| drift | 6.3% | 2.0% / 0.8% | 2.0% | 2.0% / 0% | 411 (0) | 2.8% |
| partial_fix | 10.3% | 4.0% / 0% | 4.0% | 4.0% / 0% | 541 (1) | 2.0% |
| two_step | 14.3% | 2.0% / 0% | 1.6% | 2.0% / 0% | 1500 (21); range 1486 (20) | 1.6% |
| jitter σ=0.25 | 20.2% | 11.1% / 0.8% | 9.9% | 11.1% / 0% | 472 (1) | 6.0% |
| bursty ρ=0.5 | 73.8% | 61.1% / 2.4% | 59.9% | 41.7% / 42.1% | 313 (1); check 168 | 64.7% |
| broken | 51.6% | 51.2% / 0% | 51.2% | 51.2% / 0.4% | 289 (0) | 1.2% |
| ramp | 6.3% | 1.6% / 0.4% | 2.0% | 1.6% / 0% | 2338 (149); range 2261 (133) | 6.3% |

**Cost when the model is violated** (same cells). This is the ratio of mean
runs, safe stop to posterior stop, overall and split by whether the
posterior stop's answer was right:

| scenario | safe / posterior runs | on trials where it was right / wrong |
|---|---|---|
| model holds | 1.25× | 1.24 / 1.45× |
| drift | 1.24× | 1.20 / 1.98× |
| partial_fix | 1.20× | 1.19 / 1.23× |
| jitter σ=0.25 | 1.32× | 1.21 / 1.86× |
| bursty ρ=0.25 | 1.45× | 1.15 / 2.74× |
| two_step | ≥ 1.57× | 1.53 / 2.73× |
| bursty ρ=0.5 | ≥ 1.63× | 0.96 / 2.00× |
| broken | 1.18× | 1.11 / 1.47× |
| ramp | ≥ 2.71× | 2.67 / 7.54× |

"≥" marks cells where some trials never stop and are charged the full
3,000-run budget (`bursty ρ=0.25` has 504 trials and a 4,000-run budget).
- **Shared drift is covered by the bound** (§2.3), and the cost pattern of
  the model-holds case survives.
- **A ramp is the expensive case.** When the failure rate climbs over four
  commits instead of one, any of them counts as right, and the posterior
  stop answers in 864 runs on average. Adjacent commits in the ramp differ
  by only 0.05 in failure rate. Safe stopping keeps testing the most likely
  culprit against its neighbour until every other ramp commit is ruled out,
  and 149 of 252 bisections (59%) are still going at 3,000 runs.
  `--max-range 4` barely helps: 133 are still going.
- **Under a broken commit the split narrows.** The spurious second step
  fools safe stopping as it fools the posterior stop.

Across the rest of the grid:
- **Shared drift:** 1.2–2.0% wrong across gaps, N and severities, where the
  posterior stop is at 4.4–9.5%.
- **Partial fix:** 2.0–4.0% wrong, against 7.8–10.9%.
- **Per-commit variation grows with its size:** σ=0.1: 2.2% against 6.5%;
  σ=0.25: 3.0–33% across gaps (11.2% at N=256), against 6.5–40.5%;
  σ=0.5: 44% against 56%. The range answer is usually a little lower (e.g.
  jitter at N=256: 10.4% vs 11.2%).
- **Correlated reruns:** at ρ=0.5 safe stopping is wrong 38–76% of the time
  across gaps and N, against 56–81%; at ρ=0.25, 16.7% against 34.5%; at
  ρ=0.75, 87% against 91%.
- **Broken commit:** about as wrong as the posterior stop (25–64% against
  25–68%).
- **Ambiguous truths are the main cost of safety.**
  - A 2-commit ramp leaves 3 of 504 unresolved at 4,000 runs (2 with ranges
    ≤ 4), at 977 runs on average against 514.
  - A 4-commit ramp with ranges ≤ 8 leaves 196 of 504 unresolved at 4,000
    runs, against 237 for single answers. At N=256 it leaves 249 of 510.
  - An 8-commit ramp leaves 441 of 504 unresolved at 4,000 runs.
  - Two regressions leave 21 of 252 unresolved at N=64, 0.1 → 0.3, and 39
    at 0.1 → 0.2.
  - The posterior stop's apparent accuracy on ramps (3.2–9.5% wrong) is
    helped by several answers counting as right.

![Outcomes on ramps of 2, 4 and 8 commits for git bayesect and the three safe modes](figures/fig4_ramp.png)

*Figure 4.* Ramps of 2, 4 and 8 commits at N=64, 0.1 → 0.3, α = 5%. Any
commit in the ramp counts as right. The wider the ramp, the more bisections
are still going at the budget.

### 4.8 The assumption check

- **Model holds:** identical run counts to safe stopping in every cell; no
  abstentions in the 15 fixed-rate cells.
- **Correlated reruns:**
  - ρ=0.5 at N=64, 0.1 → 0.3: wrong 41.7% (safe 61.1%, posterior 73.8%).
    It abstains in 42% of trials and stops after 168 runs on average, where
    safe stopping needs 313. The answers it does give are unreliable: 105 of
    its 146 answers were wrong.
  - The check is slower than safe stopping. Continued to 3,000 runs, it
    fires on all 252 paths, but those 105 wrong answers come first.
  - Across gaps and N: 32–42% wrong, 20–62% abstentions.
  - ρ=0.25: 13.5% wrong, 19% abstentions. ρ=0.75: 63% wrong (safe 87%,
    posterior 91%), 28% abstentions.
- **Broken commit:** 3 abstentions in 252 at 0.1 → 0.2, 1 at 0.1 → 0.3, none
  otherwise. In the 0.1 → 0.3 cell the check crosses its threshold on 120
  paths by 3,000 runs: once before an answer and 119 times after a wrong
  one.
- **Jitter, drift, partial fix, two regressions, ramps:** no effect (at most
  one abstention per cell).

It adds no runs when the model holds and keeps compute O(N), but it converts
wrong answers into abstentions substantially only for correlated reruns. So
it belongs as an opt-in aimed at flaky infrastructure, not as a general
guard.

### 4.9 Compute

Mean wall-clock per step on one core (`evidence/scripts/timing.py`: the
statistical update only, 100 warmup and 300 measured steps, three repeats;
no Git commands or state files):

| N | git bayesect step | safe stop, extra | assumption check, extra (every run) |
|---|---|---|---|
| 64 | 587 µs | +380 µs (+65%) | +746 µs |
| 256 | 695 µs | +414 µs (+60%) | +992 µs |
| 1024 | 1,069 µs | +613 µs (+57%) | +1,907 µs |
| 4096 | 2,154 µs | +865 µs (+40%) | +5,296 µs |

The added cost is under 1 ms per step for safe stopping, and about 5 ms
with the check at N=4096: negligible unless a single test run takes only
milliseconds.

### 4.10 Commit selection comes within about 10% of the oracle rate in these settings

Once the culprit and the rates are known, the cheapest way to gather
evidence is to test the culprit and its neighbour in the proportions of
§2.1. Its cost is 1/Γ* runs per nat of evidence, where Γ* is that pair's
information rate. These are the measured extra runs per nat between α = 10⁻⁵
and 10⁻⁶ at N=64, on trials that stopped at both thresholds for every rule:

| rates | oracle 1/Γ* | posterior stop | safe stop |
|---|---|---|---|
| 0.1 → 0.5 | 9.8 | 9.6 | 10.4 |
| 0.1 → 0.3 | 30.7 | 29.5 | 33.1 |
| 0.1 → 0.2 | 100.1 | 95.6 | 104.3 |

Safe stopping is 4–8% above the oracle rate, and the posterior stop a few
percent below it. A finite-threshold slope below the asymptotic rate is not
a contradiction. This suggests, but does not prove, that a different choice
of commit would gain little at small α, and that the cost of safety lies
mainly in the offset.

## 5. Claims and evidence

| claim | where | evidence |
|---|---|---|
| C1. The posterior stop is mildly overconfident in its own model, more for small gaps | post: *What 95% means*; §4.1 | 15 cells, N 16–1024, 5 rate pairs |
| C2. It is calibrated on average over its own prior | §4.1 | prior-drawn cell (512 trials) |
| C3. Its wrong answers are early stops that settle with more testing | post: *What 95% means*; §4.2 | all well-specified cells |
| C4. Realistic violations inflate its errors; correlated reruns are worst; a broken commit leaves it wrong through thousands of runs | post: *When the assumptions don't hold*; §4.3 | 7 scenarios × 3 gaps, N=256, severity sweeps |
| C5. User rate priors can double its error or worse, even weak correct ones; they don't break safe stopping | post: *When the assumptions don't hold*; §4.4 | 7 prior settings |
| C6. Safe stopping is wrong or empty at most α whenever the commits on each side of the culprit share a failure rate at each moment, including under shared drift | post: *Safe stopping*; §2.1, §2.3, §4.1, §4.7 | proof; 15 cells; drift cells |
| C7. It costs 1.18–1.35× the posterior stop's runs: 1.17–1.28× where the posterior stop is right, 1.45–5.07× where it is wrong; a uniform extension at 1.5× is still 2.6–3.0× as often wrong | post: *Safe stopping*; §4.6 | 15 cells, uniform extension at N=64 |
| C8. Stricter thresholds need tuning that depends on the unknown failure rates | post: *Safe stopping*; §4.5 | all cells |
| C9. Its compute overhead is under 1 ms per step | post: *Safe stopping*; §4.9 | timing |
| C10. It helps under mild violations, and is defeated by a broken commit or correlated reruns | post: *Safe stopping when the assumptions are wrong*; §4.7 | misspecification grid |
| C11. The assumption check converts some wrong answers to abstentions under correlated reruns, often after safe stopping has already answered | post: *Safe stopping when the assumptions are wrong*; §4.8 | misspecification grid, correlation sweep |
| C12. A gradual ramp stalls safe stopping: most bisections over a 4- or 8-commit ramp are still going at the budget, and ranges barely help | §4.7, Figure 4 | ramp cells, widths 2 / 4 / 8, N=256, ranges ≤ 8 |
| C13. Commit selection comes close to the oracle rate in the tested settings, so the cost appears to lie in the offset | §4.10 | slopes |

## 6. Reproducing

The [evidence README](README.md) has the setup and every command. From the
repository root, with the pinned dependencies in `evidence/requirements.txt`:

```
python evidence/scripts/report.py --out evidence/runs/report --verify-published   # minutes: every table, from the saved endpoints
python evidence/scripts/figures.py --summary evidence/runs/report/summary.json --out evidence/runs/figures
python evidence/scripts/run_evidence.py --out evidence/runs/run --workers 4       # hours: all 60 cells again
python evidence/scripts/replay.py --output evidence/runs/replay.json              # minutes: 63 saved paths through the production code
```

`report.py` recomputes every statistic in §4 from the 459,272 saved
trial/threshold rows and, with `--verify-published`, checks them against
the files in `results/full/`. `run_evidence.py` reruns safe stopping on the
frozen design, and `report.py --run` pairs the fresh runs with the saved
posterior-stop results. git bayesect's runs are not repeated.

## 7. Implementation

The proposed change is branch `safe`, three commits on top of upstream main
(`ba9dd48`):
- `b7340c6` adds randomized safe stopping, the range answer and the
  assumption check;
- `c3693ae` removes an unused accessor and obsolete state compatibility;
- `8668869` tidies formatting and documentation.

Replaying the 63 saved paths (104,054 observations) through the final
commit reproduces every chosen commit, outcome, final score and stopping
point ([replay-final-code.json](results/full/replay-final-code.json)). This
`evidence/` directory sits on a separate branch, `safe-evidence`, and changes
nothing outside `evidence/`.

- **Library:** `SafeExperiment` (one announced random choice: its commits,
  probabilities and predictions) and `SafeStopping` (the scores, their
  running maxima and the assumption check) sit next to `Bisector`. The
  predictions come from the Bisector's own posterior, so there is no second
  model.
- **CLI:**
  - `start` takes `--safe`, `--max-range N` and `--check-assumptions`. The
    latter two imply `--safe`.
  - The setting is saved in the state file (version 2 → 4; version 2
    bisections resume) and honoured by `run`, `pass`/`fail`, `status` and
    `log`. Default behaviour is unchanged.
  - Each random choice is saved before the commit is checked out, so
    restarting or checking out again reuses it.
  - Changing priors affects only later predictions; earlier bets keep their
    saved predictions.
  - A result recorded for a different commit, an interrupted test and
    `undo` charge the pending choice the missing-result factor (§2.1).
  - `run` checks whether it is already safe to stop before running the
    command, so resuming a finished bisection records nothing more.
  - `log` also prints a short script that restores the saved choices and
    the pending checkout.
- **Tests:** 11 new tests next to the 4 existing ones; mypy passes.
  - the evidence factors, including cancelled draws;
  - ruling out what the data contradict;
  - the assumption check;
  - a statistical validity check;
  - option parsing;
  - `run` stopping on its own, and not running again once stopped;
  - the CLI round trip;
  - resuming, prior changes and `log` restoration;
  - results for a different commit, interrupted tests, and upstream state.
- **`scripts/calibration.py`** gains `safe=`, `max_range=` and
  `check_assumptions=`, so it can make the comparison in §4.1 and §4.6 with
  its own simulation. In this study, at 0.1 → 0.2 with 64 commits, the
  posterior stop is right 90.3% of the time in 1,051 runs on average, and
  safe stopping 97.2% in 1,330 (2.8% wrong or empty).

## 8. Limitations

- The bound is a statement about a single change whose two sides each share
  a failure rate at each moment. The misspecification results come from
  scenarios chosen to be plausible, not from every way a test can misbehave.
- The bound holds only for commits drawn as safe stopping chooses them. Its
  run counts are specific to that choice.
- The assumption check's model is narrower than safe stopping's: it assumes
  fixed rates as well.
- git bayesect's results are saved runs from an earlier software
  environment. Their per-cell provenance files were unavailable; the
  original result files and global provenance are hashed instead.
- Wilson intervals treat each cell's trials as independent and identically
  distributed; the designs fix culprit positions, so they are approximate.
- A few of the hardest trials hit the run budget. They are reported as
  "didn't stop", never as errors.

## 9. Provenance

- **Code** (`evidence/scripts/`):
  - `simulation.py`: the scenarios and one trial, driving git bayesect's
    `Bisector`, `SafeExperiment` and `SafeStopping` directly;
  - `run_evidence.py`: the frozen 60-cell design ([design.json.gz](results/full/design.json.gz));
  - `report.py` and `analysis.py`: every table, `summary.json` and the CSVs;
  - `tables.py`: an independent recount of the endpoint table;
  - `replay.py`, `timing.py`: the replay and timing checks;
  - `uniform_extension.py`: the archived "keep testing" control;
  - `figures.py`: the figures.
- **Audits:** the trajectory audit
  ([trajectory-audit.json](results/full/trajectory-audit.json)) checked all
  27,287 paths and 36,527,735 observations, and replayed 180 complete
  trajectories; the largest error in a factor's expected value was
  4.4×10⁻¹⁶. [curation.json](results/full/curation.json) records the source
  hashes and transformations behind the published files, and
  [package-manifest.json](results/full/package-manifest.json) hashes every
  published file.
- **Environment:** Python 3.12.8, NumPy 2.2.3, SciPy 1.17.1 and Matplotlib
  3.10.8, as pinned in `evidence/requirements.txt`.
