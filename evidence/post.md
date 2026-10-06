# Safe stopping for git bayesect

[git bayesect](https://hauntsaninja.github.io/git_bayesect.html) finds the
commit that made a flaky test flakier, without knowing the failure rates.
It stops when one commit holds 95% of its posterior. So: **when git bayesect
says 95%, is it right 95% of the time?** We simulated about 27,000
bisections to find out, and built a stopping rule that keeps its promise.
Methods and full results are in the
[technical companion](technical-companion.md).

In short:

- **git bayesect's confidence holds up on easy cases but breaks down on hard
  ones.** When flakiness jumps a lot, it is wrong about as often as it
  claims. When the jump is small, it is wrong up to twice as often (9.5% at
  a claimed 5%).
- **When git bayesect's assumptions don't hold, it can be badly wrong while
  sounding sure:** 74% with correlated reruns, 26% with a mistaken prior.
- **Safe stopping** (`--safe`) is wrong at most 5% of the time when the
  assumptions hold, even if CI conditions change during the bisection. It
  uses 18–35% more test runs than git bayesect: 17–28% more where git
  bayesect would have been right, and 1.5–5× as many where it would have
  been wrong.
- **Some violations still defeat it.** Correlated reruns and a broken commit
  fool it nearly as badly as git bayesect. `--check-assumptions` declines to
  answer some of the time when reruns are correlated.

## What "95%" means in git bayesect

git bayesect's confidence is a posterior probability: a promise averaged
over its prior on culprits and failure rates, not about *your* test. For big
jumps in flakiness it holds. For small jumps it overclaims:

![Actual vs claimed error: git bayesect overclaims more as the flakiness jump shrinks; safe stopping stays below the claim](figures/fig1_calibration.png)

With 64 commits, at a claimed 95% it is wrong 4.2% of the time when the
failure rate goes from 10% to 50%, and 9.5% when it goes from 10% to 20%.

The wrong answers are **early stops**: an early streak looked decisive (a
median of 337 test runs, against 999 for the right answers). Keep testing
and it settles on the right commit.

## When the assumptions don't hold

git bayesect assumes that one commit changes the failure rate, and that
every run of a commit fails independently with the same probability. With
64 commits and a 10% → 30% regression, this is how often it was wrong at a
claimed 95% when ordinary wrinkles break that:

| what happens | wrong at a claimed 95% |
|---|---|
| nothing (the assumptions hold) | 5.6% |
| CI gets busier halfway through, for every commit | 6.3% |
| a later commit half-fixes the regression | 10.3% |
| unrelated commits nudge flakiness a little | 20.2% |
| one older commit always fails (a build break) | 51.6% |
| reruns of a commit are correlated | **73.8%** |

- **Correlated reruns are the worst.** git bayesect often re-tests a commit
  back to back, so with a bad runner a burst of repeats looks like
  independent confirmation.
- **A broken commit keeps misleading it:** after 3,000 test runs the answer
  is still wrong a quarter of the time.
- **Priors hurt too.** A weak prior centred on the *true* failure rates
  doubles the error. One expecting a 90% failure rate gives 26%.

## Safe stopping

Safe stopping changes both which commit git bayesect tests next and when it
stops.

For each candidate commit it tracks a running score of evidence against
that commit. Before each test it predicts how likely every commit is to
fail, and picks the commit to test at random: early on usually the one git
bayesect would pick, later mostly the most likely culprit or its neighbour.
The score of a candidate grows when the result fits those predictions better
than the candidate's claim that all commits on each side of it share one
failure rate. If the candidate is the culprit, its score reaches `1/p` with
probability at most `p`. That holds whatever the failure rates, however
often you look, and however poor the predictions are.

So ruling out candidates at 20, and stopping when one is left, is wrong at
most 5% of the time.

The guarantee needs less than git bayesect's model. Suppose the CI runner
gets busier halfway through the bisection. Both failure rates go up, and git
bayesect's model no longer describes the data. But commits on each side of
the culprit still share a failure rate at every moment, and that is all the
guarantee needs.

The price is test runs, but not a flat surcharge. When the assumptions hold
(15 settings, 16–1024 commits, five pairs of failure rates), safe stopping
was wrong, or ruled out every commit, at most 2.9% of the time:
- **where git bayesect would have been right,** it used 1.17–1.28× the test
  runs;
- **where git bayesect would have been wrong,** it used 1.5–5.1×, testing on
  instead of trusting a lucky streak.

A uniform fix is no substitute. Running every git bayesect bisection to 1.5
times as many test runs costs more than safe stopping, yet still leaves 8.3%
wrong at 10% → 20%, where safe stopping is wrong or empty 2.8% of the time.
Raising the confidence works only if you know by how much, which depends on
the unknown failure rates.

Choosing a commit and updating the scores adds well under a millisecond of
computation per test run.

## Safe stopping when the assumptions are wrong

A shared change in CI conditions is covered by the 5% bound. The other
violations are not:

![Outcomes per scenario for git bayesect and the three safe modes](figures/fig3_misspecification.png)

- **CI conditions change for every commit:** 2.0% wrong and 0.8% with every
  commit ruled out, where git bayesect is wrong 6.3% of the time.
- **A partial fix or small per-commit variation:** outside the bound, but
  safe stopping still helps: 4.0% wrong against 10.3% for a partial fix,
  and 11% against 20% for per-commit variation.
- **A broken commit defeats both methods:** safe stopping is wrong 51% of the
  time, git bayesect 52%.
- **Correlated reruns fool both:** safe stopping is wrong 61% of the time,
  against 74%.

`--check-assumptions` adds one more score. It measures how much better the
results fit two broader models than any single change with independent
runs: every commit with its own failure rate, and reruns that depend on the
previous run. If that score reaches `1/(1-confidence)`, the bisection stops
without naming a commit.

With reruns that repeat half the time, it cuts wrong answers from 74% (git
bayesect) and 61% (`--safe`) to 42%, by declining to answer in 42% of
bisections. The check is slower than safe stopping: kept going to 3,000
test runs, it fires in every one of these bisections, but in 146 of the 252
only after an answer has been given. It needs no extra test runs when the
assumptions hold, and it doesn't catch the other violations.

## Using it

Safe stopping is a mode chosen at `start`. It applies to `run`,
`pass`/`fail` and `status`, and the default behavior is unchanged:

```
git bayesect start --old $COMMIT --safe                # stop only when safe
git bayesect start --old $COMMIT --max-range 4         # may answer with up to 4 commits
git bayesect start --old $COMMIT --check-assumptions   # may decline to answer
git bayesect run python -m pytest tests/test_flaky.py
```

Test the commits it checks out: its random choices are part of the
guarantee. Priors can be changed at any time, and a restarted bisection picks
up where it left off.
