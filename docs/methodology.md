# Methodology

## Why three layers rather than one

**Detection** (TVLA) answers *does the observable depend on the secret?* It is
cheap, needs no key hypothesis, and is the standard first screen.

**Exploitation** (CPA) answers *can key material actually be recovered, and at
what cost?* Detection alone systematically overstates risk: a statistically
significant dependence is not the same as an attack. Reporting trace counts is
the honest metric.

**Attribution** answers *which instruction is responsible?* Without it, a
positive result is a warning; with it, it is a work item.

## TVLA (Test Vector Leakage Assessment)

A non-specific fixed-vs-random Welch t-test. Two groups of traces are collected:
one where the varying input is held at a fixed value, one where it is drawn at
random. For each sample position the Welch statistic is

```
t = (mean_A - mean_B) / sqrt(var_A/n_A + var_B/n_B)
```

with |t| > 4.5 conventionally taken as detection (roughly 5 sigma, with an
informal allowance for the number of samples tested). The threshold is a
convention, not a law, which is why the achieved maximum is always reported
alongside it — a reader can apply a different threshold without rerunning.

**Degenerate denominators.** If both groups are deterministic, the variance is
zero. Two cases must be distinguished and a naive implementation conflates them:

- equal means, no variance → no channel → t = 0
- different means, no variance → *perfect separation*, the strongest possible
  leak → t = ±∞

Mapping the second case to 0, as `np.nan_to_num` does by default, reports the
worst case as clean. This was a real bug in this codebase; see the README.

**The false-positive control.** Fixed-vs-*fixed* — identical secrets, identical
inputs, differing only in noise realisation — must not fire. A detector that
always fires is indistinguishable from a very sensitive one, so the negative
control is the only thing that gives a negative result meaning. It runs in CI.

## Timing test

A Welch t-test on total cycle counts rather than per-sample activity. This is
the channel available to a remote attacker with no physical access, and it is
reported separately because the countermeasures differ: constant-time
programming addresses this channel, masking addresses the power channel, and
neither addresses both.

## Trace alignment

Data-dependent execution time means traces genuinely differ in length, and that
difference *is* the timing channel, so it must not be silently destroyed.

- `pad` (default) keeps every sample and zero-pads short traces. The tail region
  then flags as leaking — correctly, because presence or absence of activity
  there depends on the secret.
- `truncate` cuts to the shortest trace, isolating power leakage from timing
  leakage. Use this to ask *"does it still leak if I fix the timing?"*

For CPA, neither is adequate. Sample index *n* corresponds to different
instructions in different traces once control flow or cache behaviour diverges.
`extract_windows` therefore walks each trace's own PC record and aligns on the
instruction of interest. On real hardware this alignment must be recovered
heuristically (static alignment, elastic alignment, correlation peaks), and
imperfect alignment is one of the main reasons published attacks need more
traces than theory predicts. In a simulator it is exact.

## CPA (Correlation Power Analysis)

For each of 256 key candidates, predict the leakage of an intermediate value and
correlate the prediction against every sample:

```
rho(k, s) = Pearson( hypothesis(p_i, k) , trace_i[s] )   over traces i
```

The candidate with the largest peak is the guess. Two details matter.

**Match the hypothesis to the physics.** Where the previous bus state is known —
and for a fixed instruction sequence it is, being the address of the preceding
access — the correct model is Hamming *distance*, not Hamming *weight*. Against
an HD-leaking bus, an HW hypothesis returns candidates one or two Hamming steps
from the true key. The attack is not failing randomly; it is succeeding against
the wrong model.

**The complement ambiguity.** Under a Hamming-weight model,
HW(p ⊕ ¬k) = 8 − HW(p ⊕ k). A candidate and its bitwise complement therefore
produce correlations of identical magnitude and opposite sign, and ranking on
|ρ| leaves an irreducible two-way tie for every key byte. This is a property of
the model, not a numerical accident. Signed ranking resolves it under the
standard assumption that switching activity increases with Hamming weight;
`complement_tie` reports when the ambiguity was present.

**Judging recovery.** By rank of the true key and by margin over the runner-up,
not by peak correlation alone. A high correlation with a near-identical
runner-up is not a recovery. `traces_to_recover` reports the smallest trace
count at which the true key first ranks first *and stays first* for the
remainder of the sweep, which is stricter than first-correct-guess and less
prone to reporting a lucky early hit.
