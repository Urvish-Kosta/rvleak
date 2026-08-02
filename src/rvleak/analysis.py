"""Leakage detection, key recovery, and attribution.

Three layers, in increasing order of strength of claim:

  detect   -- TVLA (Welch fixed-vs-random t-test) says *whether* a secret
              influences the observable. Cheap, no key hypothesis needed,
              standard practice (Goodwill et al., "A testing methodology for
              side-channel resistance validation", NIST NIAT 2011).
  exploit  -- CPA says whether the leak is strong enough to actually recover
              key material, and how many traces that takes. Detection without
              exploitation overstates risk; exploitation is the honest metric.
  attribute-- maps leaking cycles back to the instruction address responsible.
              This is the part general SCA tooling does not do, because on real
              hardware the mapping from trace sample to PC is unknown. In a
              simulator it is free, and it is what makes a result actionable to
              the engineer who has to fix the code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Conventional TVLA detection threshold (~5 sigma, Bonferroni-ish allowance for
#: the number of samples). Reported alongside the achieved maximum so a reader
#: can apply a different threshold without rerunning.
TVLA_THRESHOLD = 4.5


def align(traces, mode: str = "pad", fill: float = 0.0) -> np.ndarray:
    """Stack variable-length traces into a matrix.

    Data-dependent execution time means traces genuinely differ in length. That
    difference *is* the timing channel, so it must not be silently destroyed:

      "pad"      keeps every sample and pads short traces (default). The tail
                 region will then flag as leaking, correctly, because presence
                 or absence of activity there depends on the secret.
      "truncate" cuts to the shortest trace, isolating power leakage from
                 timing leakage. Use this to answer "does it still leak even if
                 I fix the timing?".
    """
    traces = [np.asarray(t, dtype=np.float64) for t in traces]
    if not traces:
        raise ValueError("no traces given")
    if mode == "truncate":
        n = min(len(t) for t in traces)
        return np.stack([t[:n] for t in traces])
    if mode != "pad":
        raise ValueError(f"unknown align mode {mode!r}")
    n = max(len(t) for t in traces)
    out = np.full((len(traces), n), fill, dtype=np.float64)
    for i, t in enumerate(traces):
        out[i, : len(t)] = t
    return out


def welch_t(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-sample Welch t statistic between two trace groups."""
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    n = min(a.shape[1], b.shape[1])
    a, b = a[:, :n], b[:, :n]
    va = a.var(axis=0, ddof=1) / a.shape[0]
    vb = b.var(axis=0, ddof=1) / b.shape[0]
    denom = np.sqrt(va + vb)
    diff = a.mean(axis=0) - b.mean(axis=0)
    # Zero within-group variance is not "no evidence" -- if the two groups are
    # each deterministic and their means differ, the groups are perfectly
    # separable, which is the strongest leak possible. Collapsing that to t = 0
    # (the naive nan_to_num) reports the worst case as clean. Only the genuinely
    # degenerate case, equal means and no variance, is zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(denom > 0, diff / np.where(denom > 0, denom, 1.0),
                     np.where(diff != 0, np.sign(diff) * np.inf, 0.0))
    return np.nan_to_num(t, nan=0.0, posinf=np.inf, neginf=-np.inf)


@dataclass
class TvlaResult:
    t: np.ndarray
    threshold: float
    peak_sample: int
    peak_t: float
    leaking_samples: np.ndarray
    n_fixed: int
    n_random: int

    @property
    def leaks(self) -> bool:
        return bool(self.peak_t > self.threshold)

    def summary(self) -> str:
        verdict = "LEAK DETECTED" if self.leaks else "no first-order leak detected"
        if self.leaks and np.isinf(self.peak_t):
            verdict = "LEAK DETECTED (deterministic separation)"
        return (
            f"{verdict}: max |t| = {self.peak_t:.2f} at sample {self.peak_sample} "
            f"(threshold {self.threshold}, {self.n_fixed}+{self.n_random} traces, "
            f"{len(self.leaking_samples)} samples above threshold)"
        )


def tvla(fixed, random_, *, mode: str = "pad", threshold: float = TVLA_THRESHOLD) -> TvlaResult:
    """Run a fixed-vs-random Welch t-test over two sets of traces."""
    a, b = align(fixed, mode), align(random_, mode)
    t = welch_t(a, b)
    peak = int(np.argmax(np.abs(t)))
    return TvlaResult(
        t=t,
        threshold=threshold,
        peak_sample=peak,
        peak_t=float(abs(t[peak])),
        leaking_samples=np.flatnonzero(np.abs(t) > threshold),
        n_fixed=len(fixed),
        n_random=len(random_),
    )


@dataclass
class TimingResult:
    t: float
    mean_fixed: float
    mean_random: float
    threshold: float
    distinct_counts: int

    @property
    def leaks(self) -> bool:
        return abs(self.t) > self.threshold

    def summary(self) -> str:
        if not self.leaks:
            verdict = "constant-time under this model"
        elif np.isinf(self.t):
            verdict = "TIMING LEAK (deterministic separation)"
        else:
            verdict = "TIMING LEAK"
        return (
            f"{verdict}: t = {self.t:.2f}, mean cycles fixed = {self.mean_fixed:.1f} "
            f"vs random = {self.mean_random:.1f}, {self.distinct_counts} distinct "
            f"cycle counts observed"
        )


def timing_test(fixed_cycles, random_cycles, threshold: float = TVLA_THRESHOLD) -> TimingResult:
    """Welch t-test on total execution time -- the channel visible to a remote
    attacker who cannot measure power at all."""
    f = np.asarray(fixed_cycles, dtype=np.float64)
    r = np.asarray(random_cycles, dtype=np.float64)
    denom = np.sqrt(f.var(ddof=1) / f.size + r.var(ddof=1) / r.size)
    diff = f.mean() - r.mean()
    if denom > 0:
        t = float(diff / denom)
    else:
        # Both groups deterministic: either identical (no channel) or perfectly
        # separable (total channel). See the note in welch_t.
        t = 0.0 if diff == 0 else float(np.sign(diff) * np.inf)
    return TimingResult(
        t=t,
        mean_fixed=float(f.mean()),
        mean_random=float(r.mean()),
        threshold=threshold,
        distinct_counts=len(set(map(int, np.concatenate([f, r])))),
    )


# --- Attribution ------------------------------------------------------------

@dataclass
class Attribution:
    pc: int
    peak_t: float
    sample: int
    share: float   # fraction of above-threshold samples charged to this PC


def attribute(result: TvlaResult, pc_matrix, top: int = 10) -> list[Attribution]:
    """Charge leaking samples to the instruction addresses that produced them.

    A sample index may correspond to different PCs in different traces once
    control flow diverges -- which is exactly what happens in a secret-dependent
    branch. We therefore take, per sample, the most common PC across traces, and
    aggregate.
    """
    pcs = align(pc_matrix, mode="pad", fill=-1).astype(np.int64)
    charged: dict[int, list[float]] = {}
    for s in result.leaking_samples:
        col = pcs[:, s]
        col = col[col >= 0]
        if col.size == 0:
            continue
        values, counts = np.unique(col, return_counts=True)
        pc = int(values[int(np.argmax(counts))])
        charged.setdefault(pc, []).append(abs(float(result.t[s])))

    total = sum(len(v) for v in charged.values()) or 1
    out = [
        Attribution(
            pc=pc,
            peak_t=max(ts),
            sample=int(result.leaking_samples[0]) if len(result.leaking_samples) else -1,
            share=len(ts) / total,
        )
        for pc, ts in charged.items()
    ]
    out.sort(key=lambda a: a.peak_t, reverse=True)
    return out[:top]


# --- Correlation power analysis --------------------------------------------

HW = np.array([bin(i).count("1") for i in range(256)], dtype=np.float64)


@dataclass
class CpaResult:
    correlations: np.ndarray      # (hypotheses, samples)
    best: int
    best_corr: float
    runner_up_corr: float
    sample: int
    signed: bool = True
    complement_tie: bool = False

    @property
    def margin(self) -> float:
        """Distance between the top and second hypothesis. A margin near zero
        means the "recovery" is not distinguishable from noise."""
        return self.best_corr - self.runner_up_corr

    def rank_of(self, key: int) -> int:
        return int(np.argsort(-self._peaks()).tolist().index(key))

    def _peaks(self) -> np.ndarray:
        return (np.max(self.correlations, axis=1) if self.signed
                else np.max(np.abs(self.correlations), axis=1))


def cpa(traces, hypotheses: np.ndarray, *, signed: bool = True) -> CpaResult:
    """Correlation power analysis.

    `hypotheses` is (n_traces, n_hypotheses) of predicted leakage values -- for
    the classic Hamming-weight model on an S-box input this is HW(p ^ k) for
    every candidate k. Pearson correlation is computed against every sample.

    Ranking is on *signed* correlation by default. Under a Hamming-weight model
    HW(p ^ ~k) = 8 - HW(p ^ k), so a candidate and its bitwise complement always
    produce correlations of identical magnitude and opposite sign. Ranking on
    |rho| therefore leaves an irreducible two-way tie for every key byte -- a
    real property of the model, not a numerical accident. Signed ranking
    resolves it under the (physically standard) assumption that switching
    activity increases with Hamming weight. Pass signed=False to see the
    ambiguity explicitly; `complement_tie` reports whether it was present.
    """
    x = align(traces, mode="pad")
    h = np.asarray(hypotheses, dtype=np.float64)
    if h.shape[0] != x.shape[0]:
        raise ValueError("hypothesis matrix must have one row per trace")

    xc = x - x.mean(axis=0, keepdims=True)
    hc = h - h.mean(axis=0, keepdims=True)
    xs = np.sqrt((xc ** 2).sum(axis=0))
    hs = np.sqrt((hc ** 2).sum(axis=0))
    denom = np.outer(hs, xs)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.nan_to_num((hc.T @ xc) / denom)

    peaks = np.max(corr, axis=1) if signed else np.max(np.abs(corr), axis=1)
    order = np.argsort(-peaks)
    best = int(order[0])
    abs_peaks = np.max(np.abs(corr), axis=1)
    tie = bool(
        peaks.size > 1
        and abs(abs_peaks[best] - abs_peaks[best ^ 0xFF]) < 1e-9
        and best ^ 0xFF < peaks.size
    )
    return CpaResult(
        correlations=corr,
        best=best,
        best_corr=float(peaks[best]),
        runner_up_corr=float(peaks[order[1]]) if peaks.size > 1 else 0.0,
        sample=int(np.argmax(np.abs(corr[best]))),
        signed=signed,
        complement_tie=tie,
    )


def hw_hypotheses(plaintexts, n_candidates: int = 256) -> np.ndarray:
    """HW(p ^ k) hypothesis matrix for every candidate k."""
    p = np.asarray(plaintexts, dtype=np.int64).reshape(-1, 1)
    k = np.arange(n_candidates, dtype=np.int64).reshape(1, -1)
    return HW[np.bitwise_xor(p, k)]


def traces_to_recover(traces, hypotheses, correct_key: int, step: int = 10) -> int | None:
    """Smallest trace count (on a `step` grid) at which the correct key first
    ranks first and stays first for the rest of the sweep."""
    n = len(traces)
    first_stable = None
    for m in range(step, n + 1, step):
        r = cpa(list(traces)[:m], hypotheses[:m])
        if r.best == correct_key:
            if first_stable is None:
                first_stable = m
        else:
            first_stable = None
    return first_stable


def pc_window(pc_matrix, pc: int, occurrence: int | None = None) -> np.ndarray:
    """Boolean mask over samples attributable to instruction address `pc`.

    Point-of-interest selection is normally done by eyeballing a t-test plot.
    Because the simulator knows which instruction produced each sample, it can
    be done exactly instead: "give me the samples belonging to the table load,
    on its `occurrence`-th execution". That turns POI selection from a manual,
    error-prone step into a query, and it is the main practical advantage of
    analysing a model rather than a physical board.
    """
    pcs = align(pc_matrix, mode="pad", fill=-1).astype(np.int64)
    mask = np.zeros(pcs.shape[1], dtype=bool)
    for s_idx in range(pcs.shape[1]):
        col = pcs[:, s_idx]
        col = col[col >= 0]
        if col.size == 0:
            continue
        values, counts = np.unique(col, return_counts=True)
        if int(values[int(np.argmax(counts))]) == pc:
            mask[s_idx] = True
    if occurrence is None:
        return mask
    # Split the mask into contiguous runs; keep only the requested execution.
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    out = np.zeros_like(mask)
    if occurrence < len(runs):
        a, b = runs[occurrence]
        out[a:b] = True
    return out


def extract_windows(power_list, pc_list, pc: int, occurrence: int = 0,
                    length: int | None = None) -> np.ndarray:
    """Per-trace extraction of the samples belonging to one execution of `pc`.

    `pc_window` votes on a majority PC per absolute sample index, which breaks
    as soon as traces diverge in length -- exactly what a data-dependent cache
    miss causes. The result is a window that silently shrinks to one sample for
    later occurrences, quietly destroying the analysis.

    This function instead walks each trace's own PC record, finds the
    `occurrence`-th contiguous run of `pc` in *that* trace, and stacks the
    extracted windows. Traces are thereby aligned on the instruction of
    interest rather than on wall-clock position.

    This is the single largest practical advantage of analysing a model rather
    than an oscilloscope capture: on real hardware this alignment must be
    recovered heuristically (static alignment, elastic alignment, correlation
    peaks), and imperfect alignment is one of the main reasons published
    attacks need far more traces than theory predicts. Here it is exact.
    """
    windows = []
    for power, pcs in zip(power_list, pc_list, strict=True):
        pcs = np.asarray(pcs)
        hit = pcs == pc
        runs, start = [], None
        for i, v in enumerate(hit):
            if v and start is None:
                start = i
            elif not v and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, len(hit)))
        if occurrence >= len(runs):
            windows.append(np.zeros(1))
            continue
        a, b = runs[occurrence]
        windows.append(np.asarray(power)[a:b])
    n = length or max(len(w) for w in windows)
    out = np.zeros((len(windows), n))
    for i, w in enumerate(windows):
        out[i, : min(n, len(w))] = w[:n]
    return out


def hd_hypotheses(plaintexts, table_base: int, prev_addr: int,
                  n_candidates: int = 256) -> np.ndarray:
    """Hamming-distance bus hypotheses: HD(table_base + (p ^ k), prev_addr).

    A bus does not dissipate energy proportional to the value it carries; it
    dissipates energy proportional to the number of lines that *change*. Where
    the previous bus state is known -- and for a fixed code sequence it is,
    since it is the address of the immediately preceding load -- the Hamming
    distance model is the correct one and the Hamming weight model is merely an
    approximation to it.

    The practical consequence is visible in docs/results.md: under an HD-leaking
    bus, HW-model CPA returns candidates a Hamming step or two away from the
    true key rather than the key itself. The attack is not failing at random --
    it is succeeding against the wrong model.
    """
    p = np.asarray(plaintexts, dtype=np.int64).reshape(-1, 1)
    k = np.arange(n_candidates, dtype=np.int64).reshape(1, -1)
    addr = table_base + np.bitwise_xor(p, k)
    xor = np.bitwise_xor(addr, prev_addr)
    out = np.zeros(xor.shape, dtype=np.float64)
    for bit in range(32):
        out += (xor >> bit) & 1
    return out
