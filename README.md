# rvleak

**Microarchitectural leakage analysis for RV32IM software, with per-instruction attribution.**

[![CI](https://github.com/Urvish-Kosta/rvleak/actions/workflows/ci.yml/badge.svg)](https://github.com/Urvish-Kosta/rvleak/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

rvleak is a cycle-level RV32IM simulator that emits a per-cycle activity trace
**together with an exact cycle-to-instruction mapping**, plus the statistical
machinery (TVLA, CPA) to decide whether a piece of software leaks a secret — and
to name the instruction responsible.

```
$ rvleak tvla memcmp-early-exit -n 200
power TVLA     : LEAK DETECTED: max |t| = 163.24 at sample 58
timing         : TIMING LEAK: t = 3043.44, mean cycles fixed = 195.0 vs random = 58.0
attribution    : leaking instruction addresses
                 pc=0x00000024  peak |t|= 163.24   ->  lbu  x6, 0(x6)
                 pc=0x00000030  peak |t|=  98.08   ->  bne  x6, x7, +12
```

---

## Table of contents

- [Motivation](#motivation)
- [Problem statement](#problem-statement)
- [Existing tools and the gap](#existing-tools-and-the-gap)
- [What rvleak does](#what-rvleak-does)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Design decisions and trade-offs](#design-decisions-and-trade-offs)
- [Bugs found by running the code](#bugs-found-by-running-the-code)
- [Validation strategy](#validation-strategy)
- [Limitations](#limitations)
- [Future work](#future-work)
- [References](#references)
- [License](#license)

---

## Motivation

Side-channel countermeasures are written in software but broken by hardware. A
developer writes a constant-time comparison; a cache turns it back into a timing
oracle. A cryptographer specifies a branchless ladder; a bus that dissipates
energy proportional to Hamming distance leaks the operands anyway.

Evaluating this normally requires a board, a probe, and an oscilloscope. That is
a real barrier — and more importantly, when a leak *is* found on physical
hardware, the trace tells you *when* it leaked, not *which line of code* caused
it. Mapping a spike in a captured waveform back to a source instruction is
manual, error-prone work.

In a simulator that mapping is free and exact. rvleak is built around that
observation.

## Problem statement

Given an RV32IM program, a definition of a secret, and a microarchitecture:

1. Does execution time depend on the secret? (remote timing attacker)
2. Does modelled switching activity depend on the secret? (power/EM attacker)
3. If so, **which instruction** is responsible?
4. Is the dependence strong enough to actually recover key material, and how
   many traces does that take?

## Existing tools and the gap

| Tool | Approach | Gap for this problem |
|---|---|---|
| ChipWhisperer | Real hardware capture + analysis | Needs hardware; no cycle→instruction mapping |
| Jlsca / SCARED / Lascar | Trace analysis libraries | Consume traces; do not produce or attribute them |
| ctgrind / dudect | Dynamic constant-time checking | Timing only, on the host ISA, no microarchitectural model |
| Binsec/Rel, ct-verif | Formal constant-time verification | Sound and strong, but no quantitative attack cost, and cache/power models are typically abstracted away |
| gem5, Spike | Cycle/functional simulation | No leakage model, no statistical layer |

Each side is well served. What is missing is the **join**: a simulator whose
trace output carries provenance, so that detection, exploitation, and
attribution happen in one pass over the same data. rvleak is deliberately small
and occupies exactly that join.

## What rvleak does

Three layers, in increasing order of strength of claim:

- **Detect** — TVLA fixed-vs-random Welch t-test. Says *whether* a secret
  influences the observable. No key hypothesis needed.
- **Exploit** — CPA. Says whether the leak actually recovers key material and
  at what trace cost. Detection without exploitation overstates risk.
- **Attribute** — maps leaking cycles back to instruction addresses. This is
  what makes a result actionable.

Attribution is not only a reporting nicety: it feeds back into the attack.
Point-of-interest selection, normally done by eyeballing a t-test plot, becomes
a query — *"give me the samples belonging to the table load on its 5th
execution"* — and traces are aligned on the instruction of interest rather than
on wall-clock position. Section 5.1 of [docs/results.md](docs/results.md) shows
that without this, CPA converges on a ghost peak and recovers nothing.

## Architecture

```
                      ┌────────────────────────────────────────┐
   victim .S ────────►│  asm.py    two-pass RV32IM assembler   │
   (or flat binary)   └───────────────┬────────────────────────┘
                                      │ Program(words, labels)
                                      ▼
                      ┌────────────────────────────────────────┐
                      │  isa.py    decode + architectural       │
                      │            semantics (no timing)        │
                      └───────────────┬────────────────────────┘
                                      │
                                      ▼
   ModelConfig ──────►┌────────────────────────────────────────┐
   cache geometry     │  uarch.py   in-order pipeline           │
   predictor          │   ├── SetAssocCache   (LRU, tags only)  │
   penalties          │   ├── BranchPredictor (gshare/bimodal)  │
   leakage weights    │   └── activity model  (HD/HW + noise)   │
   noise sigma        └───────────────┬────────────────────────┘
                                      │
                          Trace ──────┤  power[c]        per-cycle activity
                                      │  pc_of_cycle[c]  ◄── provenance
                                      │  cycles, hits, misses, mispredicts
                                      ▼
                      ┌────────────────────────────────────────┐
                      │  analysis.py                            │
                      │   ├── tvla()            detect          │
                      │   ├── timing_test()     detect (remote) │
                      │   ├── attribute()       ─► which PC     │
                      │   ├── extract_windows() ─► align on PC  │
                      │   └── cpa()             exploit         │
                      └───────────────┬────────────────────────┘
                                      ▼
                       campaign.py ──► report.py ──► CLI / PNG / docs
```

The split between `isa.py` (what an instruction computes) and `uarch.py` (when,
and at what modelled cost) is load-bearing. It means the functional model can be
validated independently — against Python's `pow()`, against a reference
implementation of the comparison — so that an observed timing difference is
known to be a property of the microarchitecture and not a symptom of a broken
interpreter.

## Installation

No cross-toolchain, no simulator, no hardware. Python 3.10+ and NumPy.

```bash
git clone https://github.com/Urvish-Kosta/rvleak.git
cd rvleak
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                     # ~17 s, 80 tests
```

Or with Docker:

```bash
docker build -t rvleak . && docker run --rm rvleak fullkey -n 1000
```

The built-in assembler exists specifically to avoid a riscv32 GCC/LLVM
dependency. Victims are 20–100 instructions; anything larger should be built
with a real toolchain and loaded via `Program.from_words`.

## Usage

```bash
rvleak list                                  # available victims
rvleak disasm memcmp-early-exit              # disassemble one
rvleak null                                  # false-positive control
rvleak tvla                                  # detect leaks in all victims
rvleak tvla table-lookup -n 500 --figures assets/
rvleak cpa -n 1000 --byte 3                  # recover one key byte
rvleak fullkey -n 1000                       # recover all 16
rvleak sweep --sigmas 0.5 1 2 4 8            # attack cost vs noise
```

Microarchitecture is a command-line knob, so a countermeasure can be shown to
hold across a design space rather than at a single point:

```bash
rvleak tvla table-lookup --sets 16 --ways 1 --miss-penalty 60
rvleak tvla modexp-ladder --data-dependent-div     # model an early-exit divider
rvleak fullkey --noise 8.0 -n 2000
```

As a library:

```python
from rvleak import Machine, ModelConfig, tvla
from rvleak.victims import build_table_lookup

m = Machine(ModelConfig(noise_sigma=0.5))
b = build_table_lookup(plaintext, key)
trace = m.run(b.program, memory=b.memory, regs=b.regs, trace_seed="t0")
print(trace.cycles, trace.dcache_misses, trace.power[:8])
```

## Results

Full generated output, including figures: **[docs/results.md](docs/results.md)**.
Regenerate everything with `make results` — no number in this repository is
transcribed by hand.

**Detector validation.** Fixed-vs-fixed control, 200+200 traces: max |t| = 2.93,
zero samples above the 4.5 threshold. This runs in CI, because every negative
result below depends on it.

**Detection across leaky/hardened pairs** (200 traces per group):

| victim | class | power max \|t\| | timing | mean cycles fixed / random |
|---|---|---|---|---|
| table-lookup | leaky | 25.0 | **LEAK** | 444 / 427 |
| table-lookup-preloaded | hardened | 28.8 | constant | 500 / 500 |
| modexp-square-multiply | leaky | 102.7 | **LEAK** | 1720 / 1857 |
| modexp-ladder | hardened | 75.4 | constant | 3880 / 3880 |
| memcmp-early-exit | leaky | 163.2 | **LEAK** | 195 / 58 |
| memcmp-constant-time | hardened | 72.5 | constant | 211 / 211 |

The timing column separates the pairs exactly. **The power column does not** —
and that is the substantive result, not a shortcoming of the tool. Every
hardened victim here fixes *control flow*, which is what its textbook
description promises; none of them fixes first-order power leakage, because
masking is a separate countermeasure. Preloading the table removes the cache
channel and leaves the address-bus channel entirely intact. A tool that reported
these as "secure" would be actively misleading, so `test_constant_time_is_not_power_secure`
asserts the positive detection as a deliberate expected result.

**Key recovery.** 16/16 key bytes of the table-lookup victim, from 1000 traces,
worst-case 150 traces for stable recovery of any single byte:

```
recovered key   : 2b7e151628aed2a6abf7158809cf4f3c
true key        : 2b7e151628aed2a6abf7158809cf4f3c
```

**Attack cost versus noise** (full-key recovery, 1000 traces available):

| noise sigma | bytes recovered | worst-case traces to stable recovery |
|---|---|---|
| 0.5 | 16/16 | 150 |
| 1.0 | 16/16 | 150 |
| 2.0 | 16/16 | 200 |
| 4.0 | 16/16 | 600 |
| 8.0 | 15/16 | 950 |

The monotone growth is the expected CPA behaviour and serves as a sanity check
that the noise model is doing something.

## Design decisions and trade-offs

**Per-instruction cycle accounting rather than an explicit five-stage pipeline.**
For a scalar in-order pipeline with no out-of-order effects, the two are
equivalent for cycle counting, and the per-instruction form makes cycle→PC
attribution exact rather than approximate. The cost is that overlapping
long-latency operations cannot be represented. Since attribution is the entire
point of the tool, exactness there is worth more than fidelity to an effect the
target class of cores does not have.

**Modelled leakage, not measured leakage.** The power channel is a
Hamming-weight/Hamming-distance model with additive Gaussian noise — the
standard first-order model of the DPA/CPA literature. A leak reported by rvleak
means *"this code leaks under the standard first-order model on a
microarchitecture of this shape"*. It does not mean *"this code leaks N bits on
your FPGA"*. This is stated in the module docstring, here, and in the
limitations section, because it is the single claim most easily overstated.

**Cold cache per trace.** Removes cross-trace state as a confounder, so a
detected leak is attributable to this execution's secret rather than the
previous trace's. Warm-start is available via `reset_state=False` and is the
natural extension for prime+probe experiments.

**Signed ranking in CPA.** Under a Hamming-weight model, HW(p ^ ~k) = 8 − HW(p ^ k),
so a candidate and its bitwise complement always produce correlations of
identical magnitude and opposite sign. Ranking on |ρ| leaves an irreducible
two-way tie for every key byte. Signed ranking resolves it under the physically
standard assumption that switching activity increases with Hamming weight, and
`complement_tie` reports when the ambiguity was present rather than hiding it.

**Both alignment modes exposed.** Data-dependent execution time means traces
genuinely differ in length, and that difference *is* the timing channel.
`pad` keeps it; `truncate` discards it to isolate power leakage, answering the
distinct question *"does this still leak if I fix the timing?"*.

## Bugs found by running the code

These were found by executing the tool during development, not anticipated in
design. Each has a regression test.

1. **Zero within-group variance from a shared noise seed.** The RNG was
   re-seeded identically for every run, so all traces in a group carried an
   identical noise realisation. Within-group variance was ~1e-30 and the Welch
   t statistic reached 3.2e15 — every victim, including the hardened ones,
   appeared to leak overwhelmingly. The failure mode is silent and looks like a
   spectacular positive result. `trace_seed` is now an explicit parameter.
   (`test_within_group_variance_is_nonzero`)

2. **Perfect separation reported as no leak.** `memcmp-early-exit` showed 195 vs
   58 mean cycles — total separation — and the timing test reported `t = 0.00`,
   "constant-time". Both groups were deterministic, so the denominator was zero
   and a naive `nan_to_num` mapped the worst possible case onto the same value
   as the best. Now distinguished: equal means and no variance → 0; different
   means and no variance → ±∞.
   (`test_zero_variance_with_different_means_is_infinite_not_zero`)

3. **POI windows silently collapsing under trace divergence.** Majority-vote
   attribution by absolute sample index worked for key byte 0 (21-sample window)
   and degraded to a *1-sample* window by byte 5, because data-dependent cache
   misses shift later instructions between traces. Full-key recovery was 3/16
   and the cause was invisible in the output. Fixed by extracting each trace's
   window from its own PC record; recovery went to 16/16.
   (`test_extract_windows_aligns_on_the_instruction_not_the_clock`)

4. **Leakage model mismatch mistaken for attack failure.** With alignment fixed,
   CPA returned candidates consistently *one or two Hamming steps* from the true
   key (0x7f for 0x7e, 0x2c for 0x28). The bus model leaks Hamming *distance*
   against the previous address; the hypothesis was Hamming *weight*. Not noise
   — the attack was succeeding against the wrong model. Adding `hd_hypotheses`
   took recovery from 1/16 to 16/16. Retained as a selectable option and an
   ablation, since it is a good demonstration of why model choice dominates
   attack cost. (`test_mismatched_leakage_model_degrades_recovery`)

## Validation strategy

Leakage tooling has an unusually nasty failure mode: a broken detector that
always fires looks exactly like a very sensitive one. The suite is built around
that.

- **Functional ground truth.** Every victim is checked against an independent
  Python reference (`pow()`, direct table indexing, byte comparison) before any
  leakage claim is made. A timing difference from a broken interpreter is not a
  side channel.
- **Cross-checked encoding.** The assembler and the decoder are independent
  implementations of the RV32IM encoding; round-trip tests check them against
  each other.
- **False-positive control.** Fixed-vs-fixed on every hardened victim must
  *not* fire. This is the only reason a negative result carries weight.
- **Asserted negatives.** `test_constant_time_is_not_power_secure` requires the
  tool to keep reporting power leakage in the hardened victims — so a change
  that overstates their security fails CI.
- **Ablations as tests.** Disabling POI selection must fail to recover; the
  mismatched leakage model must rank worse. Failure modes are pinned, not just
  documented.
- **Analysis validated on synthetic data.** CPA is checked against traces whose
  leakage is HW(p ^ k) by construction, independently of the simulator.
- **Monotonicity.** More noise must cost more traces. If it did not, the noise
  model would not be doing anything.

80 tests, ~17 s, run on every push.

## Limitations

Stated plainly, because the value of the tool depends on not overclaiming.

- **This is a model, not a measurement.** No claim is made about any physical
  device. Correspondence with silicon is unvalidated — there is no board in this
  project and none of the results should be read as if there were.
- **First-order only.** Second-order and higher-order leakage, glitch power,
  coupling, and static/leakage current are out of scope. A negative result rules
  out first-order HW/HD leakage and structural timing leakage, and nothing else.
- **No overlapping long-latency operations.** A consequence of per-instruction
  cycle accounting; fine for scalar in-order cores, wrong for anything with a
  scoreboard or an out-of-order window.
- **No speculative execution.** Mispredicts cost cycles but do not perform
  transient memory accesses, so Spectre-class effects cannot be modelled. This
  is the single largest architectural gap and the main item in future work.
- **Tag-only cache.** No coherence, no write buffers, no prefetcher, no TLB.
- **No CSRs, interrupts, or privilege levels.** ECALL halts.
- **The victims are microbenchmarks**, not a real cipher. They exercise the
  mechanisms (table lookup, secret-dependent branch, early exit) that real
  ciphers exhibit; they are not an AES implementation and are not presented as
  one. The lookup table is a seeded random permutation, not any cipher's S-box.

## Future work

Ordered by value, not ease.

1. **Transient execution.** Let mispredicted paths issue real cache accesses
   before squashing. This is what turns the model from a power/timing tool into
   one that can express Spectre-v1 gadgets, and the predictor and cache needed
   for it are already here.
2. **Cross-validation against RTL.** Drive the same victims through a
   Verilator-simulated RV32IM core, derive activity from signal toggle counts
   rather than an analytic model, and compare the two t-traces. This is the
   experiment that would convert "modelled leakage" into a calibrated claim, and
   it is the obvious next step for anyone with an RTL core to hand.
3. **Masking-aware analysis.** Second-order CPA (centred product combining) plus
   a masked victim pair, to extend the tool past the first-order boundary.
4. **Prime+probe and eviction-set experiments** using the existing warm-start
   path and `SetAssocCache.prime`.
5. **Leakage-aware compilation.** The attribution output is already a
   per-instruction leakage score; feeding it back as a cost function to select
   between semantically equivalent instruction sequences is a small step with a
   plausible research result at the end of it.
6. **Real-toolchain victims** via `Program.from_words`, to analyse compiled
   library code rather than hand-written microbenchmarks.

## References

- P. Kocher, J. Jaffe, B. Jun. *Differential Power Analysis.* CRYPTO 1999.
- E. Brier, C. Clavier, F. Olivier. *Correlation Power Analysis with a Leakage
  Model.* CHES 2004.
- G. Goodwill, B. Jun, J. Jaffe, P. Rohatgi. *A testing methodology for
  side-channel resistance validation.* NIST Non-Invasive Attack Testing
  Workshop, 2011.
- D. J. Bernstein. *Cache-timing attacks on AES.* 2005.
- S. Mangard, E. Oswald, T. Popp. *Power Analysis Attacks: Revealing the Secrets
  of Smart Cards.* Springer, 2007.
- A. Waterman, K. Asanović (eds.). *The RISC-V Instruction Set Manual, Volume I:
  Unprivileged ISA.*

## License

MIT — see [LICENSE](LICENSE).

## Author

Urvish Kosta — embedded systems and digital design engineer.
[GitHub](https://github.com/Urvish-Kosta) · [LinkedIn](https://www.linkedin.com/in/urvish-kosta)
