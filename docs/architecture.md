# Architecture

## Module responsibilities

| Module | Responsibility | Deliberately excluded |
|---|---|---|
| `isa.py` | Instruction decode and architectural semantics: what an instruction computes | Any notion of time, cost, or energy |
| `asm.py` | Two-pass RV32IM assembler with labels and pseudo-instructions | Optimisation, macros, linking |
| `uarch.py` | Cache, branch predictor, cycle accounting, per-cycle activity model | Instruction semantics (imported from `isa`) |
| `analysis.py` | Statistics: TVLA, timing test, CPA, attribution, windowing | Any knowledge of victims or the simulator |
| `victims.py` | Leaky/hardened victim pairs and the memory map | Analysis |
| `campaign.py` | Reproducible experiments composing the above | New statistics or new semantics |
| `report.py` | ASCII and matplotlib rendering | Computation |
| `cli.py` | Argument parsing and exit codes | Everything else |

The dependency graph is acyclic and one-directional:

```
isa ──► uarch ──► campaign ──► cli
 ▲                  ▲   ▲        │
 │                  │   │        ▼
asm ──► victims ────┘   │     report
                        │
                  analysis
```

`analysis.py` does not import `uarch.py` or `victims.py`. It consumes plain
arrays. That is what allows the CPA implementation to be tested against
synthetic traces whose leakage is known by construction, independently of
whether the simulator is correct.

## The central data structure

```python
@dataclass
class Trace:
    power: np.ndarray         # one activity sample per cycle
    pc_of_cycle: np.ndarray   # same length: which PC produced each sample
    cycles: int
    retired: int
    dcache_hits: int
    dcache_misses: int
    branch_correct: int
    branch_wrong: int
    halted: bool
    regs: list[int]
```

`pc_of_cycle` is the design's one real idea. On physical hardware the mapping
from trace sample to source instruction is unknown and must be recovered
heuristically; here it is produced as a by-product of simulation. Everything
distinctive downstream — attribution, exact point-of-interest selection, and
per-trace alignment on an instruction rather than a wall-clock position — is a
consequence of carrying it.

## Timing model

Cycles are accounted per retired instruction:

```
cycles(i) = 1
          + load_use_penalty      if this instruction reads the previous load's rd
          + dcache.miss_penalty   if a load/store misses
          + mispredict_penalty    if a branch is mispredicted
          + mul_latency - 1       for multiplies
          + div_cycles(a, b) - 1  for divides
          + icache.miss_penalty   if instruction fetch misses (optional)
```

For a scalar in-order pipeline with no out-of-order effects this is equivalent
to simulating five stages explicitly, and it makes cycle→PC attribution exact.
It cannot represent overlapping long-latency operations; see the limitations in
the README.

## Activity model

Per executed instruction:

```
activity = w_hw_operand  * (HW(rs1) + HW(rs2))
         + w_hd_result   * HD(result, previous_result)
         + w_hw_address  * HD(address, previous_address)     [memory ops]
         + w_hw_memdata  * HW(data)                          [memory ops]
         + w_miss_energy                                     [on a cache miss]
         + N(0, noise_sigma)
```

Stall cycles emit `idle_power` plus noise — bubbles are not free, and the
absence of activity where activity would otherwise be is itself informative.

Note the *distance* terms. A bus dissipates energy proportional to the number of
lines that change, not the value carried. This is why `hd_hypotheses` rather
than `hw_hypotheses` is the default attack model, and why using the wrong one
returns near-neighbours of the true key rather than the key.

## Extension points

- **New victim**: add a builder to `victims.py`, register it in `REGISTRY`, and
  add a functional correctness test against an independent reference. Mark the
  point of interest with a `poi:` label if it should be CPA-targetable.
- **New microarchitectural feature**: extend `ModelConfig` and `Machine.run`.
  Add a test proving the feature changes cycle counts in the expected direction
  when enabled and does not when disabled — `test_data_dependent_divider_leaks_only_when_enabled`
  is the template.
- **New statistic**: add to `analysis.py`, validated on synthetic data with a
  known answer, plus a false-positive control on data with no channel.
- **New leakage model**: add a hypothesis generator alongside `hw_hypotheses`
  and `hd_hypotheses`, and wire it into `_hypotheses` in `campaign.py`.
