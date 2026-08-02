"""Experiment drivers.

A campaign is a reproducible experiment: fixed seed in, trace set and verdict
out. Everything the README reports is produced by one of these functions, so any
number in the documentation can be regenerated with a single command. Nothing is
transcribed by hand.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from . import analysis, victims
from .uarch import Machine, ModelConfig


@dataclass
class TraceSet:
    power: list[np.ndarray] = field(default_factory=list)
    pcs: list[np.ndarray] = field(default_factory=list)
    cycles: list[int] = field(default_factory=list)
    inputs: list[bytes] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.power)


#: Fixed reference values. TVLA calls for one fixed input class and one random
#: class; the specific constants are arbitrary but must be stable across runs.
FIXED_PLAINTEXT = bytes([0x53] * 16)
FIXED_EXPONENT = 0x0F0F0F0F
#: For the tag-comparison victims the informative fixed class is the
#: *correct* tag: it exercises the longest matching prefix, so the
#: early-exit path is maximally distinguishable from a random guess.
FIXED_CANDIDATE = bytes([0x5A] * 16)
REFERENCE_TAG = bytes([0x5A] * 16)
#: The AES-128 test key from FIPS-197 Appendix C.1. A published test vector,
#: chosen so that a reader can recognise a successful recovery at a glance.
#: It is not a credential; secret-scanning allowlisting is documented in
#: .gitleaksignore. Override via the `key=` argument to any campaign.
DEFAULT_KEY = bytes([0x2B, 0x7E, 0x15, 0x16, 0x28, 0xAE, 0xD2, 0xA6,
                     0xAB, 0xF7, 0x15, 0x88, 0x09, 0xCF, 0x4F, 0x3C])


def _random_input(name: str, rng: random.Random) -> bytes:
    if name.startswith("modexp"):
        return rng.getrandbits(32).to_bytes(4, "little")
    return bytes(rng.getrandbits(8) for _ in range(16))


def _fixed_input(name: str) -> bytes:
    if name.startswith("modexp"):
        return FIXED_EXPONENT.to_bytes(4, "little")
    if name.startswith("memcmp"):
        return FIXED_CANDIDATE
    return FIXED_PLAINTEXT


def _build(name: str, payload: bytes, key: bytes):
    victim = victims.REGISTRY[name]
    if name.startswith("table-lookup"):
        return victim.build(payload, key)
    if name.startswith("modexp"):
        return victim.build(7, int.from_bytes(payload, "little"), 65521)
    if name.startswith("memcmp"):
        return victim.build(payload, REFERENCE_TAG)
    raise KeyError(name)


def collect(
    name: str,
    n: int,
    *,
    fixed: bool,
    cfg: ModelConfig | None = None,
    key: bytes = DEFAULT_KEY,
    seed: int = 0,
) -> TraceSet:
    """Simulate `n` executions of a victim and return the trace set.

    Each execution starts from a cold cache and a reset predictor. That is the
    conservative choice: it removes cross-trace state as a confounder, so a
    detected leak is attributable to this execution's secret rather than to the
    previous trace's. Warm-start behaviour is available by passing a Machine
    with `reset_state=False`, and is a natural extension for prime+probe style
    experiments.
    """
    cfg = cfg or ModelConfig()
    rng = random.Random(seed)
    machine = Machine(cfg)
    ts = TraceSet()
    for i in range(n):
        payload = _fixed_input(name) if fixed else _random_input(name, rng)
        build = _build(name, payload, key)
        tr = machine.run(build.program, memory=build.memory, regs=build.regs,
                         trace_seed=f"{seed}:{i}")
        ts.power.append(tr.power)
        ts.pcs.append(tr.pc_of_cycle)
        ts.cycles.append(tr.cycles)
        ts.inputs.append(payload)
    return ts


@dataclass
class CampaignResult:
    victim: str
    n_traces: int
    tvla: analysis.TvlaResult
    timing: analysis.TimingResult
    attributions: list[analysis.Attribution]
    align_mode: str

    def summary(self) -> str:
        lines = [
            f"victim         : {self.victim}",
            f"traces         : {self.n_traces} fixed + {self.n_traces} random",
            f"alignment      : {self.align_mode}",
            f"power TVLA     : {self.tvla.summary()}",
            f"timing         : {self.timing.summary()}",
        ]
        if self.attributions:
            lines.append("attribution    : leaking instruction addresses")
            for a in self.attributions[:5]:
                lines.append(
                    f"                 pc={a.pc:#010x}  peak |t|={a.peak_t:7.2f}  "
                    f"share={a.share:5.1%}"
                )
        return "\n".join(lines)


def tvla_campaign(
    name: str,
    n: int = 200,
    *,
    cfg: ModelConfig | None = None,
    align_mode: str = "pad",
    seed: int = 0,
) -> CampaignResult:
    """Full non-specific fixed-vs-random test on one victim, with attribution."""
    fixed = collect(name, n, fixed=True, cfg=cfg, seed=seed)
    rand = collect(name, n, fixed=False, cfg=cfg, seed=seed + 1)
    tv = analysis.tvla(fixed.power, rand.power, mode=align_mode)
    tm = analysis.timing_test(fixed.cycles, rand.cycles)
    attrib = analysis.attribute(tv, fixed.pcs + rand.pcs)
    return CampaignResult(name, n, tv, tm, attrib, align_mode)


@dataclass
class CpaCampaignResult:
    n_traces: int
    target_byte: int
    true_key: int
    model: str
    result: analysis.CpaResult
    rank: int
    traces_needed: int | None

    @property
    def recovered(self) -> bool:
        return self.result.best == self.true_key

    def summary(self) -> str:
        status = "RECOVERED" if self.recovered else "not recovered"
        need = self.traces_needed if self.traces_needed is not None else f"> {self.n_traces}"
        tie = "present (resolved by signed ranking)" if self.result.complement_tie else "none"
        return (
            f"CPA on key byte {self.target_byte} [{self.model} model]: {status}\n"
            f"  true key         : {self.true_key:#04x}\n"
            f"  best candidate   : {self.result.best:#04x} "
            f"(rho = {self.result.best_corr:.3f})\n"
            f"  runner-up rho    : {self.result.runner_up_corr:.3f} "
            f"(margin {self.result.margin:.3f})\n"
            f"  rank of true key : {self.rank}\n"
            f"  complement tie   : {tie}\n"
            f"  traces to stable recovery: {need} of {self.n_traces}"
        )


def _hypotheses(model: str, pt_bytes, byte_index: int) -> np.ndarray:
    if model == "hd":
        return analysis.hd_hypotheses(pt_bytes, victims.TABLE_ADDR,
                                      victims.KEY_ADDR + byte_index)
    if model == "hw":
        return analysis.hw_hypotheses(pt_bytes)
    raise ValueError(f"unknown leakage model {model!r}; use 'hd' or 'hw'")


def cpa_campaign(
    n: int = 1000,
    *,
    target_byte: int = 0,
    key: bytes = DEFAULT_KEY,
    cfg: ModelConfig | None = None,
    victim: str = "table-lookup",
    seed: int = 7,
    sweep_step: int = 25,
    model: str = "hd",
    use_poi: bool = True,
    traces: TraceSet | None = None,
) -> CpaCampaignResult:
    """Recover one key byte of the table-lookup victim by correlation analysis.

    Two knobs exist specifically so that the failure modes can be demonstrated
    rather than hidden:

      `model`   -- "hd" matches the simulated bus (Hamming distance against the
                   previous address); "hw" is the textbook Hamming-weight model
                   and recovers only the byte whose bus transition happens to be
                   dominated by weight.
      `use_poi` -- when False, all samples are used and the plaintext load's
                   leakage produces a ghost peak at candidate 0x00.

    Both defaults are the correct settings; both alternatives are exercised in
    the test suite so that regressions in the analysis cannot masquerade as
    improvements.
    """
    ts = traces if traces is not None else collect(
        victim, n, fixed=False, cfg=cfg, key=key, seed=seed)
    n = len(ts)
    pt_bytes = [p[target_byte] for p in ts.inputs]
    hyp = _hypotheses(model, pt_bytes, target_byte)

    if use_poi:
        poi_pc = _build(victim, bytes(16), key).program.labels["poi"]
        windows = list(analysis.extract_windows(ts.power, ts.pcs, poi_pc,
                                                occurrence=target_byte))
    else:
        windows = ts.power

    res = analysis.cpa(windows, hyp)
    true_key = key[target_byte]
    return CpaCampaignResult(
        n_traces=n,
        target_byte=target_byte,
        true_key=true_key,
        model=model,
        result=res,
        rank=res.rank_of(true_key),
        traces_needed=analysis.traces_to_recover(windows, hyp, true_key, step=sweep_step),
    )


@dataclass
class FullKeyResult:
    n_traces: int
    key: bytes
    recovered: list[int]
    per_byte: list[CpaCampaignResult]

    @property
    def n_correct(self) -> int:
        return sum(r.recovered for r in self.per_byte)

    @property
    def worst_case_traces(self) -> int | None:
        needs = [r.traces_needed for r in self.per_byte if r.recovered]
        return max(needs) if needs and None not in needs else None

    def summary(self) -> str:
        head = (
            f"Full-key recovery, {self.n_traces} traces, "
            f"{self.per_byte[0].model} leakage model\n"
            f"  bytes recovered : {self.n_correct}/16\n"
            f"  recovered key   : {self.recovered_hex()}\n"
            f"  true key        : {self.key.hex()}\n"
            f"  worst-case traces to stable recovery: {self.worst_case_traces}"
        )
        rows = ["", "  byte  true  best  rank   rho   margin  traces"]
        for r in self.per_byte:
            need = r.traces_needed if r.traces_needed is not None else "-"
            rows.append(
                f"  {r.target_byte:4d}  {r.true_key:#04x}  {r.result.best:#04x} "
                f"{r.rank:5d} {r.result.best_corr:6.3f} {r.result.margin:7.3f} {str(need):>7}"
            )
        return head + "\n".join(rows)

    def recovered_hex(self) -> str:
        return bytes(self.recovered).hex()


def full_key_campaign(
    n: int = 1000,
    *,
    key: bytes = DEFAULT_KEY,
    cfg: ModelConfig | None = None,
    model: str = "hd",
    seed: int = 7,
    sweep_step: int = 25,
) -> FullKeyResult:
    """Recover all 16 key bytes from a single trace set.

    Traces are collected once and reused for every byte, which is what a real
    attacker does -- collecting 16 independent sets would misrepresent the cost
    of the attack by a factor of 16.
    """
    ts = collect("table-lookup", n, fixed=False, cfg=cfg, key=key, seed=seed)
    per_byte = [
        cpa_campaign(n, target_byte=b, key=key, cfg=cfg, model=model,
                     sweep_step=sweep_step, traces=ts)
        for b in range(16)
    ]
    return FullKeyResult(n, key, [r.result.best for r in per_byte], per_byte)


def null_campaign(
    name: str = "table-lookup",
    n: int = 200,
    *,
    cfg: ModelConfig | None = None,
    seed: int = 0,
) -> CampaignResult:
    """False-positive control: fixed-vs-fixed.

    Both groups use the identical secret and identical public input, differing
    only in the noise realisation. There is no channel to find, so a detector
    that fires here is broken. This is the TVLA "fixed-vs-fixed" sanity check
    and it is the only reason a *negative* result from this tool carries any
    weight at all. It runs in CI.
    """
    a = collect(name, n, fixed=True, cfg=cfg, seed=seed)
    b = collect(name, n, fixed=True, cfg=cfg, seed=seed + 1000)
    tv = analysis.tvla(a.power, b.power)
    tm = analysis.timing_test(a.cycles, b.cycles)
    return CampaignResult(f"{name} (null: fixed-vs-fixed)", n, tv, tm, [], "pad")
