"""Microarchitectural model: caches, branch predictor, in-order pipeline, and a
per-cycle activity (power) model.

Scope and honesty note
----------------------
This is a *modelled* side channel, not a measured one. rvleak does not claim to
reproduce the analogue behaviour of any real silicon. What it does claim, and
what makes it useful, is narrower and checkable:

  1. The timing channel is structural. Cycle counts fall out of cache hits and
     misses, branch mispredictions, and variable-latency divide. If a victim's
     cycle count depends on a secret, that dependence is a genuine property of
     the code running on this microarchitecture, not an artifact of the power
     model. Any core with a cache and a predictor exhibits the same class of
     dependence.
  2. The power channel is a Hamming-weight / Hamming-distance model with
     additive Gaussian noise. This is the standard first-order model used
     throughout the DPA/CPA literature (Kocher et al. 1999; Brier, Clavier and
     Olivier 2004). It is an approximation of CMOS switching activity, and it is
     the model against which countermeasures are usually first evaluated.

So: a leak reported by rvleak means "this code leaks under the standard
first-order model on a microarchitecture of this shape". It does not mean "this
code leaks N bits on your FPGA". Absence of a reported leak is weaker still --
it rules out first-order HW/HD leakage and structural timing leakage, and
nothing else. Second-order leakage, glitch power, and coupling are out of scope
and are listed as such in the limitations section of the README.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from . import isa
from .isa import ALU, BRANCH, JUMP, LOAD, MULDIV, STORE, SYSTEM


def popcount(x: int) -> int:
    return bin(x & isa.MASK32).count("1")


# --- Configuration ----------------------------------------------------------

@dataclass
class CacheConfig:
    sets: int = 64
    ways: int = 4
    line_bytes: int = 32
    miss_penalty: int = 20

    @property
    def index_bits(self) -> int:
        return int(math.log2(self.sets))


@dataclass
class ModelConfig:
    """Everything that defines the modelled machine.

    Defaults describe a small in-order embedded core: 4 KiB 4-way D-cache,
    gshare predictor, iterative divider. These are the parameters where a real
    design makes a security-relevant choice, so they are all knobs rather than
    constants -- sweeping them is how you show a countermeasure holds across a
    design space rather than at one point in it.
    """

    dcache: CacheConfig = field(default_factory=CacheConfig)
    icache_enabled: bool = False
    icache: CacheConfig = field(default_factory=lambda: CacheConfig(sets=32, ways=2))

    mispredict_penalty: int = 3
    load_use_penalty: int = 1
    mul_latency: int = 3
    div_latency: int = 32
    #: When True the divider terminates early on small operands, which is a real
    #: and frequently overlooked leak in iterative dividers.
    div_data_dependent: bool = False

    bpred: str = "gshare"          # "gshare" | "bimodal" | "always_not_taken"
    ghr_bits: int = 6
    pht_entries: int = 256

    # First-order leakage model weights.
    w_hd_result: float = 1.0       # switching in the writeback / EX result bus
    w_hw_operand: float = 0.35     # operand bus activity
    w_hw_address: float = 1.2      # address bus -- where cache-index leaks live
    w_hw_memdata: float = 1.0      # data bus on loads/stores
    w_miss_energy: float = 4.0     # refill activity on a cache miss
    idle_power: float = 0.5        # stall / bubble cycles are not free
    noise_sigma: float = 1.0       # additive Gaussian measurement noise
    seed: int = 0


# --- Components -------------------------------------------------------------

class SetAssocCache:
    """Set-associative, write-allocate, LRU cache. Only tags are modelled."""

    def __init__(self, cfg: CacheConfig):
        self.cfg = cfg
        self.offset_bits = int(math.log2(cfg.line_bytes))
        self.tags: list[list[int]] = [[] for _ in range(cfg.sets)]
        self.hits = 0
        self.misses = 0

    def reset(self) -> None:
        self.tags = [[] for _ in range(self.cfg.sets)]
        self.hits = self.misses = 0

    def index_of(self, addr: int) -> int:
        return (addr >> self.offset_bits) % self.cfg.sets

    def access(self, addr: int) -> bool:
        """Return True on a hit. Updates LRU order and counters."""
        line = addr >> self.offset_bits
        idx = line % self.cfg.sets
        entries = self.tags[idx]
        if line in entries:
            entries.remove(line)
            entries.append(line)
            self.hits += 1
            return True
        entries.append(line)
        if len(entries) > self.cfg.ways:
            entries.pop(0)
        self.misses += 1
        return False

    def prime(self, base: int, length: int) -> None:
        """Fill the cache from a memory range (models a prime/attacker step)."""
        for a in range(base, base + length, self.cfg.line_bytes):
            self.access(a)


class BranchPredictor:
    """Bimodal or gshare 2-bit predictor with a global history register."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.pht = [1] * cfg.pht_entries
        self.ghr = 0
        self.correct = 0
        self.wrong = 0

    def reset(self) -> None:
        self.pht = [1] * self.cfg.pht_entries
        self.ghr = 0
        self.correct = self.wrong = 0

    def _index(self, pc: int) -> int:
        pc_bits = pc >> 2
        if self.cfg.bpred == "gshare":
            return (pc_bits ^ self.ghr) % self.cfg.pht_entries
        return pc_bits % self.cfg.pht_entries

    def predict(self, pc: int) -> bool:
        if self.cfg.bpred == "always_not_taken":
            return False
        return self.pht[self._index(pc)] >= 2

    def update(self, pc: int, taken: bool) -> bool:
        """Update state; return True if the prediction was correct."""
        predicted = self.predict(pc)
        if self.cfg.bpred != "always_not_taken":
            i = self._index(pc)
            self.pht[i] = min(3, self.pht[i] + 1) if taken else max(0, self.pht[i] - 1)
            self.ghr = ((self.ghr << 1) | int(taken)) & ((1 << self.cfg.ghr_bits) - 1)
        if predicted == taken:
            self.correct += 1
        else:
            self.wrong += 1
        return predicted == taken


# --- Execution result -------------------------------------------------------

@dataclass
class Trace:
    """One simulated execution.

    `power` has one sample per cycle. `pc_of_cycle` has the same length and
    records which instruction address was responsible for each sample; this is
    what turns a statistical detection ("cycle 412 leaks") into an actionable
    one ("the load at 0x0000009c leaks").
    """

    power: np.ndarray
    pc_of_cycle: np.ndarray
    cycles: int
    retired: int
    dcache_hits: int
    dcache_misses: int
    branch_correct: int
    branch_wrong: int
    halted: bool
    regs: list[int]

    @property
    def ipc(self) -> float:
        return self.retired / self.cycles if self.cycles else 0.0


class Machine:
    """In-order RV32IM core with a cache, a branch predictor, and an activity model.

    Timing is accounted per retired instruction rather than by simulating five
    pipeline stages explicitly. For a scalar in-order pipeline with no
    out-of-order effects the two are equivalent for cycle counting, and the
    per-instruction form makes cycle-to-PC attribution exact, which is the whole
    point of the tool. The cost is that this model cannot represent overlapping
    long-latency operations -- see the limitations section.
    """

    def __init__(self, cfg: ModelConfig | None = None):
        self.cfg = cfg or ModelConfig()
        self.dcache = SetAssocCache(self.cfg.dcache)
        self.icache = SetAssocCache(self.cfg.icache)
        self.bpred = BranchPredictor(self.cfg)

    # -- memory helpers ----------------------------------------------------
    def _load(self, mem: bytearray, addr: int, width: int, signed: bool) -> int:
        raw = int.from_bytes(mem[addr:addr + width], "little")
        return isa.u32(isa.sext(raw, width * 8)) if signed else raw

    def _store(self, mem: bytearray, addr: int, width: int, value: int) -> None:
        mem[addr:addr + width] = (value & ((1 << (width * 8)) - 1)).to_bytes(width, "little")

    def _div_cycles(self, a: int, b: int) -> int:
        if not self.cfg.div_data_dependent:
            return self.cfg.div_latency
        # Early-terminating divider: iterate only over significant dividend bits.
        significant = max(1, isa.u32(a).bit_length())
        return max(2, min(self.cfg.div_latency, significant))

    # -- main loop ---------------------------------------------------------
    def run(
        self,
        program,
        *,
        memory: bytearray | None = None,
        regs: dict[int, int] | None = None,
        max_instructions: int = 2_000_000,
        mem_size: int = 1 << 16,
        reset_state: bool = True,
        trace_seed: int | None = None,
    ) -> Trace:
        """Execute `program` and return its `Trace`.

        `trace_seed` seeds the additive measurement noise for *this* execution.
        It must differ between traces in a campaign: re-using one seed makes
        every trace in a group carry an identical noise realisation, driving
        within-group variance to zero and the Welch t statistic to infinity.
        That failure mode is silent -- it looks like an extremely strong leak --
        so the parameter is explicit rather than implicit.
        """
        cfg = self.cfg
        if reset_state:
            self.dcache.reset()
            self.icache.reset()
            self.bpred.reset()
        rng = random.Random(f"{cfg.seed}:{trace_seed}")

        mem = memory if memory is not None else bytearray(mem_size)
        x = [0] * 32
        for r, v in (regs or {}).items():
            x[r] = isa.u32(v)

        base = program.base
        words = program.words
        pc = base
        power: list[float] = []
        pcs: list[int] = []
        prev_result = 0
        prev_addr = 0
        pending_load_rd = -1
        halted = False
        retired = 0

        def emit(pc_val: int, activity: float, count: int = 1) -> None:
            for _ in range(count):
                power.append(activity + rng.gauss(0.0, cfg.noise_sigma))
                pcs.append(pc_val)

        while retired < max_instructions:
            idx = (pc - base) >> 2
            if idx < 0 or idx >= len(words):
                break
            ins = isa.decode(words[idx])
            if ins.kind == SYSTEM:
                halted = True
                break

            a, b = x[ins.rs1], x[ins.rs2]
            stall = 0

            # Load-use interlock: the previous load's destination is read now.
            if pending_load_rd > 0 and pending_load_rd in (ins.rs1, ins.rs2):
                stall += cfg.load_use_penalty
            pending_load_rd = -1

            if cfg.icache_enabled and not self.icache.access(pc):
                stall += cfg.icache.miss_penalty

            result = 0
            next_pc = pc + 4
            activity = cfg.w_hw_operand * (popcount(a) + popcount(b))

            if ins.kind in (ALU, MULDIV):
                if ins.name in ("lui", "auipc"):
                    result = isa.u32(ins.imm + (pc if ins.name == "auipc" else 0))
                else:
                    operand_b = b if ins.kind == MULDIV or ins.rs2 or ins.name in (
                        "add", "sub", "sll", "slt", "sltu", "xor", "srl", "sra", "or", "and"
                    ) else ins.imm
                    if ins.name.endswith("i") and ins.name not in ("mulhi",):
                        operand_b = ins.imm
                    if ins.name in ("slli", "srli", "srai"):
                        operand_b = ins.imm
                    result = isa.ALU_OPS[ins.name](a, operand_b)
                if ins.kind == MULDIV:
                    stall += (
                        cfg.mul_latency - 1 if ins.name.startswith("mul")
                        else self._div_cycles(a, b) - 1
                    )
                if ins.rd:
                    x[ins.rd] = result

            elif ins.kind == LOAD:
                addr = isa.u32(a + ins.imm) % len(mem)
                width = isa.LOAD_WIDTH[ins.name]
                result = self._load(mem, addr, width, ins.name in isa.LOAD_SIGNED)
                hit = self.dcache.access(addr)
                if not hit:
                    stall += cfg.dcache.miss_penalty
                    activity += cfg.w_miss_energy
                activity += cfg.w_hw_address * popcount(addr ^ prev_addr)
                activity += cfg.w_hw_memdata * popcount(result)
                prev_addr = addr
                if ins.rd:
                    x[ins.rd] = result
                    pending_load_rd = ins.rd

            elif ins.kind == STORE:
                addr = isa.u32(a + ins.imm) % len(mem)
                width = isa.STORE_WIDTH[ins.name]
                self._store(mem, addr, width, b)
                if not self.dcache.access(addr):
                    stall += cfg.dcache.miss_penalty
                    activity += cfg.w_miss_energy
                activity += cfg.w_hw_address * popcount(addr ^ prev_addr)
                activity += cfg.w_hw_memdata * popcount(b)
                prev_addr = addr
                result = b

            elif ins.kind == BRANCH:
                taken = isa.BRANCH_OPS[ins.name](a, b)
                if not self.bpred.update(pc, taken):
                    stall += cfg.mispredict_penalty
                if taken:
                    next_pc = isa.u32(pc + ins.imm)
                result = isa.u32(next_pc)

            elif ins.kind == JUMP:
                link = pc + 4
                next_pc = (
                    isa.u32(pc + ins.imm) if ins.name == "jal"
                    else isa.u32(a + ins.imm) & ~1
                )
                if ins.rd:
                    x[ins.rd] = link
                result = next_pc

            x[0] = 0
            activity += cfg.w_hd_result * popcount(result ^ prev_result)
            prev_result = result

            emit(pc, activity)
            if stall:
                emit(pc, cfg.idle_power, stall)
            retired += 1
            pc = next_pc

        return Trace(
            power=np.asarray(power, dtype=np.float64),
            pc_of_cycle=np.asarray(pcs, dtype=np.int64),
            cycles=len(power),
            retired=retired,
            dcache_hits=self.dcache.hits,
            dcache_misses=self.dcache.misses,
            branch_correct=self.bpred.correct,
            branch_wrong=self.bpred.wrong,
            halted=halted,
            regs=x,
        )
