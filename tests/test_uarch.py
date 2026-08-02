import numpy as np
import pytest

from rvleak.asm import assemble
from rvleak.uarch import CacheConfig, Machine, ModelConfig, SetAssocCache
from rvleak.victims import (
    OUT_ADDR,
    TABLE,
    build_memcmp_constant_time,
    build_memcmp_early_exit,
    build_modexp_ladder,
    build_modexp_square_multiply,
    build_table_lookup,
)


def run(build, cfg=None, **kw):
    return Machine(cfg).run(build.program, memory=build.memory, regs=build.regs, **kw)


# --- functional correctness -------------------------------------------------
# The timing and power model is only meaningful if the machine computes the
# right answers, so every victim is checked against an independent Python
# reference before any leakage claim is made.

def test_table_lookup_computes_correct_output():
    pt = bytes(range(16))
    key = bytes([0xAA] * 16)
    b = build_table_lookup(pt, key)
    run(b)
    assert list(b.memory[OUT_ADDR:OUT_ADDR + 16]) == [TABLE[p ^ 0xAA] for p in pt]


@pytest.mark.parametrize("exponent", [0, 1, 10, 15, 0x1234, 0xFFFF])
def test_modexp_variants_match_python_pow(exponent):
    for builder in (build_modexp_square_multiply, build_modexp_ladder):
        b = builder(7, exponent, 65521)
        tr = run(b)
        assert tr.regs[5] == pow(7, exponent, 65521), builder.__name__


def test_memcmp_early_exit_result():
    ref = bytes([0x5A] * 16)
    assert run(build_memcmp_early_exit(ref, ref)).regs[13] == 1
    assert run(build_memcmp_early_exit(bytes(16), ref)).regs[13] == 0


def test_memcmp_constant_time_result_is_zero_iff_equal():
    ref = bytes([0x5A] * 16)
    assert run(build_memcmp_constant_time(ref, ref)).regs[13] == 0
    assert run(build_memcmp_constant_time(bytes(16), ref)).regs[13] != 0


def test_program_halts_on_ecall():
    tr = run(build_modexp_ladder(3, 5, 101))
    assert tr.halted


# --- timing model -----------------------------------------------------------

def test_hardened_modexp_is_constant_time():
    cycles = {run(build_modexp_ladder(7, e, 65521)).cycles
              for e in (0, 1, 0xFFFF, 0x0F0F0F0F, 0xFFFFFFFF)}
    assert len(cycles) == 1


def test_leaky_modexp_is_not_constant_time():
    cycles = {run(build_modexp_square_multiply(7, e, 65521)).cycles
              for e in (1, 0xFF, 0xFFFF)}
    assert len(cycles) > 1


def test_early_exit_time_tracks_matching_prefix():
    """Execution time must increase monotonically with the length of the
    matching prefix -- that is precisely the oracle the attack uses."""
    ref = bytes([0x5A] * 16)
    times = []
    for prefix in (0, 4, 8, 16):
        cand = ref[:prefix] + bytes([0x00] * (16 - prefix))
        times.append(run(build_memcmp_early_exit(cand, ref)).cycles)
    assert times == sorted(times)
    assert times[0] < times[-1]


def test_load_use_penalty_costs_a_cycle():
    src_dep = "li x1, 0x100\nlw x2, 0(x1)\naddi x3, x2, 1\necall"
    src_indep = "li x1, 0x100\nlw x2, 0(x1)\naddi x3, x0, 1\necall"
    cfg = ModelConfig()
    cfg.load_use_penalty = 1
    m = Machine(cfg)
    dep = m.run(assemble(src_dep), mem_size=1 << 12).cycles
    indep = m.run(assemble(src_indep), mem_size=1 << 12).cycles
    assert dep - indep == 1


def test_branch_mispredict_penalty_is_applied():
    # The PHT starts weakly not-taken, so a loop that is actually taken must
    # mispredict at least once before the counter saturates.
    src = "li x1, 0\nli x2, 5\nloop: addi x1, x1, 1\nblt x1, x2, loop\necall"
    cheap = Machine(ModelConfig(mispredict_penalty=0)).run(assemble(src)).cycles
    dear = Machine(ModelConfig(mispredict_penalty=10)).run(assemble(src)).cycles
    assert dear > cheap


def test_data_dependent_divider_leaks_only_when_enabled():
    src = "divu x3, x1, x2\necall"
    prog = assemble(src)
    for flag, expect_equal in ((False, True), (True, False)):
        cfg = ModelConfig(div_data_dependent=flag)
        m = Machine(cfg)
        small = m.run(prog, regs={1: 3, 2: 7}).cycles
        large = m.run(prog, regs={1: 0xF0000000, 2: 7}).cycles
        assert (small == large) is expect_equal


# --- trace invariants -------------------------------------------------------

def test_trace_lengths_and_attribution_agree():
    tr = run(build_table_lookup(bytes(16), bytes(16)))
    assert len(tr.power) == len(tr.pc_of_cycle) == tr.cycles
    assert tr.retired > 0 and tr.ipc <= 1.0


def test_every_attributed_pc_is_within_the_program():
    b = build_table_lookup(bytes(16), bytes(16))
    tr = run(b)
    limit = b.program.base + 4 * len(b.program.words)
    assert tr.pc_of_cycle.min() >= b.program.base
    assert tr.pc_of_cycle.max() < limit


def test_noise_seed_changes_trace_but_not_timing():
    """Different noise realisations must not perturb the cycle count; if they
    did, the timing and power channels would be confounded."""
    b = build_table_lookup(bytes(16), bytes(16))
    a = Machine().run(b.program, memory=bytearray(b.memory), regs=b.regs, trace_seed="a")
    c = Machine().run(b.program, memory=bytearray(b.memory), regs=b.regs, trace_seed="b")
    assert a.cycles == c.cycles
    assert not np.array_equal(a.power, c.power)


def test_identical_seed_is_reproducible():
    b = build_table_lookup(bytes(16), bytes(16))
    a = Machine().run(b.program, memory=bytearray(b.memory), regs=b.regs, trace_seed="x")
    c = Machine().run(b.program, memory=bytearray(b.memory), regs=b.regs, trace_seed="x")
    assert np.array_equal(a.power, c.power)


# --- cache ------------------------------------------------------------------

def test_cache_hit_after_miss():
    c = SetAssocCache(CacheConfig(sets=4, ways=2, line_bytes=16))
    assert c.access(0x100) is False
    assert c.access(0x104) is True
    assert (c.hits, c.misses) == (1, 1)


def test_cache_lru_eviction():
    c = SetAssocCache(CacheConfig(sets=1, ways=2, line_bytes=16))
    c.access(0x00)
    c.access(0x10)
    c.access(0x20)          # evicts 0x00
    assert c.access(0x00) is False
    assert c.access(0x20) is True


def test_preloading_removes_misses():
    pt, key = bytes(range(16)), bytes([0x11] * 16)
    plain = run(build_table_lookup(pt, key))
    from rvleak.victims import build_table_lookup_preloaded
    warm = run(build_table_lookup_preloaded(pt, key))
    assert warm.cycles > plain.cycles       # the warm-up loop costs cycles
    assert warm.dcache_misses <= plain.dcache_misses + 8
