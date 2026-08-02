"""Victim programs.

Every victim ships as a *pair*: a naive implementation with a known leak, and a
hardened counterpart implementing the textbook mitigation. The pairs are the
tool's own regression suite. A leakage detector that only ever fires is useless;
the hardened variants are what demonstrate that a negative result means
something, and they are asserted in the test suite.

Memory map (flat, 64 KiB):
    0x1000  plaintext (16 B)
    0x1100  key       (16 B)
    0x1200  output    (16 B)
    0x1400  candidate buffer (comparison victims)
    0x1500  reference buffer (comparison victims)
    0x2000  256-byte lookup table
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from .asm import Program, assemble

PT_ADDR = 0x1000
KEY_ADDR = 0x1100
OUT_ADDR = 0x1200
CAND_ADDR = 0x1400
REF_ADDR = 0x1500
TABLE_ADDR = 0x2000
MEM_SIZE = 1 << 16

#: A fixed byte permutation standing in for an AES S-box / T-table. The actual
#: contents are irrelevant to the channel -- the leak is in the *index*, not the
#: value -- so a seeded permutation avoids shipping any specific cipher table.
TABLE = list(range(256))
random.Random(1234).shuffle(TABLE)


@dataclass
class Build:
    program: Program
    memory: bytearray
    regs: dict[int, int]


@dataclass
class Victim:
    name: str
    description: str
    hardened: bool
    build: Callable[..., Build]
    secret_bytes: int = 16


def _blank_memory() -> bytearray:
    mem = bytearray(MEM_SIZE)
    mem[TABLE_ADDR:TABLE_ADDR + 256] = bytes(TABLE)
    return mem


# --- 1. Table-lookup victim (cache-index / address-bus leakage) --------------

_TABLE_LOOKUP_SRC = """
        li   a0, {pt}
        li   a1, {key}
        li   a2, {tbl}
        li   a3, {out}
        li   t0, 0
        li   s1, {n}
loop:   add  t1, a0, t0
        lbu  t1, 0(t1)
        add  t2, a1, t0
        lbu  t2, 0(t2)
        xor  t3, t1, t2
        add  t3, a2, t3
poi:    lbu  t4, 0(t3)
        add  t5, a3, t0
        sb   t4, 0(t5)
        addi t0, t0, 1
        blt  t0, s1, loop
        ecall
"""


def build_table_lookup(plaintext: bytes, key: bytes, n: int = 16) -> Build:
    """S-box style lookup: index = plaintext[i] ^ key[i].

    Leaks twice over: the address bus carries HW(p^k) directly, and the cache
    set index is a function of the top bits of p^k, so hit/miss timing carries
    the rest. This is the mechanism behind Bernstein's 2005 cache-timing attack
    on AES and its descendants.
    """
    src = _TABLE_LOOKUP_SRC.format(
        pt=hex(PT_ADDR), key=hex(KEY_ADDR), tbl=hex(TABLE_ADDR),
        out=hex(OUT_ADDR), n=n,
    )
    mem = _blank_memory()
    mem[PT_ADDR:PT_ADDR + len(plaintext)] = plaintext
    mem[KEY_ADDR:KEY_ADDR + len(key)] = key
    return Build(assemble(src), mem, {})


_TABLE_PRELOADED_SRC = """
        li   a2, {tbl}
        li   t0, 0
        li   s1, 256
warm:   add  t1, a2, t0
        lbu  t1, 0(t1)
        addi t0, t0, {line}
        blt  t0, s1, warm
""" + _TABLE_LOOKUP_SRC


def build_table_lookup_preloaded(plaintext: bytes, key: bytes, n: int = 16,
                                 line: int = 32) -> Build:
    """Same lookup, but the whole table is walked into the cache first.

    This is the standard "preload the table" mitigation. It removes the *cache*
    channel while leaving the address-bus channel untouched -- a partial fix
    that is often mistaken for a complete one, which is precisely why it is
    worth being able to demonstrate the difference.
    """
    src = _TABLE_PRELOADED_SRC.format(
        pt=hex(PT_ADDR), key=hex(KEY_ADDR), tbl=hex(TABLE_ADDR),
        out=hex(OUT_ADDR), n=n, line=line,
    )
    mem = _blank_memory()
    mem[PT_ADDR:PT_ADDR + len(plaintext)] = plaintext
    mem[KEY_ADDR:KEY_ADDR + len(key)] = key
    return Build(assemble(src), mem, {})


# --- 2. Modular exponentiation (control-flow / timing leakage) ---------------

_SQUARE_MULTIPLY_SRC = """
        li   t0, 1
        mv   t1, a0
        mv   t2, a1
loop:   beqz t2, done
        andi t3, t2, 1
        beqz t3, skip
        mul  t0, t0, t1
        remu t0, t0, a2
skip:   mul  t1, t1, t1
        remu t1, t1, a2
        srli t2, t2, 1
        j    loop
done:   ecall
"""

_LADDER_SRC = """
        li   t0, 1
        mv   t1, a0
        li   s1, 0
loop:   li   s2, 32
        bge  s1, s2, done
        li   s2, 31
        sub  s2, s2, s1
        srl  t3, a1, s2
        andi t3, t3, 1
        sub  t4, x0, t3
        mul  t5, t0, t1
        remu t5, t5, a2
        mul  t6, t0, t0
        remu t6, t6, a2
        mul  s3, t1, t1
        remu s3, s3, a2
        not  s4, t4
        and  s5, t6, s4
        and  s6, t5, t4
        or   t0, s5, s6
        and  s5, t5, s4
        and  s6, s3, t4
        or   t1, s5, s6
        addi s1, s1, 1
        j    loop
done:   ecall
"""


def build_modexp_square_multiply(base: int, exponent: int, modulus: int) -> Build:
    """Right-to-left square-and-multiply. Iteration count depends on the
    position of the exponent's most significant set bit, and each set bit costs
    one extra multiply plus one extra modulo. Total cycles therefore encode both
    the bit length and the Hamming weight of the secret exponent."""
    return Build(assemble(_SQUARE_MULTIPLY_SRC), _blank_memory(),
                 {10: base, 11: exponent, 12: modulus})


def build_modexp_ladder(base: int, exponent: int, modulus: int) -> Build:
    """Branchless fixed-32-iteration ladder with arithmetic-mask selection.

    Both candidate results are always computed and the secret bit only steers a
    mask, so the instruction sequence is identical for every exponent."""
    return Build(assemble(_LADDER_SRC), _blank_memory(),
                 {10: base, 11: exponent, 12: modulus})


# --- 3. Tag comparison (early-exit leakage) ---------------------------------

_MEMCMP_EARLY_SRC = """
        li   a0, {cand}
        li   a1, {ref}
        li   t0, 0
        li   s1, {n}
        li   a3, 1
loop:   bge  t0, s1, done
        add  t1, a0, t0
        lbu  t1, 0(t1)
        add  t2, a1, t0
        lbu  t2, 0(t2)
        bne  t1, t2, fail
        addi t0, t0, 1
        j    loop
fail:   li   a3, 0
done:   ecall
"""

_MEMCMP_CT_SRC = """
        li   a0, {cand}
        li   a1, {ref}
        li   t0, 0
        li   s1, {n}
        li   a3, 0
loop:   bge  t0, s1, done
        add  t1, a0, t0
        lbu  t1, 0(t1)
        add  t2, a1, t0
        lbu  t2, 0(t2)
        xor  t3, t1, t2
        or   a3, a3, t3
        addi t0, t0, 1
        j    loop
done:   ecall
"""


def build_memcmp_early_exit(candidate: bytes, reference: bytes) -> Build:
    """Byte-by-byte tag comparison that returns on the first mismatch. The
    classic MAC-verification oracle: execution time reveals the length of the
    matching prefix, which reduces forgery from exponential to linear work."""
    src = _MEMCMP_EARLY_SRC.format(cand=hex(CAND_ADDR), ref=hex(REF_ADDR),
                                   n=len(reference))
    mem = _blank_memory()
    mem[CAND_ADDR:CAND_ADDR + len(candidate)] = candidate
    mem[REF_ADDR:REF_ADDR + len(reference)] = reference
    return Build(assemble(src), mem, {})


def build_memcmp_constant_time(candidate: bytes, reference: bytes) -> Build:
    """OR-accumulating comparison: always touches every byte, result is the
    accumulated XOR difference. Control flow is independent of the data."""
    src = _MEMCMP_CT_SRC.format(cand=hex(CAND_ADDR), ref=hex(REF_ADDR),
                                n=len(reference))
    mem = _blank_memory()
    mem[CAND_ADDR:CAND_ADDR + len(candidate)] = candidate
    mem[REF_ADDR:REF_ADDR + len(reference)] = reference
    return Build(assemble(src), mem, {})


REGISTRY: dict[str, Victim] = {
    "table-lookup": Victim(
        "table-lookup",
        "S-box style table lookup indexed by plaintext ^ key",
        hardened=False, build=build_table_lookup,
    ),
    "table-lookup-preloaded": Victim(
        "table-lookup-preloaded",
        "Same lookup with the table preloaded into cache (partial mitigation)",
        hardened=True, build=build_table_lookup_preloaded,
    ),
    "modexp-square-multiply": Victim(
        "modexp-square-multiply",
        "Square-and-multiply modular exponentiation (secret-dependent branch)",
        hardened=False, build=build_modexp_square_multiply, secret_bytes=4,
    ),
    "modexp-ladder": Victim(
        "modexp-ladder",
        "Branchless masked ladder modular exponentiation",
        hardened=True, build=build_modexp_ladder, secret_bytes=4,
    ),
    "memcmp-early-exit": Victim(
        "memcmp-early-exit",
        "Tag comparison returning on first mismatch",
        hardened=False, build=build_memcmp_early_exit,
    ),
    "memcmp-constant-time": Victim(
        "memcmp-constant-time",
        "OR-accumulating constant-time tag comparison",
        hardened=True, build=build_memcmp_constant_time,
    ),
}
