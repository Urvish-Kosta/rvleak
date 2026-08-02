"""A minimal two-pass RV32IM assembler.

Rationale: the victims analysed by rvleak are 20-100 instructions long. Taking a
hard dependency on a riscv32 GCC/LLVM cross-toolchain to build them would make
the tool unreproducible for anyone who does not already have one installed, and
would make CI slow and fragile. A ~200-line assembler removes that dependency
entirely; anything larger than these victims should be built with a real
toolchain and loaded as a flat binary via `Program.from_words`.

Supported syntax:
    label:                      define a label
    op rd, rs1, rs2             R-type
    op rd, rs1, imm             I-type (imm may be a label for branches/jal)
    op rd, imm(rs1)             loads
    op rs2, imm(rs1)            stores
    .word 0xdeadbeef            raw datum
Pseudo-instructions: nop, li (12-bit range or lui+addi), mv, j, ret, beqz, bnez.
Comments start with '#'. Registers accept x0-x31 and the standard ABI names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .isa import MASK32, decode

ABI = {
    "zero": 0, "ra": 1, "sp": 2, "gp": 3, "tp": 4, "t0": 5, "t1": 6, "t2": 7,
    "s0": 8, "fp": 8, "s1": 9, "a0": 10, "a1": 11, "a2": 12, "a3": 13,
    "a4": 14, "a5": 15, "a6": 16, "a7": 17, "s2": 18, "s3": 19, "s4": 20,
    "s5": 21, "s6": 22, "s7": 23, "s8": 24, "s9": 25, "s10": 26, "s11": 27,
    "t3": 28, "t4": 29, "t5": 30, "t6": 31,
}

R_TYPE = {
    "add": (0x33, 0, 0x00), "sub": (0x33, 0, 0x20), "sll": (0x33, 1, 0x00),
    "slt": (0x33, 2, 0x00), "sltu": (0x33, 3, 0x00), "xor": (0x33, 4, 0x00),
    "srl": (0x33, 5, 0x00), "sra": (0x33, 5, 0x20), "or": (0x33, 6, 0x00),
    "and": (0x33, 7, 0x00), "mul": (0x33, 0, 0x01), "mulh": (0x33, 1, 0x01),
    "mulhsu": (0x33, 2, 0x01), "mulhu": (0x33, 3, 0x01), "div": (0x33, 4, 0x01),
    "divu": (0x33, 5, 0x01), "rem": (0x33, 6, 0x01), "remu": (0x33, 7, 0x01),
}
I_TYPE = {
    "addi": 0, "slti": 2, "sltiu": 3, "xori": 4, "ori": 6, "andi": 7,
}
SHIFT_I = {"slli": (1, 0x00), "srli": (5, 0x00), "srai": (5, 0x20)}
LOADS = {"lb": 0, "lh": 1, "lw": 2, "lbu": 4, "lhu": 5}
STORES = {"sb": 0, "sh": 1, "sw": 2}
BRANCHES = {"beq": 0, "bne": 1, "blt": 4, "bge": 5, "bltu": 6, "bgeu": 7}


class AsmError(SyntaxError):
    """Raised for malformed assembly input."""


def reg(token: str) -> int:
    t = token.strip().lower()
    if t in ABI:
        return ABI[t]
    if re.fullmatch(r"x([0-9]|[12][0-9]|3[01])", t):
        return int(t[1:])
    raise AsmError(f"not a register: {token!r}")


def _imm(token: str, labels: dict[str, int], pc: int, *, relative: bool) -> int:
    t = token.strip()
    if t in labels:
        return labels[t] - pc if relative else labels[t]
    try:
        return int(t, 0)
    except ValueError as exc:
        raise AsmError(f"not an immediate or known label: {token!r}") from exc


@dataclass
class Program:
    """An assembled program: a word image plus its symbol table."""

    words: list[int]
    labels: dict[str, int] = field(default_factory=dict)
    base: int = 0

    @classmethod
    def from_words(cls, words, base: int = 0) -> Program:
        return cls([w & MASK32 for w in words], {}, base)

    def disassemble(self) -> list[str]:
        out = []
        for i, w in enumerate(self.words):
            try:
                out.append(f"{self.base + 4 * i:08x}: {decode(w)}")
            except Exception:
                out.append(f"{self.base + 4 * i:08x}: .word {w:#010x}")
        return out


def _split_ops(rest: str) -> list[str]:
    return [p.strip() for p in rest.split(",")] if rest.strip() else []


def _mem_operand(tok: str) -> tuple[str, str]:
    m = re.fullmatch(r"\s*(-?\w+)?\s*\(\s*(\w+)\s*\)\s*", tok)
    if not m:
        raise AsmError(f"bad memory operand {tok!r}")
    return (m.group(1) or "0"), m.group(2)


def _encode_r(op, rd, rs1, rs2):
    opcode, f3, f7 = R_TYPE[op]
    return (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | opcode


def _encode_i(opcode, f3, rd, rs1, imm):
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | opcode


def _encode_s(f3, rs1, rs2, imm):
    imm &= 0xFFF
    return (
        ((imm >> 5) << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12)
        | ((imm & 0x1F) << 7) | 0x23
    )


def _encode_b(f3, rs1, rs2, imm):
    imm &= 0x1FFF
    return (
        (((imm >> 12) & 1) << 31) | (((imm >> 5) & 0x3F) << 25) | (rs2 << 20)
        | (rs1 << 15) | (f3 << 12) | (((imm >> 1) & 0xF) << 8)
        | (((imm >> 11) & 1) << 7) | 0x63
    )


def _encode_j(rd, imm):
    imm &= 0x1FFFFF
    return (
        (((imm >> 20) & 1) << 31) | (((imm >> 1) & 0x3FF) << 21)
        | (((imm >> 11) & 1) << 20) | (((imm >> 12) & 0xFF) << 12)
        | (rd << 7) | 0x6F
    )


# Pseudo-instructions are expanded before pass 1 so that label offsets are
# computed against the real, post-expansion instruction count.
def _expand(op: str, ops: list[str]) -> list[tuple[str, list[str]]]:
    if op == "nop":
        return [("addi", ["x0", "x0", "0"])]
    if op == "mv":
        return [("addi", [ops[0], ops[1], "0"])]
    if op == "not":
        return [("xori", [ops[0], ops[1], "-1"])]
    if op == "neg":
        return [("sub", [ops[0], "x0", ops[1]])]
    if op == "j":
        return [("jal", ["x0", ops[0]])]
    if op == "call":
        return [("jal", ["ra", ops[0]])]
    if op == "ret":
        return [("jalr", ["x0", "ra", "0"])]
    if op == "beqz":
        return [("beq", [ops[0], "x0", ops[1]])]
    if op == "bnez":
        return [("bne", [ops[0], "x0", ops[1]])]
    if op == "li":
        value = int(ops[1], 0) & MASK32
        if -2048 <= (value - (1 << 32) if value & 0x80000000 else value) < 2048:
            return [("addi", [ops[0], "x0", ops[1]])]
        # lui takes the upper 20 bits; addi's sign-extended immediate must be
        # compensated for by rounding the upper part up when bit 11 is set.
        lo = value & 0xFFF
        hi = (value + 0x800) & 0xFFFFF000
        seq = [("lui", [ops[0], hex(hi >> 12)])]
        if lo:
            seq.append(("addi", [ops[0], ops[0], str(lo - 4096 if lo & 0x800 else lo)]))
        return seq
    return [(op, ops)]


def assemble(source: str, base: int = 0) -> Program:
    """Assemble `source` into a `Program` loaded at `base`."""
    items: list[tuple[str, list[str]] | tuple[str, list[str], str]] = []
    labels: dict[str, int] = {}
    pc = base

    for lineno, raw in enumerate(source.splitlines(), 1):
        line = raw.split("#")[0].strip()
        while line:
            m = re.match(r"^([A-Za-z_.$][\w.$]*):\s*", line)
            if not m:
                break
            labels[m.group(1)] = pc
            line = line[m.end():].strip()
        if not line:
            continue
        parts = line.split(None, 1)
        op = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""
        if op == ".word":
            items.append((".word", [rest.strip()]))
            pc += 4
            continue
        try:
            expansion = _expand(op, _split_ops(rest))
        except (IndexError, ValueError) as exc:
            raise AsmError(f"line {lineno}: bad operands for {op!r}") from exc
        for eop, eops in expansion:
            items.append((eop, eops))
            pc += 4

    words: list[int] = []
    pc = base
    for op, ops in items:
        try:
            words.append(_encode(op, ops, labels, pc))
        except AsmError:
            raise
        except Exception as exc:
            raise AsmError(f"cannot encode {op} {', '.join(ops)}: {exc}") from exc
        pc += 4
    return Program(words, labels, base)


def _encode(op: str, ops: list[str], labels: dict[str, int], pc: int) -> int:
    if op == ".word":
        return _imm(ops[0], labels, pc, relative=False) & MASK32
    if op in R_TYPE:
        return _encode_r(op, reg(ops[0]), reg(ops[1]), reg(ops[2]))
    if op in I_TYPE:
        return _encode_i(0x13, I_TYPE[op], reg(ops[0]), reg(ops[1]),
                         _imm(ops[2], labels, pc, relative=False))
    if op in SHIFT_I:
        f3, f7 = SHIFT_I[op]
        sh = _imm(ops[2], labels, pc, relative=False) & 0x1F
        return _encode_i(0x13, f3, reg(ops[0]), reg(ops[1]), (f7 << 5) | sh)
    if op in LOADS:
        imm, rs1 = _mem_operand(ops[1])
        return _encode_i(0x03, LOADS[op], reg(ops[0]), reg(rs1),
                         _imm(imm, labels, pc, relative=False))
    if op in STORES:
        imm, rs1 = _mem_operand(ops[1])
        return _encode_s(STORES[op], reg(rs1), reg(ops[0]),
                         _imm(imm, labels, pc, relative=False))
    if op in BRANCHES:
        return _encode_b(BRANCHES[op], reg(ops[0]), reg(ops[1]),
                         _imm(ops[2], labels, pc, relative=True))
    if op in ("lui", "auipc"):
        upper = (_imm(ops[1], labels, pc, relative=False) & 0xFFFFF) << 12
        return upper | (reg(ops[0]) << 7) | (0x37 if op == "lui" else 0x17)
    if op == "jal":
        return _encode_j(reg(ops[0]), _imm(ops[1], labels, pc, relative=True))
    if op == "jalr":
        if len(ops) == 2:
            imm, rs1 = _mem_operand(ops[1])
            return _encode_i(0x67, 0, reg(ops[0]), reg(rs1),
                             _imm(imm, labels, pc, relative=False))
        return _encode_i(0x67, 0, reg(ops[0]), reg(ops[1]),
                         _imm(ops[2], labels, pc, relative=False))
    if op == "ecall":
        return 0x73
    raise AsmError(f"unknown mnemonic {op!r}")
