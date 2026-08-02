"""RV32IM instruction decode and functional (architectural) semantics.

This module is deliberately free of any timing or leakage concepts. It defines
*what* an instruction computes; `uarch.py` defines *when* and *at what modelled
energy cost*. Keeping the two apart means the functional model can be validated
independently of the microarchitectural model, which is the only way to be
confident that an observed timing difference is a property of the
microarchitecture and not an artifact of a broken interpreter.

Supported: the full RV32I base integer set except FENCE/EBREAK/CSR*, plus the
RV32M multiply/divide extension. ECALL is repurposed as a clean halt.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

MASK32 = 0xFFFFFFFF


def u32(x: int) -> int:
    """Truncate to an unsigned 32-bit value."""
    return x & MASK32


def s32(x: int) -> int:
    """Reinterpret the low 32 bits of `x` as a signed two's-complement value."""
    x &= MASK32
    return x - (1 << 32) if x & 0x80000000 else x


def sext(value: int, bits: int) -> int:
    """Sign-extend `bits`-wide `value` to a Python int."""
    sign = 1 << (bits - 1)
    return (value & (sign - 1)) - (value & sign)


# --- Instruction categories -------------------------------------------------
# These drive the timing model, so they are part of the decoded form rather
# than being re-derived by string matching in the pipeline.

ALU = "alu"
BRANCH = "branch"
JUMP = "jump"
LOAD = "load"
STORE = "store"
MULDIV = "muldiv"
SYSTEM = "system"


@dataclass(frozen=True)
class Insn:
    """A decoded instruction."""

    raw: int
    name: str
    kind: str
    rd: int
    rs1: int
    rs2: int
    imm: int

    def __str__(self) -> str:
        n = f"{self.name:<6}"
        if self.name in ("lui", "auipc"):
            return f"{n} x{self.rd}, {self.imm >> 12:#x}"
        if self.kind == LOAD:
            return f"{n} x{self.rd}, {self.imm}(x{self.rs1})"
        if self.kind == STORE:
            return f"{n} x{self.rs2}, {self.imm}(x{self.rs1})"
        if self.kind == BRANCH:
            return f"{n} x{self.rs1}, x{self.rs2}, {self.imm:+d}"
        if self.name == "jal":
            return f"{n} x{self.rd}, {self.imm:+d}"
        if self.name == "jalr":
            return f"{n} x{self.rd}, {self.imm}(x{self.rs1})"
        if self.name == "ecall":
            return "ecall"
        if self.name in ("slli", "srli", "srai") or (
            self.name.endswith("i") and self.kind == ALU
        ):
            return f"{n} x{self.rd}, x{self.rs1}, {self.imm}"
        return f"{n} x{self.rd}, x{self.rs1}, x{self.rs2}"


class DecodeError(ValueError):
    """Raised when a word does not correspond to a supported instruction."""


_BRANCH_F3 = {0: "beq", 1: "bne", 4: "blt", 5: "bge", 6: "bltu", 7: "bgeu"}
_LOAD_F3 = {0: "lb", 1: "lh", 2: "lw", 4: "lbu", 5: "lhu"}
_STORE_F3 = {0: "sb", 1: "sh", 2: "sw"}
_OPIMM_F3 = {0: "addi", 2: "slti", 3: "sltiu", 4: "xori", 6: "ori", 7: "andi"}
_MULDIV_F3 = {
    0: "mul",
    1: "mulh",
    2: "mulhsu",
    3: "mulhu",
    4: "div",
    5: "divu",
    6: "rem",
    7: "remu",
}


def decode(word: int) -> Insn:
    """Decode a 32-bit instruction word into an `Insn`."""
    word &= MASK32
    opcode = word & 0x7F
    rd = (word >> 7) & 0x1F
    f3 = (word >> 12) & 0x7
    rs1 = (word >> 15) & 0x1F
    rs2 = (word >> 20) & 0x1F
    f7 = (word >> 25) & 0x7F

    def mk(name, kind, *, rd_=rd, rs1_=rs1, rs2_=rs2, imm=0):
        return Insn(word, name, kind, rd_, rs1_, rs2_, imm)

    if opcode == 0x37:
        return mk("lui", ALU, rs1_=0, rs2_=0, imm=sext(word & 0xFFFFF000, 32))
    if opcode == 0x17:
        return mk("auipc", ALU, rs1_=0, rs2_=0, imm=sext(word & 0xFFFFF000, 32))
    if opcode == 0x6F:
        imm = (
            ((word >> 31) & 1) << 20
            | ((word >> 12) & 0xFF) << 12
            | ((word >> 20) & 1) << 11
            | ((word >> 21) & 0x3FF) << 1
        )
        return mk("jal", JUMP, rs1_=0, rs2_=0, imm=sext(imm, 21))
    if opcode == 0x67:
        return mk("jalr", JUMP, rs2_=0, imm=sext(word >> 20, 12))
    if opcode == 0x63:
        if f3 not in _BRANCH_F3:
            raise DecodeError(f"bad branch funct3 {f3}")
        imm = (
            ((word >> 31) & 1) << 12
            | ((word >> 7) & 1) << 11
            | ((word >> 25) & 0x3F) << 5
            | ((word >> 8) & 0xF) << 1
        )
        return mk(_BRANCH_F3[f3], BRANCH, rd_=0, imm=sext(imm, 13))
    if opcode == 0x03:
        if f3 not in _LOAD_F3:
            raise DecodeError(f"bad load funct3 {f3}")
        return mk(_LOAD_F3[f3], LOAD, rs2_=0, imm=sext(word >> 20, 12))
    if opcode == 0x23:
        if f3 not in _STORE_F3:
            raise DecodeError(f"bad store funct3 {f3}")
        imm = ((word >> 25) & 0x7F) << 5 | ((word >> 7) & 0x1F)
        return mk(_STORE_F3[f3], STORE, rd_=0, imm=sext(imm, 12))
    if opcode == 0x13:
        if f3 in _OPIMM_F3:
            return mk(_OPIMM_F3[f3], ALU, rs2_=0, imm=sext(word >> 20, 12))
        if f3 == 1:
            return mk("slli", ALU, rs2_=0, imm=rs2)
        if f3 == 5:
            return mk("srai" if f7 == 0x20 else "srli", ALU, rs2_=0, imm=rs2)
        raise DecodeError(f"bad op-imm funct3 {f3}")
    if opcode == 0x33:
        if f7 == 0x01:
            return mk(_MULDIV_F3[f3], MULDIV)
        table = {
            (0, 0x00): "add",
            (0, 0x20): "sub",
            (1, 0x00): "sll",
            (2, 0x00): "slt",
            (3, 0x00): "sltu",
            (4, 0x00): "xor",
            (5, 0x00): "srl",
            (5, 0x20): "sra",
            (6, 0x00): "or",
            (7, 0x00): "and",
        }
        if (f3, f7) not in table:
            raise DecodeError(f"bad op funct3/7 {f3}/{f7:#x}")
        return mk(table[(f3, f7)], ALU)
    if opcode == 0x73:
        return mk("ecall", SYSTEM, rd_=0, rs1_=0, rs2_=0)
    raise DecodeError(f"unsupported opcode {opcode:#04x} in word {word:#010x}")


# --- ALU semantics ----------------------------------------------------------

def _div(a: int, b: int) -> int:
    a, b = s32(a), s32(b)
    if b == 0:
        return MASK32
    if a == -(1 << 31) and b == -1:
        return u32(a)
    q = abs(a) // abs(b)
    return u32(-q if (a < 0) != (b < 0) else q)


def _rem(a: int, b: int) -> int:
    a, b = s32(a), s32(b)
    if b == 0:
        return u32(a)
    if a == -(1 << 31) and b == -1:
        return 0
    r = abs(a) % abs(b)
    return u32(-r if a < 0 else r)


ALU_OPS: dict[str, Callable[[int, int], int]] = {
    "add": lambda a, b: u32(a + b),
    "addi": lambda a, b: u32(a + b),
    "sub": lambda a, b: u32(a - b),
    "sll": lambda a, b: u32(a << (b & 31)),
    "slli": lambda a, b: u32(a << (b & 31)),
    "srl": lambda a, b: u32(a) >> (b & 31),
    "srli": lambda a, b: u32(a) >> (b & 31),
    "sra": lambda a, b: u32(s32(a) >> (b & 31)),
    "srai": lambda a, b: u32(s32(a) >> (b & 31)),
    "slt": lambda a, b: int(s32(a) < s32(b)),
    "slti": lambda a, b: int(s32(a) < s32(b)),
    "sltu": lambda a, b: int(u32(a) < u32(b)),
    "sltiu": lambda a, b: int(u32(a) < u32(b)),
    "xor": lambda a, b: u32(a ^ b),
    "xori": lambda a, b: u32(a ^ b),
    "or": lambda a, b: u32(a | b),
    "ori": lambda a, b: u32(a | b),
    "and": lambda a, b: u32(a & b),
    "andi": lambda a, b: u32(a & b),
    "mul": lambda a, b: u32(s32(a) * s32(b)),
    "mulh": lambda a, b: u32((s32(a) * s32(b)) >> 32),
    "mulhu": lambda a, b: u32((u32(a) * u32(b)) >> 32),
    "mulhsu": lambda a, b: u32((s32(a) * u32(b)) >> 32),
    "div": _div,
    "divu": lambda a, b: MASK32 if u32(b) == 0 else u32(a) // u32(b),
    "rem": _rem,
    "remu": lambda a, b: u32(a) if u32(b) == 0 else u32(a) % u32(b),
}

BRANCH_OPS: dict[str, Callable[[int, int], bool]] = {
    "beq": lambda a, b: u32(a) == u32(b),
    "bne": lambda a, b: u32(a) != u32(b),
    "blt": lambda a, b: s32(a) < s32(b),
    "bge": lambda a, b: s32(a) >= s32(b),
    "bltu": lambda a, b: u32(a) < u32(b),
    "bgeu": lambda a, b: u32(a) >= u32(b),
}

LOAD_WIDTH = {"lb": 1, "lbu": 1, "lh": 2, "lhu": 2, "lw": 4}
STORE_WIDTH = {"sb": 1, "sh": 2, "sw": 4}
LOAD_SIGNED = {"lb", "lh", "lw"}
