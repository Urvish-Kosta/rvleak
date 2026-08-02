import random

import pytest

from rvleak import isa
from rvleak.asm import assemble


def test_sign_helpers():
    assert isa.s32(0xFFFFFFFF) == -1
    assert isa.u32(-1) == 0xFFFFFFFF
    assert isa.sext(0x800, 12) == -2048


@pytest.mark.parametrize("name,a,b,expected", [
    ("add", 1, 2, 3),
    ("sub", 0, 1, 0xFFFFFFFF),
    ("sra", 0x80000000, 4, 0xF8000000),
    ("srl", 0x80000000, 4, 0x08000000),
    ("slt", 0xFFFFFFFF, 1, 1),
    ("sltu", 0xFFFFFFFF, 1, 0),
    ("mulh", 0xFFFFFFFF, 0xFFFFFFFF, 0),
    ("mulhu", 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE),
])
def test_alu_ops(name, a, b, expected):
    assert isa.ALU_OPS[name](a, b) == expected


def test_division_edge_cases():
    """RISC-V specifies exact results for divide-by-zero and signed overflow
    rather than trapping; getting these wrong is a classic silent bug."""
    assert isa.ALU_OPS["div"](5, 0) == 0xFFFFFFFF
    assert isa.ALU_OPS["rem"](5, 0) == 5
    assert isa.ALU_OPS["divu"](5, 0) == 0xFFFFFFFF
    assert isa.ALU_OPS["remu"](5, 0) == 5
    assert isa.ALU_OPS["div"](isa.u32(-(1 << 31)), 0xFFFFFFFF) == isa.u32(-(1 << 31))
    assert isa.ALU_OPS["rem"](isa.u32(-(1 << 31)), 0xFFFFFFFF) == 0


def test_signed_division_rounds_toward_zero():
    assert isa.s32(isa.ALU_OPS["div"](isa.u32(-7), 2)) == -3
    assert isa.s32(isa.ALU_OPS["rem"](isa.u32(-7), 2)) == -1


def test_decode_rejects_garbage():
    with pytest.raises(isa.DecodeError):
        isa.decode(0x0000007F)


def test_encode_decode_roundtrip():
    """Every instruction the assembler emits must decode back to the same
    mnemonic and operands -- the assembler and the decoder are independent
    implementations of the encoding, so this is a real cross-check."""
    src = """
        add  x1, x2, x3
        sub  x4, x5, x6
        addi x7, x8, -100
        slli x9, x10, 7
        srai x11, x12, 3
        lw   x13, 16(x14)
        sb   x15, -8(x16)
        lui  x17, 0xABCDE
        mul  x18, x19, x20
        divu x21, x22, x23
    """
    prog = assemble(src)
    names = [isa.decode(w).name for w in prog.words]
    assert names == ["add", "sub", "addi", "slli", "srai", "lw", "sb",
                     "lui", "mul", "divu"]
    lw = isa.decode(prog.words[5])
    assert (lw.rd, lw.rs1, lw.imm) == (13, 14, 16)
    sb = isa.decode(prog.words[6])
    assert (sb.rs1, sb.rs2, sb.imm) == (16, 15, -8)


def test_random_alu_against_reference():
    rng = random.Random(0)
    for _ in range(500):
        a, b = rng.getrandbits(32), rng.getrandbits(32)
        assert isa.ALU_OPS["add"](a, b) == (a + b) & 0xFFFFFFFF
        assert isa.ALU_OPS["xor"](a, b) == a ^ b
        assert isa.ALU_OPS["mul"](a, b) == (isa.s32(a) * isa.s32(b)) & 0xFFFFFFFF
