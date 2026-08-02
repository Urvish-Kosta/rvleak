import pytest

from rvleak import isa
from rvleak.asm import AsmError, assemble


def test_labels_are_pc_relative():
    prog = assemble("""
    start:  addi x1, x0, 1
            beq  x1, x0, start
    """)
    branch = isa.decode(prog.words[1])
    assert branch.imm == -4


def test_forward_reference():
    prog = assemble("""
            j    end
            addi x1, x0, 1
    end:    ecall
    """)
    assert isa.decode(prog.words[0]).imm == 8


def test_li_small_is_one_instruction():
    assert len(assemble("li x1, 100").words) == 1


def test_li_large_expands_and_compensates_sign():
    """lui+addi is only correct if the upper part is rounded up when the low
    12 bits have bit 11 set, because addi sign-extends its immediate."""
    prog = assemble("li x1, 0x12345800")
    assert len(prog.words) == 2
    lui, addi = isa.decode(prog.words[0]), isa.decode(prog.words[1])
    assert isa.u32(lui.imm + addi.imm) == 0x12345800


def test_pseudo_instructions():
    assert isa.decode(assemble("nop").words[0]).name == "addi"
    assert isa.decode(assemble("ret").words[0]).name == "jalr"
    assert isa.decode(assemble("mv x1, x2").words[0]).rs1 == 2


def test_abi_register_names():
    prog = assemble("add a0, t0, s1")
    ins = isa.decode(prog.words[0])
    assert (ins.rd, ins.rs1, ins.rs2) == (10, 5, 9)


def test_comments_and_blank_lines_ignored():
    assert len(assemble("# nothing\n\n   addi x1, x0, 0 # trailing").words) == 1


def test_unknown_mnemonic_raises():
    with pytest.raises(AsmError):
        assemble("frobnicate x1, x2")


def test_bad_register_raises():
    with pytest.raises(AsmError):
        assemble("add x1, x2, x99")
