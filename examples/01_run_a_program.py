#!/usr/bin/env python3
"""Assemble a program, run it, and inspect the cycle-level trace.

    python examples/01_run_a_program.py
"""

from rvleak import Machine, ModelConfig, assemble

SRC = """
        li   a0, 0x1000        # base address of a small array
        li   t0, 0             # index
        li   t1, 16            # count
        li   a1, 0             # accumulator
loop:   slli t2, t0, 2
        add  t2, a0, t2
        lw   t3, 0(t2)
        add  a1, a1, t3
        addi t0, t0, 1
        blt  t0, t1, loop
        ecall
"""


def main() -> None:
    program = assemble(SRC)
    print("Disassembly")
    for line in program.disassemble():
        print(" ", line)

    memory = bytearray(1 << 16)
    for i in range(16):
        memory[0x1000 + 4 * i: 0x1004 + 4 * i] = (i * 3).to_bytes(4, "little")

    machine = Machine(ModelConfig(noise_sigma=0.0))
    trace = machine.run(program, memory=memory)

    print("\nExecution")
    print(f"  accumulator (a1) : {trace.regs[11]}  (expected {sum(i * 3 for i in range(16))})")
    print(f"  retired          : {trace.retired} instructions")
    print(f"  cycles           : {trace.cycles}")
    print(f"  IPC              : {trace.ipc:.3f}")
    print(f"  D-cache          : {trace.dcache_hits} hits / {trace.dcache_misses} misses")
    print(f"  branches         : {trace.branch_correct} correct / {trace.branch_wrong} wrong")

    print("\nFirst 12 cycles, with the instruction responsible for each:")
    for c in range(12):
        print(f"  cycle {c:3d}  pc={trace.pc_of_cycle[c]:#06x}  activity={trace.power[c]:6.2f}")


if __name__ == "__main__":
    main()
