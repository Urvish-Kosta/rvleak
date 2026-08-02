#!/usr/bin/env python3
"""Detect a leak and identify the instruction responsible.

    python examples/02_detect_a_leak.py
"""

from rvleak import campaign, report


def main() -> None:
    # Always establish the false-positive control first. Without it, a positive
    # result below would mean nothing -- a detector that always fires is
    # indistinguishable from a very sensitive one.
    print("=== Control: fixed vs fixed (no channel exists) ===")
    null = campaign.null_campaign("memcmp-early-exit", 150)
    print(null.tvla.summary())
    assert not null.tvla.leaks, "detector fired on a no-channel control"

    print("\n=== Leaky victim: early-exit tag comparison ===")
    leaky = campaign.tvla_campaign("memcmp-early-exit", 150)
    print(leaky.summary())
    print(report.t_trace_ascii(leaky.tvla.t))

    build = campaign._build("memcmp-early-exit", bytes(16), campaign.DEFAULT_KEY)
    listing = {int(line.split(":")[0], 16): line for line in build.program.disassemble()}
    print("\nLeaking instructions:")
    for a in leaky.attributions[:4]:
        print(f"  |t|={a.peak_t:8.2f}  {listing.get(a.pc, hex(a.pc))}")

    print("\n=== Hardened counterpart: OR-accumulating comparison ===")
    hardened = campaign.tvla_campaign("memcmp-constant-time", 150)
    print(hardened.timing.summary())
    print(hardened.tvla.summary())
    print(
        "\nNote: the hardened version is constant-time but still leaks in power.\n"
        "Fixing control flow is not masking. Reporting it as 'secure' would be\n"
        "the more dangerous answer."
    )


if __name__ == "__main__":
    main()
