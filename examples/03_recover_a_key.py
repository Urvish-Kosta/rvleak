#!/usr/bin/env python3
"""Recover key material by correlation power analysis.

    python examples/03_recover_a_key.py
"""

from rvleak import campaign, report


def main() -> None:
    print("=== Correct configuration: HD bus model, POI selection on ===")
    good = campaign.cpa_campaign(600, target_byte=0)
    print(good.summary())
    print(report.correlation_ascii(good.result, good.true_key))

    print("\n=== Ablation: point-of-interest selection disabled ===")
    print(campaign.cpa_campaign(600, target_byte=0, use_poi=False).summary())
    print(
        "  -> converges on candidate 0x00: the plaintext load earlier in the\n"
        "     loop leaks HW(p) directly and produces a ghost peak."
    )

    print("\n=== Ablation: Hamming-weight hypothesis against a Hamming-distance bus ===")
    print(campaign.cpa_campaign(600, target_byte=3, model="hw").summary())
    print("  -> returns a near-neighbour of the true key: right attack, wrong model.")

    print("\n=== Full key ===")
    print(campaign.full_key_campaign(600, sweep_step=100).summary())


if __name__ == "__main__":
    main()
