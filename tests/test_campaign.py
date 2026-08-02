"""End-to-end behaviour of the tool: the claims the README makes must hold."""

import numpy as np
import pytest

from rvleak import campaign, victims
from rvleak.uarch import ModelConfig

LEAKY_TIMING = ["table-lookup", "modexp-square-multiply", "memcmp-early-exit"]
CONSTANT_TIME = ["table-lookup-preloaded", "modexp-ladder", "memcmp-constant-time"]


def test_null_control_does_not_fire():
    """The single most important test in the suite. Fixed-vs-fixed has no
    channel; if the detector reports one, every positive result is worthless."""
    res = campaign.null_campaign("table-lookup", 150)
    assert not res.tvla.leaks, res.tvla.summary()
    assert not res.timing.leaks


@pytest.mark.parametrize("victim", CONSTANT_TIME)
def test_null_control_on_every_hardened_victim(victim):
    assert not campaign.null_campaign(victim, 100).tvla.leaks


def test_within_group_variance_is_nonzero():
    """Regression test. Re-seeding the noise identically per trace drove
    within-group variance to zero and the t statistic to ~1e15, which looked
    like an overwhelming leak but was a measurement-model bug."""
    ts = campaign.collect("memcmp-constant-time", 8, fixed=True)
    m = np.stack([t[:min(map(len, ts.power))] for t in ts.power])
    assert m.var(axis=0).max() > 0.1


@pytest.mark.parametrize("victim", LEAKY_TIMING)
def test_leaky_victims_leak_in_timing(victim):
    res = campaign.tvla_campaign(victim, 100)
    assert res.timing.leaks, res.timing.summary()


@pytest.mark.parametrize("victim", CONSTANT_TIME)
def test_hardened_victims_are_constant_time(victim):
    res = campaign.tvla_campaign(victim, 100)
    assert not res.timing.leaks, res.timing.summary()


@pytest.mark.parametrize("victim", CONSTANT_TIME)
def test_constant_time_is_not_power_secure(victim):
    """A deliberately asserted negative result. Fixing control flow does not
    fix first-order power leakage, and the tool must keep saying so -- if this
    test ever starts failing, either the model or a victim has changed in a way
    that overstates the security of these mitigations."""
    res = campaign.tvla_campaign(victim, 100)
    assert res.tvla.leaks, res.tvla.summary()


def test_attribution_points_at_a_real_instruction():
    res = campaign.tvla_campaign("memcmp-early-exit", 100)
    build = campaign._build("memcmp-early-exit", bytes(16), campaign.DEFAULT_KEY)
    limit = build.program.base + 4 * len(build.program.words)
    assert res.attributions
    for a in res.attributions:
        assert build.program.base <= a.pc < limit
    assert abs(sum(a.share for a in res.attributions) - 1.0) < 1e-6


def test_cpa_recovers_key_byte_zero():
    res = campaign.cpa_campaign(400, target_byte=0)
    assert res.recovered
    assert res.rank == 0
    assert res.traces_needed is not None and res.traces_needed <= 400


def test_mismatched_leakage_model_degrades_recovery():
    """The HW model against an HD-leaking bus finds near-neighbours of the key,
    not the key. Documenting this as a test keeps the README honest about why
    the 'hd' model is the default."""
    hd = campaign.cpa_campaign(400, target_byte=3, model="hd")
    hw = campaign.cpa_campaign(400, target_byte=3, model="hw")
    assert hd.recovered
    assert hw.rank > hd.rank


def test_poi_selection_is_necessary():
    """Without attribution-guided windowing the plaintext load produces a ghost
    peak at candidate 0x00."""
    with_poi = campaign.cpa_campaign(400, target_byte=0, use_poi=True)
    without = campaign.cpa_campaign(400, target_byte=0, use_poi=False)
    assert with_poi.recovered
    assert not without.recovered


def test_full_key_recovery():
    res = campaign.full_key_campaign(600, sweep_step=200)
    assert res.n_correct == 16, res.summary()
    assert bytes(res.recovered) == campaign.DEFAULT_KEY


def test_higher_noise_needs_more_traces():
    """Monotonicity sanity check: the attack must get harder as sigma grows.
    If it did not, the noise model would not be doing anything."""
    quiet = campaign.cpa_campaign(600, cfg=ModelConfig(noise_sigma=0.5), sweep_step=50)
    loud = campaign.cpa_campaign(600, cfg=ModelConfig(noise_sigma=8.0), sweep_step=50)
    assert quiet.result.best_corr > loud.result.best_corr


def test_every_registry_victim_builds_and_runs():
    for name in victims.REGISTRY:
        ts = campaign.collect(name, 2, fixed=False)
        assert len(ts) == 2
        assert all(len(p) > 0 for p in ts.power)
