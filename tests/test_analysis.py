import numpy as np
import pytest

from rvleak import analysis


def test_align_pad_preserves_all_samples():
    m = analysis.align([np.ones(3), np.ones(5)], mode="pad")
    assert m.shape == (2, 5)
    assert m[0, 3] == 0.0


def test_align_truncate_cuts_to_shortest():
    m = analysis.align([np.ones(3), np.ones(5)], mode="truncate")
    assert m.shape == (2, 3)


def test_align_rejects_unknown_mode():
    with pytest.raises(ValueError):
        analysis.align([np.ones(2)], mode="stretch")


def test_welch_t_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(200, 20))
    b = rng.normal(size=(200, 20))
    assert np.abs(analysis.welch_t(a, b)).max() < 4.5


def test_welch_t_detects_a_shifted_group():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(200, 20))
    b = rng.normal(size=(200, 20))
    b[:, 7] += 2.0
    t = analysis.welch_t(a, b)
    assert int(np.argmax(np.abs(t))) == 7
    assert np.abs(t[7]) > 4.5


def test_zero_variance_with_different_means_is_infinite_not_zero():
    """Regression test. Two deterministic groups with different means are
    perfectly separable -- the strongest possible leak. An earlier version
    passed this through nan_to_num and reported t = 0, i.e. 'clean'."""
    a = np.full((10, 3), 1.0)
    b = np.full((10, 3), 2.0)
    t = analysis.welch_t(a, b)
    assert np.isinf(t).all()

    tr = analysis.timing_test([100] * 10, [50] * 10)
    assert np.isinf(tr.t) and tr.leaks


def test_zero_variance_with_equal_means_is_zero():
    t = analysis.welch_t(np.ones((5, 3)), np.ones((5, 3)))
    assert np.all(t == 0.0)
    assert not analysis.timing_test([10] * 5, [10] * 5).leaks


def test_cpa_recovers_a_synthetic_key():
    """Ground-truth check of the CPA implementation itself, independent of the
    simulator: build traces whose leakage is HW(p ^ k) by construction."""
    rng = np.random.default_rng(3)
    key = 0x5C
    pts = rng.integers(0, 256, size=400)
    traces = []
    for p in pts:
        t = rng.normal(0, 1.0, size=12)
        t[5] += analysis.HW[int(p) ^ key]
        traces.append(t)
    hyp = analysis.hw_hypotheses(pts)
    res = analysis.cpa(traces, hyp)
    assert res.best == key
    assert res.sample == 5
    assert res.rank_of(key) == 0


def test_complement_ambiguity_is_reported_and_resolved():
    """HW(p ^ ~k) = 8 - HW(p ^ k), so |rho| cannot separate a key from its
    complement. Signed ranking must, and the tie must be flagged."""
    rng = np.random.default_rng(4)
    key = 0x5C
    pts = rng.integers(0, 256, size=400)
    traces = [np.array([rng.normal(0, 0.5) + analysis.HW[int(p) ^ key]]) for p in pts]
    hyp = analysis.hw_hypotheses(pts)

    unsigned = analysis.cpa(traces, hyp, signed=False)
    assert unsigned.best in (key, key ^ 0xFF)
    assert unsigned.complement_tie

    signed = analysis.cpa(traces, hyp, signed=True)
    assert signed.best == key


def test_hd_hypotheses_match_a_direct_computation():
    pts = [0x00, 0xFF, 0x3C]
    h = analysis.hd_hypotheses(pts, 0x2000, 0x1100)
    for i, p in enumerate(pts):
        for k in (0, 1, 0x2B):
            expected = bin((0x2000 + (p ^ k)) ^ 0x1100).count("1")
            assert h[i, k] == expected


def test_extract_windows_aligns_on_the_instruction_not_the_clock():
    """Regression test for the alignment failure. Two traces execute the same
    instruction at different absolute positions; index-based selection would
    pick up the wrong samples, per-trace extraction must not."""
    power = [np.array([0.0, 1.0, 2.0, 9.0, 9.0, 3.0]),
             np.array([0.0, 0.0, 0.0, 1.0, 2.0, 9.0, 9.0])]
    pcs = [np.array([0, 4, 4, 8, 8, 12]),
           np.array([0, 0, 0, 4, 4, 8, 8])]
    w = analysis.extract_windows(power, pcs, pc=8, occurrence=0)
    assert w.shape == (2, 2)
    assert np.array_equal(w, np.array([[9.0, 9.0], [9.0, 9.0]]))


def test_extract_windows_handles_missing_occurrence():
    w = analysis.extract_windows([np.ones(4)], [np.zeros(4)], pc=99, occurrence=0)
    assert w.shape == (1, 1)


def test_traces_to_recover_returns_none_when_never_stable():
    rng = np.random.default_rng(5)
    pts = rng.integers(0, 256, size=60)
    traces = [rng.normal(size=4) for _ in pts]      # pure noise, no leak
    hyp = analysis.hw_hypotheses(pts)
    assert analysis.traces_to_recover(traces, hyp, 0x11, step=20) is None
