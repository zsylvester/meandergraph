"""
Tests for meandergraph.dtw: the numba DTW must reproduce librosa's result
exactly (when librosa is installed), and the coarse-to-fine version must return
valid, near-optimal paths.
"""
import numpy as np
import pytest

import meandergraph as mg
from meandergraph.dtw import block_average, dtw_exact, dtw_fast


def random_curves(rng, n, m, kind="walk"):
    if kind == "ties":  # small integer coordinates give many exactly tied path costs
        a = rng.integers(0, 6, (2, n)).astype(float)
        b = rng.integers(0, 6, (2, m)).astype(float)
    else:
        a = np.cumsum(rng.normal(size=(2, n)), axis=1)
        b = np.cumsum(rng.normal(size=(2, m)), axis=1)
    return a, b


def meander_pair(n, shift=0.15):
    """Two similar sinusoidal 'banklines' of n points, the second slightly migrated."""
    s = np.linspace(0, 20 * np.pi, n)
    x1, y1 = s * 50.0, 80.0 * np.sin(s)
    x2, y2 = s * 50.0 + 4.0 * np.cos(1.3 * s), 84.0 * np.sin(s + shift) + 6.0
    return x1, y1, x2, y2


def check_path(p, q, n, m):
    assert (p[0], q[0]) == (n - 1, m - 1)
    assert (p[-1], q[-1]) == (0, 0)
    dp, dq = -np.diff(p), -np.diff(q)  # the path is ordered end-to-start
    assert np.all((dp >= 0) & (dq >= 0) & (dp + dq >= 1) & (dp <= 1) & (dq <= 1))


def path_cost(p, q, x1, y1, x2, y2):
    return float(np.sum(np.hypot(x1[p] - x2[q], y1[p] - y2[q])))


@pytest.mark.parametrize("kind", ["walk", "ties"])
def test_dtw_exact_matches_librosa(kind):
    librosa_sequence = pytest.importorskip("librosa.sequence")
    rng = np.random.default_rng(0)
    for _ in range(25):
        n, m = rng.integers(2, 300, size=2)
        a, b = random_curves(rng, n, m, kind)
        p, q, cost = dtw_exact(a[0], a[1], b[0], b[1])
        D, wp = librosa_sequence.dtw(a, b)
        assert np.array_equal(p, wp[:, 0])
        assert np.array_equal(q, wp[:, 1])
        assert cost == D[-1, -1]


def test_dtw_exact_path_is_valid_and_cost_consistent():
    rng = np.random.default_rng(1)
    a, b = random_curves(rng, 120, 90)
    p, q, cost = dtw_exact(a[0], a[1], b[0], b[1])
    check_path(p, q, 120, 90)
    assert cost == pytest.approx(path_cost(p, q, a[0], a[1], b[0], b[1]))


def test_dtw_degenerate_sizes():
    for n, m in [(1, 1), (1, 5), (5, 1), (2, 2)]:
        rng = np.random.default_rng(n * 10 + m)
        a, b = random_curves(rng, n, m)
        p, q, _ = dtw_exact(a[0], a[1], b[0], b[1])
        check_path(p, q, n, m)
        p, q, _ = dtw_fast(a[0], a[1], b[0], b[1])
        check_path(p, q, n, m)


def test_dtw_rejects_bad_input():
    with pytest.raises(ValueError):
        dtw_exact(np.array([0.0, np.nan]), np.zeros(2), np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError):
        dtw_exact(np.zeros(3), np.zeros(2), np.zeros(2), np.zeros(2))


def test_block_average_drops_partial_block():
    assert np.allclose(block_average(np.arange(7.0), 3), [1.0, 4.0])


def test_dtw_fast_is_valid_and_near_optimal():
    x1, y1, x2, y2 = meander_pair(6000)
    p, q, cost = dtw_exact(x1, y1, x2, y2)
    pf, qf, cost_f = dtw_fast(x1, y1, x2, y2, downsample=8, radius=2)
    check_path(pf, qf, 6000, 6000)
    assert cost_f == pytest.approx(path_cost(pf, qf, x1, y1, x2, y2))
    assert cost_f >= cost * (1 - 1e-12)  # a constrained optimum cannot beat the global one
    assert cost_f <= cost * 1.001


def test_dtw_fast_default_uses_exact_for_small_problems():
    x1, y1, x2, y2 = meander_pair(800)
    p, q, cost = dtw_exact(x1, y1, x2, y2)
    pf, qf, cost_f = dtw_fast(x1, y1, x2, y2)
    assert np.array_equal(p, pf) and np.array_equal(q, qf) and cost == cost_f


def test_correlate_curves_methods():
    x1, y1, x2, y2 = meander_pair(3000)
    p, q, cost = mg.correlate_curves(x1, x2, y1, y2)
    pe, qe, coste = dtw_exact(x1, y1, x2, y2)
    assert np.array_equal(p, pe) and np.array_equal(q, qe) and cost == coste
    pf, qf, costf = mg.correlate_curves(x1, x2, y1, y2, method="fast", downsample=6)
    check_path(pf, qf, 3000, 3000)
    with pytest.raises(ValueError):
        mg.correlate_curves(x1, x2, y1, y2, method="nope")
    with pytest.raises(TypeError):
        mg.correlate_curves(x1, x2, y1, y2, downsample=4)
