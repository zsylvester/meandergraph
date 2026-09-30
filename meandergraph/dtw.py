"""
Dynamic time warping (DTW) of two 2D curves, implemented with numba so that
librosa is not needed.

Two implementations are provided:

- :func:`dtw_exact` -- full DTW that reproduces ``librosa.sequence.dtw(X, Y)``
  (Euclidean point-to-point cost, steps (1,1), (0,1), (1,0), same tie-breaking)
  path for path. Unlike librosa it never builds the N x M cost matrix or the
  float accumulated-cost matrix; it computes point distances on the fly, keeps
  two rows of accumulated cost, and stores one ``uint8`` per cell for the
  backtracking. Time is O(N*M), memory is N*M bytes.

- :func:`dtw_fast` -- coarse-to-fine DTW. The curves are block-averaged, the
  coarse problem is solved (recursively, if it is still large), and the
  full-resolution DTW is then run only inside a corridor around the up-sampled
  coarse path. Time and memory are O(N*w) for corridor width w. The result is
  the optimal path *within the corridor*: it equals the exact path whenever the
  exact path stays inside the corridor, and its cost is never lower than the
  exact cost.

All functions return the warping path end-to-start (like librosa) as two integer
arrays ``p`` (indices along the first curve) and ``q`` (second curve), and the
accumulated cost of the path.
"""
from typing import Optional, Tuple

import numpy as np
from numba import njit

__all__ = ["dtw_exact", "dtw_fast", "block_average"]

_BASE_SIZE = 2500  # dtw_fast solves problems up to this many points exactly


def _prepare(x1, y1, x2, y2):
    x1 = np.ascontiguousarray(x1, dtype=np.float64)
    y1 = np.ascontiguousarray(y1, dtype=np.float64)
    x2 = np.ascontiguousarray(x2, dtype=np.float64)
    y2 = np.ascontiguousarray(y2, dtype=np.float64)
    if x1.shape != y1.shape or x2.shape != y2.shape or x1.ndim != 1 or x2.ndim != 1:
        raise ValueError("x and y coordinates of each curve must be 1D arrays of equal length")
    if len(x1) == 0 or len(x2) == 0:
        raise ValueError("curves must contain at least one point")
    if np.isnan(x1).any() or np.isnan(y1).any() or np.isnan(x2).any() or np.isnan(y2).any():
        raise ValueError("curve coordinates contain NaN values")
    return x1, y1, x2, y2


@njit(cache=True)
def _backtrack(steps, n, m, row_lo, row_offset):
    """Follow the stored steps from (n-1, m-1) back to (0, 0).

    steps is either a full (n, m) array (row_offset is None-like, signalled by
    row_offset[0] < 0) or a flat ragged array addressed as
    steps[row_offset[i] + (j - row_lo[i])].
    """
    p = np.empty(n + m, dtype=np.int64)
    q = np.empty(n + m, dtype=np.int64)
    i = n - 1
    j = m - 1
    k = 0
    p[0] = i
    q[0] = j
    k = 1
    ragged = row_offset[0] >= 0
    while i != 0 or j != 0:
        if ragged:
            st = steps[row_offset[i] + (j - row_lo[i])]
        else:
            st = steps[i * m + j]
        if st == 0:
            i -= 1
            j -= 1
        elif st == 1:
            j -= 1
        else:
            i -= 1
        if i < 0 or j < 0:
            break
        p[k] = i
        q[k] = j
        k += 1
    return p[:k].copy(), q[:k].copy()


@njit(cache=True)
def _dtw_exact_kernel(x1, y1, x2, y2):
    n = len(x1)
    m = len(x2)
    steps = np.empty(n * m, dtype=np.uint8)
    prev = np.empty(m, dtype=np.float64)
    cur = np.empty(m, dtype=np.float64)
    inf = np.inf
    for i in range(n):
        xi = x1[i]
        yi = y1[i]
        for j in range(m):
            dx = xi - x2[j]
            dy = yi - y2[j]
            c = np.sqrt(dx * dx + dy * dy)
            if i == 0 and j == 0:
                cur[0] = c
                steps[0] = 0
                continue
            # same candidate order and strict '<' as librosa, so ties resolve
            # identically: diagonal, then (0,1), then (1,0)
            best = inf
            st = 0
            if i > 0 and j > 0:
                v = prev[j - 1] + c
                if v < best:
                    best = v
                    st = 0
            if j > 0:
                v = cur[j - 1] + c
                if v < best:
                    best = v
                    st = 1
            if i > 0:
                v = prev[j] + c
                if v < best:
                    best = v
                    st = 2
            cur[j] = best
            steps[i * m + j] = st
        tmp = prev
        prev = cur
        cur = tmp
    cost = prev[m - 1]
    dummy = np.full(1, -1, dtype=np.int64)
    p, q = _backtrack(steps, n, m, dummy, dummy)
    return p, q, cost


@njit(cache=True)
def _dtw_corridor_kernel(x1, y1, x2, y2, lo, hi):
    """DTW restricted to columns lo[i]..hi[i] (inclusive) of each row i."""
    n = len(x1)
    m = len(x2)
    offset = np.empty(n, dtype=np.int64)
    total = 0
    for i in range(n):
        offset[i] = total
        total += hi[i] - lo[i] + 1
    steps = np.zeros(total, dtype=np.uint8)
    inf = np.inf
    prev = np.full(m, inf)
    cur = np.full(m, inf)
    for i in range(n):
        xi = x1[i]
        yi = y1[i]
        li = lo[i]
        hii = hi[i]
        for j in range(li, hii + 1):
            dx = xi - x2[j]
            dy = yi - y2[j]
            c = np.sqrt(dx * dx + dy * dy)
            if i == 0 and j == 0:
                cur[0] = c
                continue
            best = inf
            st = 0
            if i > 0 and j > 0:
                v = prev[j - 1] + c
                if v < best:
                    best = v
                    st = 0
            if j > 0:
                v = cur[j - 1] + c
                if v < best:
                    best = v
                    st = 1
            if i > 0:
                v = prev[j] + c
                if v < best:
                    best = v
                    st = 2
            cur[j] = best
            steps[offset[i] + (j - li)] = st
        # the row before this one is no longer needed: clear its window so that
        # it can be reused as the 'current' row (entries outside a row's window
        # must read as infinity)
        if i > 0:
            for j in range(lo[i - 1], hi[i - 1] + 1):
                prev[j] = inf
        tmp = prev
        prev = cur
        cur = tmp
    cost = prev[m - 1]
    if not np.isfinite(cost):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), cost, False
    p, q = _backtrack(steps, n, m, lo, offset)
    touches = False
    for k in range(len(p)):
        i = p[k]
        j = q[k]
        if (j == lo[i] and lo[i] > 0) or (j == hi[i] and hi[i] < m - 1):
            touches = True
            break
    return p, q, cost, touches


def block_average(values: np.ndarray, factor: int) -> np.ndarray:
    """Downsample by averaging blocks of `factor` consecutive samples.

    A trailing partial block is dropped (the caller re-anchors the path at the
    true ends of the curves).
    """
    n = len(values) // factor
    return values[:n * factor].reshape(n, factor).mean(axis=1)


def dtw_exact(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Full dynamic time warping of two 2D curves; same result as
    ``librosa.sequence.dtw(np.vstack((x1, y1)), np.vstack((x2, y2)))``.

    Returns
    -------
    p, q : 1D int arrays
        Warping path indices along the first and second curve, ordered from
        the last point pair to the first.
    cost : float
        Accumulated cost of the path.
    """
    x1, y1, x2, y2 = _prepare(x1, y1, x2, y2)
    p, q, cost = _dtw_exact_kernel(x1, y1, x2, y2)
    return p, q, float(cost)


def _corridor_from_coarse_path(pc, qc, factor, n, m, radius):
    """Full-resolution column window (lo, hi) for each row, around a coarse path."""
    nc = int(pc.max()) + 1
    mc = int(qc.max()) + 1
    jmin = np.full(nc, mc, dtype=np.int64)
    jmax = np.full(nc, -1, dtype=np.int64)
    np.minimum.at(jmin, pc, qc)
    np.maximum.at(jmax, pc, qc)
    rows = np.minimum(np.arange(n) // factor, nc - 1)
    r = radius * factor
    lo = jmin[rows] * factor - r
    hi = (jmax[rows] + 1) * factor - 1 + r
    last_block = rows == nc - 1  # includes the samples dropped by block_average
    hi[last_block] = m - 1
    lo = np.clip(lo, 0, m - 1)
    hi = np.clip(hi, 0, m - 1)
    lo[0] = 0
    hi[n - 1] = m - 1
    # the corridor has to be monotone and connected from row to row
    hi = np.maximum.accumulate(hi)
    lo = np.minimum.accumulate(lo[::-1])[::-1]
    lo[1:] = np.minimum(lo[1:], hi[:-1] + 1)
    return lo.astype(np.int64), hi.astype(np.int64)


def dtw_fast(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray,
             downsample: Optional[int] = None, radius: int = 2,
             max_widenings: int = 3) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Coarse-to-fine dynamic time warping of two 2D curves.

    The curves are block-averaged by `downsample`, the coarse DTW path is found
    (recursively if the coarse problem is still larger than about 2500 points),
    and the full-resolution DTW is computed only within `radius` coarse blocks
    of that path. If the resulting path runs along the edge of the corridor, the
    corridor is widened and the DTW repeated (up to `max_widenings` times).

    Parameters
    ----------
    x1, y1, x2, y2 : 1D arrays
        Coordinates of the two curves.
    downsample : int, optional
        Block-averaging factor. By default it is chosen so that the coarse
        problem has about 1000-2500 points.
    radius : int
        Corridor half-width, in coarse blocks, around the coarse path.
    max_widenings : int
        How many times the radius is doubled if the path touches the corridor
        edge.

    Returns
    -------
    p, q, cost : as for :func:`dtw_exact`. ``cost`` is the cost of the best path
    inside the corridor, i.e. >= the exact DTW cost.
    """
    x1, y1, x2, y2 = _prepare(x1, y1, x2, y2)
    return _dtw_fast(x1, y1, x2, y2, downsample, radius, max_widenings)


def _dtw_fast(x1, y1, x2, y2, downsample, radius, max_widenings):
    n, m = len(x1), len(x2)
    size = max(n, m)
    if size <= _BASE_SIZE and downsample is None:
        p, q, cost = _dtw_exact_kernel(x1, y1, x2, y2)
        return p, q, float(cost)
    if downsample is None:
        downsample = int(np.ceil(size / 1500.0))
    factor = max(int(downsample), 2)
    if min(n, m) // factor < 2:
        p, q, cost = _dtw_exact_kernel(x1, y1, x2, y2)
        return p, q, float(cost)
    cx1, cy1 = block_average(x1, factor), block_average(y1, factor)
    cx2, cy2 = block_average(x2, factor), block_average(y2, factor)
    pc, qc, _ = _dtw_fast(cx1, cy1, cx2, cy2, None, radius, max_widenings)
    r = radius
    for attempt in range(max_widenings + 1):
        lo, hi = _corridor_from_coarse_path(pc, qc, factor, n, m, r)
        p, q, cost, touches = _dtw_corridor_kernel(x1, y1, x2, y2, lo, hi)
        if np.isfinite(cost) and not touches:
            break
        r *= 2
    if not np.isfinite(cost):  # corridor never connected: fall back to the full DTW
        p, q, cost = _dtw_exact_kernel(x1, y1, x2, y2)
    return p, q, float(cost)
