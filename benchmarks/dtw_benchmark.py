"""
Benchmark and compare DTW implementations on real Mamore bankline pairs.

Methods
-------
librosa       librosa.sequence.dtw (reference; needs librosa installed)
exact         meandergraph.dtw.dtw_exact  (identical paths to librosa)
fast          meandergraph.dtw.dtw_fast   (coarse-to-fine corridor DTW)
coarse        block-average -> exact DTW -> interpolate the path back up (the
              approach of the ChronoLog functions; no full-resolution refinement)

Each measurement runs in its own subprocess so that peak memory is meaningful.

Usage:  python benchmarks/dtw_benchmark.py [--sizes 5000 15000 25000] [--pairs 0 10 30]
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from glob import glob

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)


def _rss_gb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9


def upsample_warping_path(p_coarse, q_coarse, factor, n1, n2):
    """Map a path found on block-averaged curves back to full resolution
    (ChronoLog's `_upsample_warping_path`, kept for comparison)."""
    pc, qc = np.asarray(p_coarse), np.asarray(q_coarse)
    if pc[0] > pc[-1]:
        pc, qc = pc[::-1], qc[::-1]
    center = (factor - 1) / 2.0
    x1 = np.concatenate([[0.0], pc * factor + center, [n1 - 1.0]])
    x2 = np.concatenate([[0.0], qc * factor + center, [n2 - 1.0]])
    i, j = np.arange(n1), np.arange(n2)
    q_of_p = np.clip(np.round(np.interp(i, x1, x2)), 0, n2 - 1).astype(int)
    p_of_q = np.clip(np.round(np.interp(j, x2, x1)), 0, n1 - 1).astype(int)
    path = np.vstack([np.column_stack([i, q_of_p]), np.column_stack([p_of_q, j])])
    path = path[np.lexsort((path[:, 1], path[:, 0]))]
    keep = np.ones(len(path), dtype=bool)
    keep[1:] = np.any(np.diff(path, axis=0) != 0, axis=1)
    path = path[keep]
    return path[::-1, 0], path[::-1, 1]


def load_pair(pair, size, bank="rb"):
    """Two successive banklines (whole river), resampled to about `size` points."""
    import geopandas as gpd
    from meandergraph.correlation import resample_centerline

    files = sorted(glob(os.path.join(ROOT, "data", f"{bank}*.shp")))
    lines = []
    for f in (files[pair], files[pair + 1]):
        g = gpd.read_file(f)
        x, y = np.array(g["geometry"][0].xy[0]), np.array(g["geometry"][0].xy[1])
        length = np.sum(np.hypot(np.diff(x), np.diff(y)))
        ds = length / size
        x, y = resample_centerline(x, y, ds)[:2]
        lines.append((x, y))
    (x1, y1), (x2, y2) = lines
    return x1, y1, x2, y2


def worker(args):
    x1, y1, x2, y2 = load_pair(args.pair, args.size)
    params = json.loads(args.params)
    from meandergraph import dtw as md
    if args.method in ("exact", "fast"):  # compile before timing
        md.dtw_exact(x1[:30], y1[:30], x2[:30], y2[:30])
        md.dtw_fast(x1[:3000], y1[:3000], x2[:3000], y2[:3000], **({} if args.method == "exact" else params))
    if args.method == "librosa":
        from librosa.sequence import dtw
        dtw(np.vstack((x1[:30], y1[:30])), np.vstack((x2[:30], y2[:30])))
    base = _rss_gb()
    times = []
    for _ in range(args.repeats):
        t = time.perf_counter()
        if args.method == "librosa":
            D, wp = dtw(np.vstack((x1, y1)), np.vstack((x2, y2)))
            p, q, cost = wp[:, 0], wp[:, 1], D[-1, -1]
            del D
        elif args.method == "exact":
            p, q, cost = md.dtw_exact(x1, y1, x2, y2)
        elif args.method == "fast":
            p, q, cost = md.dtw_fast(x1, y1, x2, y2, **params)
        elif args.method == "coarse":
            f = params["downsample"]
            pc, qc, cost = md.dtw_exact(md.block_average(x1, f), md.block_average(y1, f),
                                         md.block_average(x2, f), md.block_average(y2, f))
            p, q = upsample_warping_path(pc, qc, f, len(x1), len(x2))
            cost = float(np.sum(np.hypot(x1[p] - x2[q], y1[p] - y2[q])))  # cost of the upsampled path
        times.append(time.perf_counter() - t)
    dt = min(times)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    np.savez(args.out, p=p, q=q, x1=x1, y1=y1, x2=x2, y2=y2)
    print(json.dumps(dict(time=dt, time_median=float(np.median(times)), peak_gb=peak, base_gb=base, cost=float(cost), n=len(x1), m=len(x2))))


def path_metrics(ref, res):
    """Difference between a path and the reference (exact) path."""
    x1, y1, x2, y2 = res["x1"], res["y1"], res["x2"], res["y2"]
    pr, qr, p, q = ref["p"], ref["q"], res["p"], res["q"]
    n = len(x1)
    cells_ref = set(zip(pr.tolist(), qr.tolist()))
    same = sum((a, b) in cells_ref for a, b in zip(p.tolist(), q.tolist()))
    def qmid(pp, qq):
        s = np.bincount(pp, weights=qq, minlength=n); c = np.bincount(pp, minlength=n)
        return s / np.maximum(c, 1)
    dq = np.abs(qmid(p, q) - qmid(pr, qr))
    ds = np.median(np.hypot(np.diff(x1), np.diff(y1)))  # point spacing (m)
    matched = np.hypot(x1[p] - x2[q], y1[p] - y2[q])
    matched_ref = np.hypot(x1[pr] - x2[qr], y1[pr] - y2[qr])
    return dict(cells_in_ref_pct=100 * same / len(p), ref_cells_found_pct=100 * same / len(pr),
                max_index_dev=float(dq.max()), mean_index_dev=float(dq.mean()),
                max_dev_m=float(dq.max() * ds), p99_dev_m=float(np.percentile(dq, 99) * ds),
                mean_match_dist_m=float(matched.mean()), ref_mean_match_dist_m=float(matched_ref.mean()))


REPEATS = 3


def run(method, params, pair, size, outdir):
    out = os.path.join(outdir, f"{method}_{json.dumps(params, sort_keys=True)}_{pair}_{size}.npz")
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--method", method,
           "--params", json.dumps(params), "--pair", str(pair), "--size", str(size), "--out", out, "--repeats", str(REPEATS)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, None, r.stderr[-300:]
    line = [l for l in r.stdout.splitlines() if l.startswith("{")][-1]
    return json.loads(line), np.load(out), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--method"); ap.add_argument("--params", default="{}")
    ap.add_argument("--pair", type=int, default=0); ap.add_argument("--size", type=int, default=5000)
    ap.add_argument("--out", default="")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--sizes", type=int, nargs="+", default=[5000, 15000, 25000])
    ap.add_argument("--pairs", type=int, nargs="+", default=[0, 10, 30])
    ap.add_argument("--skip-librosa-above", type=int, default=30000)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    global REPEATS
    REPEATS = args.repeats
    if args.worker:
        return worker(args)
    configs = [("librosa", {}), ("exact", {}),
               ("fast", {"radius": 1}), ("fast", {"radius": 2}), ("fast", {"radius": 4}),
               ("fast", {"downsample": 8, "radius": 2}), ("fast", {"downsample": 16, "radius": 2}),
               ("coarse", {"downsample": 4}), ("coarse", {"downsample": 8}), ("coarse", {"downsample": 16})]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for size in args.sizes:
            for pair in args.pairs:
                ref = None
                for method, params in configs:
                    if method == "librosa" and size > args.skip_librosa_above:
                        continue
                    meas, res, err = run(method, params, pair, size, tmp)
                    if meas is None:
                        print("FAILED", method, params, size, pair, err); continue
                    if method == "exact":
                        ref = res
                    row = dict(method=method, params=params, size=size, pair=pair, **meas)
                    if ref is not None and method != "exact":
                        row.update(path_metrics(ref, res))
                    row["_res"] = None
                    rows.append(row)
                    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "_res"}), flush=True)
    if args.json:
        json.dump([{k: v for k, v in r.items() if k != "_res"} for r in rows], open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
