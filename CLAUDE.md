# CLAUDE.md

## What this repo is

`meandergraph` is a research Python module that uses directed graphs (networkx) to
describe, analyze, and visualize the migration of meandering river channels through
time. It works with either:

- **centerlines** from simulated channels (e.g., meanderpy models), or
- **banklines** digitized from satellite imagery (e.g., the Mamore River, Bolivia —
  yearly left/right bank shapefiles for 1986–2018 in `data/`).

Core idea: successive channel lines are correlated point-to-point with dynamic time
warping (`meandergraph/dtw.py`); the correlated points become nodes of a `nx.DiGraph`
with two edge types:

- `'channel'` edges — connect consecutive points *along* one centerline/bankline
  (one line per time step; the line index equals the node's `age` attribute).
- `'radial'` edges — connect a point on line *t* to its correlated point on line
  *t+1*, i.e., trajectories of bank migration *across* time.

From this line graph, quadrilateral-ish polygons are built between consecutive lines
and consecutive radial trajectories (`create_polygon_graph`,
`create_simple_polygon_graph`). Polygons carry attributes (`age`, `width`, `length`,
`migr_rate`, `direction`, `curv`) and are aggregated into `Scroll` and `Bar` objects
(scroll = one-timestep depositional area, bar = connected set of scrolls). This
enables maps of migration rate, bar age, and curvature, plus per-polygon export to
shapefiles. Cutoffs are detected two ways: large jumps in distance along radial paths
(`max_dist` in `create_graph_from_channel_lines`) and large one-step depositional
areas (`cutoff_area` in the bar/scroll functions).

The bankline workflow uses **two parallel graphs**: `graph1` = right bank, `graph2` =
left bank, built and processed identically, then combined for channel polygons and
bars.

## Layout

- [meandergraph/](meandergraph/) — the core library, a package imported as `mg`
  (`import meandergraph as mg`). Not pip-installed; notebooks put the repo root on
  `sys.path` (`sys.path.append('../')`). [`__init__.py`](meandergraph/__init__.py)
  re-exports every public name of the submodules, so `mg.<function>` works without
  naming a submodule. Each submodule lists its public names in `__all__`; a new
  public function must be added there or it will not be reachable as `mg.<name>`.
  - `geometry.py` — `fix_geometry` (the single place invalid shapely geometries are
    repaired), `compute_distance`, `directionOfPoint`, `ensure_multipolygon`.
  - `dtw.py` — the DTW itself, in numba (no librosa): `dtw_exact` (full DTW; paths
    and costs are identical to `librosa.sequence.dtw`, but ~10x faster and ~14x less
    memory at 25,000 points) and `dtw_fast` (coarse-to-fine: solves a block-averaged
    problem, then the full-resolution DTW only in a corridor around it; ~60x faster
    than `dtw_exact` at 25,000 points but only optimal *within* the corridor).
  - `correlation.py` — DTW correlation (`correlate_curves`,
    `correlate_set_of_curves`; both take `method='exact'` (default) or `'fast'`),
    resampling, curvature, timesteps, `find_indices`, `restrict_and_correlate_lines`.
  - `graph.py` — the line graph (channel + radial edges), path finding, node
    thinning, edge directions, `radial_successor` / `channel_successor` helpers.
  - `polygons.py` — polygon graphs (`create_polygon_graph`,
    `create_simple_polygon_graph`), channel polygons, one-step differences.
  - `bars.py` — `Scroll` / `Bar` dataclasses and the bar-building pipeline.
  - `plot.py` — the `plot_*` functions.
- [meandergraph/meandergraph_3D.py](meandergraph/meandergraph_3D.py) — 3D
  visualization of the graphs/polygons as stratigraphy, using **mayavi** (not in
  `environment.yml`).
- `archive/` (gitignored) — old paper figures, animations, design files, abstracts,
  and `mg_temp.py` (an older snapshot of the module; its `polygon_width_and_length`
  has been merged into `meandergraph/bars.py`). Do not develop here.
- Notebooks in `examples/` (upstream layout since PR #3):
  - `meandergraph_Mamore_banks_simple_example.ipynb` — the canonical bankline
    workflow (referenced by the README).
  - `meandergraph_Mamore_banks_example.ipynb`, `Plot_Mamore_meandergraph_data.ipynb`
    — longer bankline analyses.
  - `meandergraph_example.ipynb`, `Meandergraph_large_model.ipynb`,
    `Meandergraph_read_model.ipynb` — centerline/simulation-based examples (use
    meanderpy output HDF5 files in the repo root).
  - Note: only the simple example and `meandergraph_example.ipynb` have been
    updated to the current API; the others still have stale cells (see Pitfalls).
- `data/` — Mamore bankline shapefiles (`lb_YYYY.*`, `rb_YYYY.*`; 1986–2018, with
  2002 and 2012 missing).
- Repo root — data files used/produced by the notebooks (`.hdf5`, `.npz`,
  `.gpickle`, `mamore_right_bank.*`; all gitignored) plus `docs/images/` for the
  README figure. There is still no `setup.py`/`pyproject.toml`.
- `tests/` — a synthetic-data smoke test of the bankline pipeline
  (`tests/test_pipeline_smoke.py`; run `python -m pytest tests`). It checks
  structural properties (counts, well-formedness), not exact values.

## Typical bankline workflow (what the code is for)

1. Read left/right bank shapefiles into coordinate lists `X1/Y1` (right), `X2/Y2` (left).
2. Restrict all lines to a study reach and resample to constant spacing
   (`restrict_and_correlate_lines`, or manually: `find_indices` + `resample_centerline`).
3. Correlate successive lines: `correlate_set_of_curves` → `P, Q, costs`.
4. Build graphs: `create_graph_from_channel_lines(X, Y, P, Q, n_points, max_dist, ...)`.
5. Thin dense post-cutoff nodes: `remove_high_density_nodes(graph, min_dist, max_dist)`.
6. Add left/right directions to radial edges: `add_edge_directions_to_bank_graph`.
7. Bars/scrolls: `plot_bars_from_banks`, `create_scrolls_and_find_connected_scrolls`,
   `create_polygon_graphs_and_bar_graphs` → `wbars` (list of `Bar` objects).
8. Maps: `plot_bar_graphs` (`'migration'`/`'curvature'`/`'age'`),
   `plot_migration_rate_map`, `plot_age_map`, `plot_simple_polygon_graph`.
9. Export polygons + attributes as a GeoDataFrame → shapefile (EPSG:32620 for Mamore).

## Environment

- Use the dedicated conda env **`meandergraph`**
  (`/Users/zoltan/miniforge3/envs/meandergraph`). **Never install packages into
  `base`.** Run things via `conda run -n meandergraph python ...` or activate the env.
- `conda` is at `/Users/zoltan/miniforge3/bin/conda` (may not be on PATH in
  non-interactive shells).
- [meandergraph/environment.yml](meandergraph/environment.yml) is incomplete: it is
  missing `mayavi` (needed only by `meandergraph_3D.py`) and `pytest` (needed to run
  `tests/`).
- `numba` is a required dependency (it runs the DTW loops in `meandergraph/dtw.py`);
  `librosa` is no longer used. The first call to a DTW function compiles (and caches)
  the kernels, which takes a few seconds.
- `benchmarks/dtw_benchmark.py` compares librosa / exact / fast / coarse-only DTW on the
  Mamore banklines (time, peak memory, difference from the exact path).

## Collaboration

GitHub PRs #2–#4 (2023–2024, from `cmspeed`) added non-uniform timestep support
(`timesteps` argument of `create_graph_from_channel_lines`,
`add_timesteps_to_line_graph`, node attribute `timestep`), made
`create_graph_from_channel_lines` always add curvature, moved the example
notebooks to `examples/`, and updated many docstrings. This was merged with the
local Phase 0–2 work in Sep 2026. `plot_migration_rate_map` no longer takes
`dt`/`saved_ts` (it uses the `timestep` node attribute), and
`create_simple_polygon_graph` takes an (unused) `X` argument.

## Pitfalls / current state (as of 2026-09)

The code was written around 2021–2023 against shapely 1.8, networkx 2.x and
matplotlib < 3.9 and has since been updated for shapely 2, networkx 3 and
matplotlib 3.9+ (`.geoms` iteration, `mpl.colormaps`, `pickle` instead of
`nx.write_gpickle`). Remaining rough edges:

- Notebooks other than the simple example and `meandergraph_example.ipynb` still have
  stale cells: `meandergraph_Mamore_banks_example.ipynb` and
  `Meandergraph_large_model.ipynb` unpack `correlate_set_of_curves` into 2 values (it
  returns 3: `P, Q, costs`), and `meandergraph_Mamore_banks_example.ipynb` also uses
  `nx.write_gpickle`, `gdf.crs = {'init': ...}` and `plot_age_map(..., W=...)`;
  `Plot_Mamore_meandergraph_data.ipynb` uses `nx.read_gpickle`.
- Two bare `except:` clauses remain in `graph.py`.
- The Phase 0, 2 and most of 3–4 items of [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md)
  are done, as is dropping librosa; Phase 1 (packaging) and Phase 5 are not.

Other conventions to keep in mind:

- Many functions both compute *and* plot (names starting with `plot_` often return
  the computed objects and are called for their return values).
- `restrict_and_correlate_lines` returns restricted copies and does not mutate its `X`, `Y` list
   arguments.
- Compute steps are separated from plotting where it matters: `plot_bars_from_banks`
  / `plot_bars_from_centerline`, `create_scrolls_and_find_connected_scrolls` and
  `Bar.create_merged_polygons` are thin wrappers over `compute_bars_from_banks`,
  `compute_bars_from_centerline`, `compute_scrolls_and_connections` and
  `Bar.compute_merged_polygons`. Use the `compute_*` versions to avoid creating figures.
- `Bar` and `Scroll` are dataclasses; optional attributes such as `merged_polygons`
  and `bank_type` exist from construction and are `None` until set.
- `create_polygon_graph` builds each polygon from `node_1`, `node_2` and the path
  along the next line between their radial successors (the "outer boundary"); any
  outer-boundary length >= 2 nodes yields a polygon (earlier versions silently
  skipped lengths other than 2-5, so results can contain a few more polygons than
  older outputs).
- `create_graph_from_channel_lines` resolves centerline points to nodes with a
  `(centerline index, point index) -> node id` dict; where DTW maps several
  trajectories to the same point, the first node created wins.
- `correlate_curves(..., method='fast')` is opt-in: on cutoff-affected pairs it can
  return a path whose cost is a few percent above the exact one (see
  [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md), Phase 3.5). Default `'exact'` reproduces
  the old librosa output exactly. The `band_rad` argument of `correlate_curves` is unused
  (it never had an effect).
- Node attributes: `x`, `y`, `age`, `curv` (`timestep` when timesteps are given); graph-level attributes:
  `number_of_centerlines`, `x`, `y` (arrays over all nodes, indexed by node id),
  `start_nodes`, `cutoff_nodes`.
- Long-running loops use `tqdm`/`trange`; building graphs for ~30 banklines takes
  minutes, not seconds.
