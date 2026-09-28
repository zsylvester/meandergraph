# CLAUDE.md

## What this repo is

`meandergraph` is a research Python module that uses directed graphs (networkx) to
describe, analyze, and visualize the migration of meandering river channels through
time. It works with either:

- **centerlines** from simulated channels (e.g., meanderpy models), or
- **banklines** digitized from satellite imagery (e.g., the Mamore River, Bolivia —
  yearly left/right bank shapefiles for 1986–2018 in `data/`).

Core idea: successive channel lines are correlated point-to-point with dynamic time
warping (`librosa.sequence.dtw`); the correlated points become nodes of a `nx.DiGraph`
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

- [meandergraph/meandergraph.py](meandergraph/meandergraph.py) — the entire core
  library (~2100 lines, imported as `mg`). Not installed as a package; notebooks
  import it from the same directory.
- [meandergraph/meandergraph_3D.py](meandergraph/meandergraph_3D.py) — 3D
  visualization of the graphs/polygons as stratigraphy, using **mayavi** (not in
  `environment.yml`).
- `archive/` (gitignored) — old paper figures, animations, design files, abstracts,
  and `mg_temp.py` (an older snapshot of the module; its `polygon_width_and_length`
  has been merged into `meandergraph.py`). Do not develop here.
- Notebooks in `examples/` (upstream layout since PR #3):
  - `meandergraph_Mamore_banks_simple_example.ipynb` — the canonical bankline
    workflow (referenced by the README).
  - `meandergraph_Mamore_banks_example.ipynb`, `Plot_Mamore_meandergraph_data.ipynb`
    — longer bankline analyses.
  - `meandergraph_example.ipynb`, `Meandergraph_large_model.ipynb`,
    `Meandergraph_read_model.ipynb` — centerline/simulation-based examples (use
    meanderpy output HDF5 files in the repo root).
  - Note: several notebook cells are stale relative to current function signatures
    (see Pitfalls).
- `data/` — Mamore bankline shapefiles (`lb_YYYY.*`, `rb_YYYY.*`; 1986–2018, with
  2002 and 2012 missing).
- Repo root — data files used/produced by the notebooks (`.hdf5`, `.npz`,
  `.gpickle`, `mamore_right_bank.*`; all gitignored) plus `docs/images/` for the
  README figure. There is still no `setup.py`/`pyproject.toml` and no test suite
  (Phase 1 of the improvement plan).

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
  missing `geopandas` (needed by all bankline notebooks) and `mayavi` (needed only by
  `meandergraph_3D.py`).
- `librosa` is used *only* for its `dtw` function.

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

The code was written around 2021–2023 against **shapely 1.8, networkx 2.x,
matplotlib < 3.9** and has not been updated since:

- `mpl.cm.get_cmap` (used in `plot_bars_from_centerline`, `plot_bars_from_banks`)
  was removed in matplotlib 3.9 — still unfixed (Phase 3).
- `nx.write_gpickle`/`read_gpickle` (used in notebooks) were removed in networkx 3.0
  — still unfixed (Phase 3).
- `plot_bar_lines` still iterates a MultiLineString with `for l in line:`
  (shapely 2 needs `.geoms`) — Phase 3.
- Notebook cells unpack `correlate_curves`/`correlate_set_of_curves` into 2 values;
  the functions now return 3 (`p, q, cost`). Other notebook calls
  (`create_polygon_graphs_and_bar_graphs`, `plot_age_map`) also use outdated
  signatures — Phase 5.
- The Phase 2 bug list in [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) was fixed in
  Aug 2026 (verified by a synthetic smoke test of the non-plotting pipeline);
  Phases 0 and 2 of the plan are done, Phases 1 and 3–5 are not.

Other conventions to keep in mind:

- Many functions both compute *and* plot (names starting with `plot_` often return
  the computed objects and are called for their return values).
- `restrict_and_correlate_lines` mutates its `X`, `Y` list arguments in place.
- Node attributes: `x`, `y`, `age`, `curv`; graph-level attributes:
  `number_of_centerlines`, `x`, `y` (arrays over all nodes, indexed by node id),
  `start_nodes`, `cutoff_nodes`.
- Long-running loops use `tqdm`/`trange`; building graphs for ~30 banklines takes
  minutes, not seconds.
