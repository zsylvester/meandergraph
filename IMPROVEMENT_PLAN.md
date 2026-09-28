# meandergraph — refactor & improvement plan

Written 2026-08-31, after a full read of the codebase. Phases are ordered so that
each one leaves the repo in a working state; bug references use current line numbers
in `meandergraph/meandergraph.py`.

> **Status (2026-08-31):** Phases 0 and 2 are DONE (bug fixes verified with a
> synthetic smoke test of the non-plotting pipeline in the `meandergraph` env;
> notebook updates from item 2.8 deferred to Phase 5 as planned). Phase 1 items
> 2–4 and Phases 3–5 remain.

## Phase 0 — Repo hygiene (cheap, do first) — DONE

The repo root is dominated by figure/animation artifacts from past papers
(~temp*.svg, .ai, .gif, .mp4, .gpickle, .hdf5, .jpeg, abstracts, movie_frames/,
screenshots/), plus `__pycache__/`, `.DS_Store`, and `.ipynb_checkpoints/` that are
untracked but clutter every `git status`.

1. Add a `.gitignore` (`__pycache__/`, `.DS_Store`, `.ipynb_checkpoints/`,
   `meandergraph/cache/`, `*.gpickle`, `temp*`, etc.).
2. Move keeper figures into `docs/images/` (the README references
   `meander_graph_1.svg` — keep that path working or update the README).
3. Decide what to do with the large data files (`*.hdf5`, `*.gpickle`,
   `movie_frames/`): delete, or park outside the repo / in a GitHub release. The
   `data/` shapefiles should stay — the README example depends on them.
4. Delete `meandergraph/mg_temp.py` **after** salvaging `polygon_width_and_length`
   (line 684 there), which the current module needs (see bug 5).

## Phase 1 — Environment & packaging

1. A dedicated conda env **`meandergraph`** now exists (python 3.11, shapely 2.1,
   networkx 3.6, matplotlib 3.11, numpy 2.4, librosa 0.11, geopandas 1.1). The
   module imports cleanly there, but parts of it fail at *runtime* with these
   versions — that's what Phases 2–3 fix. Never install into `base`.
2. Fix `meandergraph/environment.yml`: add `geopandas`; note that `mayavi` is
   needed only for `meandergraph_3D.py` (consider making 3D an optional extra —
   mayavi is a heavy, fragile install).
3. Add a `pyproject.toml` so the package is `pip install -e .`-able and the
   notebooks stop depending on the working directory ("coming soon to pip" in the
   README). Suggested layout:
   `src/meandergraph/{__init__.py, correlation.py, graph.py, polygons.py, bars.py, plot.py, io.py}`
   (Phase 4 does the actual splitting; packaging can come first with the single file.)
4. Add `pytest` scaffolding + a tiny synthetic test dataset (e.g., 5 short
   sine-wave centerlines) so correlation → graph → polygons can be tested in seconds.

## Phase 2 — Bug fixes (all verified against the current code) — DONE

1. **`remove_high_density_nodes`, lines 637–638** — checks `if node in
   graph2.graph['start_nodes']` but removes `n` (the loop variable of the *radial*
   path). Should test and remove the same variable; as written it can raise
   `ValueError` or silently corrupt `start_nodes`.
2. **`plot_bars_from_centerline`, lines 966–974** — the `else` is attached to the
   `for` loop (for–else), so after plotting the parts of a MultiPolygon it *always*
   calls `bar.exterior` on that MultiPolygon → `AttributeError`. Compare with the
   correct structure in `plot_bars_from_banks` (lines 1148–1169). Also `for b in
   bar:` must be `bar.geoms` under shapely 2.
3. **`plot_bar_lines`, line 1576** — `for common_node in set(radial_path) and
   set(source_nodes):` — `and` returns the second set, so it iterates over *all*
   source nodes instead of the intersection. Should be `&`.
4. **`plot_bar_graphs`, line 1739** — calls
   `plot_curvature_map(wbars[i], graph1, graph2, vmin, vmax, W, ax)` but the
   signature (line 1447) is `(wbar, vmin, vmax, W, cmap, ax)` → `TypeError`
   whenever `plot_type='curvature'`.
5. **`add_polygon_width_and_length`, line 1427** — calls `polygon_width_and_length`,
   which does not exist in this module (the comment even asks "where is this
   function??"). It lives in `mg_temp.py` line 684; move it over.
6. **`create_polygon_graph`, lines 851–860** — the `i == 0` fallback polygon uses
   `length` and `width`, which are only assigned in *earlier* iterations; on an
   unlucky first node this raises `NameError`. Similarly `width_2`/`length_1`/
   `length_2` are unassigned when `outer_poly_boundary` is empty or has >5 nodes —
   the polygon is then built with stale values from a previous iteration.
7. **`one_step_difference_no_plot` / `one_step_difference_no_jump`** — assume
   `ch1.difference(ch2)` returns a MultiPolygon (`bar.geoms`); under shapely 2 a
   single-Polygon result has no `.geoms` → `AttributeError`. Wrap results
   consistently (e.g., a small `as_multipolygon()` helper).
8. **Stale notebook API** — notebooks unpack `p, q = mg.correlate_curves(...)` and
   `P, Q = mg.correlate_set_of_curves(...)`, but both now also return costs; the
   simple-example notebook also calls `create_polygon_graphs_and_bar_graphs` with an
   extra `cutoffs` argument and `plot_age_map` with a `W=` kwarg that no longer
   exists. Update the notebooks (Phase 5) or restore compatible signatures.
9. **Bare `except:` clauses** (lines 729, 1833, and in `Bar.add_polygon_graphs`
   1949–1952, 1985–1993) — swallow real errors (including KeyboardInterrupt);
   narrow to the expected shapely/networkx exceptions.
10. Minor: duplicate `from shapely.ops import unary_union` import (lines 12/15);
    `find_radial_path_2` is a copy of `find_radial_path` without ages — merge;
    docstring of `correlate_curves` documents 2 return values but returns 3;
    `restrict_and_correlate_lines` mutates its `X`, `Y` arguments in place —
    return new lists instead.

## Phase 3 — Modernization (make it run on the new env)

1. **matplotlib**: replace `mpl.cm.get_cmap('viridis')` (lines 937, 1123; removed
   in mpl 3.9) with `plt.get_cmap(...)` or `mpl.colormaps[...]`.
2. **shapely 2**: audit every multi-geometry iteration (`for b in bar:`chunks,
   `for l in line:` in `plot_bar_lines`) → `.geoms`; use
   `shapely.make_valid`/`shapely.validation` instead of the `buffer(0)` idiom where
   possible (keep `fix_geometry` as the single place this happens).
3. **networkx 3**: notebooks use `nx.write_gpickle`/`read_gpickle` (removed) —
   switch to the `pickle` module directly, and consider a versioned save format.
4. **geopandas**: `gdf.crs = {'init': 'epsg:32620'}` → `gdf.set_crs(32620)`.
5. **Drop librosa**: it's imported solely for `dtw`. Options: vendor a small DTW
   implementation (it's ~40 lines with numba or plain numpy), or use `dtaidistance`.
   This removes the heaviest dependency (audio stack, numba, soundfile).
6. Run the full Mamore simple example end-to-end in the new env as the acceptance
   test for this phase.

## Phase 4 — Refactor (structure, not behavior)

1. **Split the 2100-line module** into focused submodules (see Phase 1.3):
   correlation (DTW, resampling, curvature), graph building, polygon graphs,
   bars/scrolls, plotting, io. Keep `import meandergraph as mg` working via
   `__init__.py` re-exports so old notebooks only need minimal changes.
2. **Separate computation from plotting.** `plot_bars_from_banks`,
   `create_scrolls_and_find_connected_scrolls` (which creates its own figures as a
   side effect), and `Bar.create_merged_polygons` all mix the two. Compute functions
   should return data; thin `plot_*` wrappers take an `ax`.
3. **Kill the copy-paste ladders**:
   - `create_polygon_graph` lines 738–806 handle outer boundaries of 2/3/4/5 nodes
     as four near-identical blocks (and silently skip >5). Replace with one loop
     over `outer_poly_boundary` of any length — this also fixes bug 6.
   - `meandergraph_3D.plot_meander_graph_in_3D` has the same disease (4/5/6-point
     polygons as separate blocks); triangulate generically.
   - The repeated "find the radial/channel successor of a node" 4-liner appears
     ~10×; extract `radial_successor(graph, node)` / `channel_successor(graph, node)`
     helpers.
4. **Performance**:
   - `create_graph_from_channel_lines` line 462 matches nodes by float equality on
     coordinates with `np.where((x == ...) & (y == ...))` — O(N) per node, O(N²)
     total, and fragile. Keep a dict from (cl_number, point index) → node id built
     during node creation instead.
   - `Bar.add_polygon_graphs` compares every polygon pair across adjacent paths
     with `relate()` (O(n²) shapely calls); use `shapely.STRtree` to prune.
   - `create_scrolls_and_find_connected_scrolls` has a 4-deep loop over
     bars×scrolls×scrolls×10; STRtree again.
5. **Data model**: make `Bar` and `Scroll` dataclasses; document node/edge/graph
   attributes in one place; consider an explicit `age`-indexed structure instead of
   the parallel `graph.graph['x']`/`['y']` arrays that must stay in sync with node
   ids (a recurring source of subtle coupling — `remove_high_density_nodes` removes
   nodes but the global arrays keep stale entries and indices).
6. Type hints + numpydoc docstrings on the public API; `logging`/`tqdm` instead of
   bare `print`.

## Phase 5 — Notebooks, docs, CI

1. Make `meandergraph_Mamore_banks_simple_example.ipynb` the single canonical
   bankline example, updated to the refactored API and executed top-to-bottom in the
   new env (replace the interactive `plt.ginput` cell with hardcoded points, keep
   `%matplotlib qt` optional). Update or clearly mark the other notebooks as
   archival.
2. README: real install instructions (conda env + `pip install -e .`), a short
   "concepts" section (channel vs. radial edges, scrolls, bars, cutoff parameters
   `n_points`, `max_dist`, `cutoff_area`), and parameter guidance from the Mamore
   example.
3. GitHub Actions: run the pytest suite + execute the example notebook
   (`nbconvert --execute` with a reduced dataset) on push.
4. Longer term: publish to PyPI, archive on Zenodo for citability (there are AGU
   2022 / ICFS 2023 abstracts in the repo — a JOSS paper may be worth considering).
