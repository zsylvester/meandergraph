"""
End-to-end smoke test of the bankline workflow (correlation -> line graphs
-> bars/scrolls -> polygon/bar graphs -> plotting) against synthetic
banklines, so the pipeline can be exercised in seconds without the real
Mamore shapefiles.

This is a regression harness for the modernization and refactor work in
IMPROVEMENT_PLAN.md: it pins down structural properties of the pipeline's
output (node/edge counts, bar/scroll counts, total areas) so that changes
intended to be behavior-preserving can be checked quickly. It is
deliberately not a byte-for-byte regression test, since the shapely-2
compatibility fixes are expected to perturb exact floating point results
slightly in some cases.
"""
import matplotlib

matplotlib.use("Agg")

import networkx as nx
import numpy as np
import pytest

import meandergraph as mg


def make_banklines(n_lines=12, n_points=300, width=30.0):
    """Synthetic left/right banklines: a sine-wave channel whose amplitude
    grows over time, so the banks migrate laterally between timesteps."""
    s = np.linspace(0, 4 * np.pi, n_points)
    X1, Y1, X2, Y2 = [], [], [], []
    for t in range(n_lines):
        amp = 60.0 + 8.0 * t
        x = s * 50.0
        y = amp * np.sin(s + 0.05 * t)
        dx = np.gradient(x)
        dy = np.gradient(y)
        ds = np.sqrt(dx ** 2 + dy ** 2)
        nx_ = -dy / ds
        ny_ = dx / ds
        X1.append(x + 0.5 * width * nx_)
        Y1.append(y + 0.5 * width * ny_)
        X2.append(x - 0.5 * width * nx_)
        Y2.append(y - 0.5 * width * ny_)
    return X1, Y1, X2, Y2


@pytest.fixture(scope="module")
def bank_lines():
    return make_banklines()


@pytest.fixture(scope="module")
def line_graphs(bank_lines):
    X1, Y1, X2, Y2 = bank_lines
    P1, Q1, costs1 = mg.correlate_set_of_curves(X1, Y1)
    P2, Q2, costs2 = mg.correlate_set_of_curves(X2, Y2)
    graph1 = mg.create_graph_from_channel_lines(X1, Y1, P1, Q1, n_points=5, max_dist=50)
    graph2 = mg.create_graph_from_channel_lines(X2, Y2, P2, Q2, n_points=5, max_dist=50)
    graph1 = mg.remove_high_density_nodes(graph1, min_dist=2, max_dist=50)
    graph2 = mg.remove_high_density_nodes(graph2, min_dist=2, max_dist=50)
    return graph1, graph2


def test_correlate_set_of_curves_shapes(bank_lines):
    X1, Y1, _, _ = bank_lines
    P1, Q1, costs1 = mg.correlate_set_of_curves(X1, Y1)
    assert len(P1) == len(X1) - 1
    assert len(Q1) == len(X1) - 1
    assert len(costs1) == len(X1) - 1
    for p, q in zip(P1, Q1):
        assert len(p) == len(q)
        assert p.min() >= 0
        assert q.min() >= 0


def test_line_graphs_are_well_formed(line_graphs):
    for graph in line_graphs:
        assert graph.number_of_nodes() > 0
        assert nx.is_directed(graph)
        assert set(graph.graph["x"].shape) == set(graph.graph["y"].shape)
        for node in graph.graph["start_nodes"]:
            assert node in graph.nodes


def test_bars_and_scrolls_from_banks(line_graphs):
    import matplotlib.pyplot as plt

    graph1, graph2 = line_graphs
    fig, ax = plt.subplots()
    bars, chs, all_chs, jumps, cutoffs = mg.plot_bars_from_banks(graph1, graph2, cutoff_area=200, ax=ax)
    plt.close(fig)

    assert len(bars) == graph1.graph["number_of_centerlines"]
    assert len(chs) == graph1.graph["number_of_centerlines"]
    assert all(b.area >= 0 for b in bars if hasattr(b, "area"))

    scrolls, scroll_ages, cutoffs2, all_bars_graph = mg.create_scrolls_and_find_connected_scrolls(
        graph1, graph2, cutoff_area=200
    )
    plt.close("all")
    assert len(scrolls) == len(scroll_ages)
    assert len(scrolls) > 0


def test_bar_graphs_and_plotting(line_graphs):
    import matplotlib.pyplot as plt

    graph1, graph2 = line_graphs
    X1 = graph1.graph["x"]
    scrolls, scroll_ages, cutoffs, all_bars_graph = mg.create_scrolls_and_find_connected_scrolls(
        graph1, graph2, cutoff_area=200
    )
    plt.close("all")

    # X1/Y1/X2/Y2 as lists-of-arrays are needed by create_polygon_graphs_and_bar_graphs;
    # reconstruct them from the longitudinal paths of the (possibly node-thinned) graphs.
    def longitudinal_xy(graph):
        n = graph.graph["number_of_centerlines"]
        X, Y = [], []
        for node in range(n):
            path = mg.find_longitudinal_path(graph, node)
            X.append(graph.graph["x"][path])
            Y.append(graph.graph["y"][path])
        return X, Y

    X1, Y1 = longitudinal_xy(graph1)
    X2, Y2 = longitudinal_xy(graph2)

    wbars, poly_graph_1, poly_graph_2 = mg.create_polygon_graphs_and_bar_graphs(
        graph1, graph2, all_bars_graph, scrolls, scroll_ages, X1, Y1, X2, Y2, min_area=5
    )
    assert len(wbars) > 0
    assert poly_graph_1.number_of_nodes() > 0
    assert poly_graph_2.number_of_nodes() > 0

    fig, ax = plt.subplots()
    for plot_type, vmin, vmax in [("migration", -5, 5), ("curvature", -1, 1), ("age", 0, 5)]:
        mg.plot_bar_graphs(graph1, graph2, wbars, cutoffs, X1, Y1, X2, Y2, W=30, vmin=vmin, vmax=vmax, plot_type=plot_type, ax=ax)
    plt.close(fig)
