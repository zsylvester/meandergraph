"""
'Scroll' (one-timestep depositional polygon) and 'Bar' (connected set of
scrolls) objects, and the functions that build them from a pair of
bankline graphs.
"""
import random
import numpy as np
import networkx as nx
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, LineString, JOIN_STYLE
from shapely.ops import unary_union
from shapely.errors import GEOSException

from .graph import find_longitudinal_path, find_next_node
from .geometry import fix_geometry, compute_distance
from .polygons import create_polygon_graph
from .plot import plot_bars_from_banks, fill_polygon

__all__ = [
    "merge_polygons", "find_sparse_inds",
    "create_scrolls_and_find_connected_scrolls",
    "create_polygon_graphs_and_bar_graphs",
    "polygon_width_and_length", "add_polygon_width_and_length",
    "Bar", "Scroll",
]

def merge_polygons(graph, nodes, sparse_inds, polys):
    """
    Merge polygons.

    Parameters
    ----------
    graph : directed graph
        Bar graph; graph containing bar objects
    sparse_inds : list 
        List of sparse indicies
    polys : list
        List of shapely polygons
        
    Returns
    -------
    polys : list 
        List of merged polygons
    """

    inds = np.arange(len(nodes))
    poly = graph.nodes[nodes[0]]['poly']
    for i in range(len(sparse_inds)-1):
        for j in inds[sparse_inds[i]+1 : sparse_inds[i+1]+1]:
            if (j == sparse_inds[i]+1) and (i!=0):
                poly = graph.nodes[nodes[j]]['poly']
            else:
                poly = poly.union(graph.nodes[nodes[j]]['poly'])
        polys.append(poly)
    return polys


def find_sparse_inds(graph, nodes, min_area):
    """
    Make a list of sparse indicies.

    Parameters
    ----------
    graph : directed graph
        Bar graph; graph containing bar objects
    nodes : list 
        List of nodes
    min_area : float
        Minimum allowable area of a shapely polygon
        
    Returns
    -------
    sparse_inds : list
        List of sparse indices
    """
    areas = []
    for node in nodes:
        area = graph.nodes[node]['poly'].area
        areas.append(area)
    running_sum = 0
    sparse_inds = [0]
    for i in range(len(areas)):
        running_sum += areas[i]
        if running_sum > min_area:
            sparse_inds.append(i)
            running_sum = 0
    sparse_inds.append(len(nodes))
    return sparse_inds


def create_scrolls_and_find_connected_scrolls(graph1, graph2, cutoff_area):
    """
    Make a list of 'scroll' objects and plot them.

    Parameters
    ----------
    graph1 : directed graph
        Bankline graph.
    graph2 : directed graph
        Bankline graph.
    cutoff_area : float
        Maximum continuous area (created through channel bank movement in one timestep) that is still considered a bar and not a cutoff.
        
    Returns
    -------
    scrolls : list
        List of 'scroll' objects
    scroll_ages : list
        List of ages corresponding to the 'scroll' objects
    cutoffs : list
        Shapely polygons that represent cutoffs.
    all_bars_graph: directed graph
        A graph containing all of the 'bar' objects.
    """
    # create scrolls
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # bars, chs, all_chs, jumps, cutoffs = plot_bars2(graph, cutoff_area, ax1, W)
    bars, chs, all_chs, jumps, cutoffs = plot_bars_from_banks(graph1, graph2, cutoff_area, ax1)

    # remove cutoffs from list of scrolls of same age:
    new_bars = []
    for bar in bars:
        if type(bar) == MultiPolygon:
            bar = MultiPolygon([P for P in bar.geoms if P.area < cutoff_area])
        new_bars.append(bar) 
    bars = new_bars

    n_scrolls = [] # number of scrolls in each 'bar'
    for bar in bars:
        if type(bar) == MultiPolygon:
            n_scrolls.append(len(bar.geoms))
        else:
            n_scrolls.append(1)

    scrolls = []
    scroll_ages = []
    count = 0
    single_polygon_count = 0
    for bar in bars:
        if type(bar) == MultiPolygon:
            for scroll in bar.geoms:
                scrolls.append(scroll)
                scroll_ages.append(count)
        else:
            scrolls.append(bar)
            scroll_ages.append(count)
            single_polygon_count += 1
        count += 1

    connections = []
    for n in range(1,10): # outer loop used for fluctuations of centerlines to ensure they are part of the same 'bar'
        for i in trange(n, len(bars)): # start at 'n' so that 'i-n' does not wrap around to the end of the list
            for j in range(n_scrolls[i]):
                for k in range(n_scrolls[i-n]):
                    if (type(bars[i-n]) == MultiPolygon) and (type(bars[i]) == MultiPolygon):
                        if bars[i-n].geoms[k].buffer(1.0).overlaps(bars[i].geoms[j]):
                            connections.append((sum(n_scrolls[:i]) + j, sum(n_scrolls[:i-n]) + k))
                    if (type(bars[i-n]) == Polygon) and (type(bars[i]) == MultiPolygon):
                        if bars[i-n].buffer(1.0).overlaps(bars[i].geoms[j]):
                            connections.append((sum(n_scrolls[:i]) + j, sum(n_scrolls[:i-n]) + 1))
                    if (type(bars[i-n]) == MultiPolygon) and (type(bars[i]) == Polygon):
                        if bars[i-n].geoms[k].buffer(1.0).overlaps(bars[i]):
                            connections.append((sum(n_scrolls[:i]) + 1, sum(n_scrolls[:i-n]) + k))

    all_bars_graph = nx.Graph()
    for i in range(len(connections)):
        all_bars_graph.add_edge(connections[i][0], connections[i][1])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for component in nx.connected_components(all_bars_graph):
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b, 0.5)
        for i in component:
            if scrolls[i].area > 1.0:
                ax.fill(scrolls[i].exterior.xy[0], scrolls[i].exterior.xy[1], facecolor=color, edgecolor='k')
    return scrolls, scroll_ages, cutoffs, all_bars_graph


def create_polygon_graphs_and_bar_graphs(graph1, graph2, all_bars_graph, scrolls, scroll_ages, X1, Y1, X2, Y2, min_area):
    """
    Make a directed graphs that contain shapely polygons representing the banks,
    and add these graphs to 'bar' objects.

    Parameters
    ----------
    graph1 : directed graph
        Bankline graph.
    graph2 : directed graph
        Bankline graph.
    all_bars_graph: directed graph
        A graph containing all of the 'bar' objects.
    scrolls : list
        List of 'scroll' objects
    scroll_ages : list
        List of ages corresponding to the 'scroll' objects
    X1 : list 
        x coordinate arrays.
    Y1 : list
        y coordinate arrays.
    X2 : list 
        x coordinate arrays.
    Y2 : list
        y coordinate arrays.
    min_area : float
        Minimum allowable area of a shapely polygon

    Returns
    -------
    wbars : list
        List of 'bar' objects
    poly_graph_1 : directed graph
        Directed graph containing shapely polygons
    poly_graph_2 : directed_graph
        Directed graph containing shapely polygons
    """
    # create polygon graphs for the banks:
    poly_graph_1 = create_polygon_graph(graph1)
    poly_graph_2 = create_polygon_graph(graph2)
    # create list of Bar objects:
    wbars = []
    count = 0
    for component in nx.connected_components(all_bars_graph):
        wbar = Bar(count, [])
        for i in component:
            if scrolls[i].area > 0:
                # if current scroll intersects the left bank of the same age:
                if scrolls[i].buffer(1.0).intersects(LineString(np.vstack((X2[scroll_ages[i]], Y2[scroll_ages[i]])).T)):
                    bank = 'left'
                elif scrolls[i].buffer(1.0).intersects(LineString(np.vstack((X1[scroll_ages[i]], Y1[scroll_ages[i]])).T)):
                    bank = 'right'
                else:
                    xa = X1[scroll_ages[i]][0]
                    xb = X1[scroll_ages[i]][1]
                    ya = Y1[scroll_ages[i]][0]
                    yb = Y1[scroll_ages[i]][1]
                    x = scrolls[i].centroid.x
                    y = scrolls[i].centroid.y
                    if np.sign((x-xa) * (yb-ya) - (y-ya) * (xb-xa)) < 0:
                        bank = 'left'
                    else:
                        bank ='right'
                wbar.scrolls.append(Scroll(i, scroll_ages[i], bank, scrolls[i], wbar, [])) 
        wbar.create_polygon() # create bar polygon
        if wbar.polygon.area > min_area:
            wbars.append(wbar)
            count += 1
    # add polygon graphs to bars:
    for i in trange(len(wbars)):
        n_right_banks = 0
        n_left_banks = 0
        for scroll in wbars[i].scrolls:
            if scroll.bank == 'left':
                n_left_banks += 1
            if scroll.bank == 'right':
                n_right_banks += 1
        if n_right_banks > n_left_banks:
            wbars[i].add_polygon_graphs(poly_graph_1)
        else:
            wbars[i].add_polygon_graphs(poly_graph_2)   
    return wbars, poly_graph_1, poly_graph_2


def polygon_width_and_length(graph, node):
    """
    Compute the width and length of the polygon that starts at 'node' in a bank graph.

    Parameters
    ----------
    graph : directed graph
        Graph of center- or banklines.
    node : int
        Node at the inner, upstream corner of the polygon.

    Returns
    -------
    width : float
        Mean along-channel width of the polygon.
    length : float
        Mean cross-channel (migration) length of the polygon.
    """

    path = find_longitudinal_path(graph, node)
    node_1 = node
    width_1 = 0
    width_2 = 0
    length_1 = 0
    length_2 = 0
    if len(path) > 1:
        node_2 = path[1]
        node_1_children = list(graph.successors(node_1))
        node_2_children = list(graph.successors(node_2))
        node_3 = False
        node_4 = False
        for n in node_1_children:
            if graph[node_1][n]['edge_type'] == 'radial':
                node_4 = n
        for n in node_2_children:
            if graph[node_2][n]['edge_type'] == 'radial':
                node_3 = n
        if (not node_3) and (len(path) > 2):
            node_2 = path[2]
            node_2_children = list(graph.successors(node_2))
            for n in node_2_children:
                if graph[node_2][n]['edge_type'] == 'radial':
                    node_3 = n
        if (not node_4) and node_3 and (len(path) > 2):
            node_3_children = list(graph.successors(node_3))
            for n in node_3_children:
                if graph[node_3][n]['edge_type'] == 'channel':
                    node_4 = n
        x1 = graph.nodes[node_1]['x']
        x2 = graph.nodes[node_2]['x']
        y1 = graph.nodes[node_1]['y']
        y2 = graph.nodes[node_2]['y']
        width_1 = compute_distance(x1, x2, y1, y2)
        try:
            outer_poly_boundary = nx.shortest_path(graph, source=node_4, target=node_3)
        except (nx.NetworkXNoPath, nx.NodeNotFound): # if there is no path between node 4 and node 3
            outer_poly_boundary = []
        if len(outer_poly_boundary) == 2: # 2 nodes on the outer boundary
            x3 = graph.nodes[node_3]['x']
            x4 = graph.nodes[node_4]['x']
            y3 = graph.nodes[node_3]['y']
            y4 = graph.nodes[node_4]['y']
            width_2 = compute_distance(x3, x4, y3, y4)
            length_1 = compute_distance(x1, x4, y1, y4)
            length_2 = compute_distance(x2, x3, y2, y3)
        if len(outer_poly_boundary) == 3: # 3 nodes on the outer boundary
            x3 = graph.nodes[outer_poly_boundary[2]]['x']
            x4 = graph.nodes[outer_poly_boundary[1]]['x']
            x5 = graph.nodes[outer_poly_boundary[0]]['x']
            y3 = graph.nodes[outer_poly_boundary[2]]['y']
            y4 = graph.nodes[outer_poly_boundary[1]]['y']
            y5 = graph.nodes[outer_poly_boundary[0]]['y']
            width_2 = compute_distance(x3, x4, y3, y4) + compute_distance(x4, x5, y4, y5)
            length_1 = compute_distance(x1, x5, y1, y5)
            length_2 = compute_distance(x2, x3, y2, y3)
    return 0.5*(width_1 + width_2), 0.5*(length_1 + length_2)


def add_polygon_width_and_length(wbars, graph1, graph2):
    for wbar in tqdm(wbars):
        if wbar.scrolls[-1].bank == 'left':
            graph = graph2
        if wbar.scrolls[-1].bank == 'right':
            graph = graph1
        for node in wbar.bar_graph.nodes:
            width, length = polygon_width_and_length(graph, node)
            wbar.bar_graph.nodes[node]['width'] = width
            wbar.bar_graph.nodes[node]['length'] = length


class Bar:
    def __init__(self, number, scrolls):
        self.number = number
        self.scrolls = scrolls
    def plot(self, ax, color):
        """
        Make bar and scroll plot.

        Parameters
        ----------
        ax : int
            Axes for plotting
        color : str
            String correponding to desired color for plot
        """
        fill_polygon(self.polygon, ax, facecolor='w', edgecolor='k', linewidth=2)
        for scroll in self.scrolls:
            fill_polygon(scroll.polygon, ax, facecolor=color, edgecolor='k', linewidth=0.5)
    def create_polygon(self):
        """
        Create bar polygon from component scrolls
        """
        whole_bar = self.scrolls[0].polygon
        for scroll in self.scrolls:
            # using 'union' can result in topological errors
            whole_bar = unary_union([whole_bar, scroll.polygon])
        whole_bar = whole_bar.buffer(0.1, 1, join_style=JOIN_STYLE.mitre).buffer(-0.1, 1, join_style=JOIN_STYLE.mitre)
        self.polygon = whole_bar
    def add_polygon_graphs(self, graph):
        """
        Add polygon graphs to bars.

        Parameters
        ----------
        graph : directed graph
            Polygon graph
        """
        # the input 'graph' has to be a polygon graph
        nodes = []
        polys = []
        for i in range(len(graph.graph['cl_start_nodes'])):
            for scroll in self.scrolls:
                if scroll.age == i:
                    path = find_longitudinal_path(graph, graph.graph['cl_start_nodes'][i])
                    for node in path:
                        if scroll.polygon.overlaps(graph.nodes[node]['poly']) or scroll.polygon.contains(graph.nodes[node]['poly']):
                            if graph.nodes[node]['poly'].is_valid:
                                try:
                                    poly = self.polygon.intersection(graph.nodes[node]['poly'])
                                except GEOSException:
                                    poly = graph.nodes[node]['poly']
                            else:
                                poly = self.polygon.intersection(fix_geometry(graph.nodes[node]['poly']))
                            if graph.nodes[node]['poly'].difference(scroll.polygon).area > 0: #accounting for intra-point bar erosion
                                poly = scroll.polygon.intersection(graph.nodes[node]['poly'])
                            if poly.area > 0:
                                nodes.append(node)
                                polys.append(poly)
                                scroll.small_polygons.append(poly)               
        bar_graph = graph.subgraph(nodes).copy() # copy the nodes from the input polygon graph that are relevant for the bar
        for i in range(len(nodes)): # add polygons as attributes
            bar_graph.nodes[nodes[i]]['poly'] = polys[i]
        bar_radial_graph = nx.DiGraph() # create radial graph for bar
        bar_radial_graph.add_nodes_from(bar_graph)
        source_nodes = [] # find source nodes
        source_node_ages = []
        for node in bar_graph.nodes:
            if bar_graph.in_degree(node) == 0:
                source_nodes.append(node)
                source_node_ages.append(bar_graph.nodes[node]['age'])
        sort_inds = np.argsort(source_node_ages)
        source_nodes = np.array(source_nodes)[sort_inds] # sort source nodes by age      
        for i in range(len(source_nodes) - 1):
            path1 = find_longitudinal_path(bar_graph, source_nodes[i])
            path2 = find_longitudinal_path(bar_graph, source_nodes[i+1])
            for node1 in path1:
                for node2 in path2:
                    poly1 = bar_graph.nodes[node1]['poly']
                    poly2 = bar_graph.nodes[node2]['poly']
                    if not poly1.is_valid:
                        poly1 = fix_geometry(poly1)
                    if not poly2.is_valid:
                        poly2 = fix_geometry(poly2)
                    try:
                        if poly1.relate(poly2) == 'FF2F11212':
                            bar_radial_graph.add_edge(node1, node2, edge_type = 'radial')
                            bar_radial_graph.nodes[node1]['x'] = poly1.centroid.x
                            bar_radial_graph.nodes[node1]['y'] = poly1.centroid.y
                            bar_radial_graph.nodes[node2]['x'] = poly2.centroid.x
                            bar_radial_graph.nodes[node2]['y'] = poly2.centroid.y
                    except GEOSException:
                        pass
        self.bar_graph = bar_graph
        self.bar_radial_graph = bar_radial_graph
    def plot_polygons(self, ax, plot_graphs):
        """
        Plot bar polygons 

        Parameters
        ----------
        ax : int
            Axes for plotting
        plot_graphs : boolean
            True or False; determines whether to plot bar polygons on top of scroll polygons
        """
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b, 0.5)
        for scroll in self.scrolls: # plot cropped polygons
            for small_polygon in scroll.small_polygons:
                fill_polygon(small_polygon, ax, facecolor=color, edgecolor='k', linewidth = 0.5)
        if plot_graphs:
            for (s, e) in tqdm(self.bar_graph.edges):
                ax.plot([self.bar_graph.nodes[s]['poly'].centroid.x, self.bar_graph.nodes[e]['poly'].centroid.x],
                        [self.bar_graph.nodes[s]['poly'].centroid.y, self.bar_graph.nodes[e]['poly'].centroid.y], 
                        'r', linewidth = 1)
            for (s, e) in tqdm(self.bar_radial_graph.edges):
                ax.plot([self.bar_radial_graph.nodes[s]['x'], self.bar_radial_graph.nodes[e]['x']],
                        [self.bar_radial_graph.nodes[s]['y'], self.bar_radial_graph.nodes[e]['y']], 
                        'g', linewidth = 1)
        fill_polygon(self.polygon, ax, facecolor='none', edgecolor='k', linewidth = 2)
    def create_merged_polygons(self, ax, min_area):
        """
        Create merged bar polygons 

        Parameters
        ----------
        ax : int
            Axes for plotting
        min_area : float
            Minimum allowable area of a shapely polygon
        """ 
        source_nodes = []
        for node in self.bar_radial_graph.nodes:
            if self.bar_radial_graph.in_degree(node) == 0:
                source_nodes.append(node)
        polys = []
        for source_node in source_nodes:
            start_node = source_node
            nodes = list(nx.dfs_preorder_nodes(self.bar_radial_graph, start_node))
            bifurcations = []
            for node in nodes:
                if len(list(self.bar_radial_graph.successors(node))) == 2:
                    bifurcations.append(node)
            nodes = find_next_node(self.bar_radial_graph, start_node)
            sparse_inds = find_sparse_inds(self.bar_graph, nodes, min_area)
            polys = merge_polygons(self.bar_graph, nodes, sparse_inds, polys)
            for node in bifurcations:
                nodes = find_next_node(self.bar_radial_graph, list(self.bar_radial_graph.successors(node))[1])
                sparse_inds = find_sparse_inds(self.bar_graph, nodes, min_area)
                polys = merge_polygons(self.bar_graph, nodes, sparse_inds, polys)
        for poly in polys:
            fill_polygon(poly, ax, facecolor='none', edgecolor='k', linewidth=0.5)
        fill_polygon(self.polygon, ax, facecolor='none', edgecolor='b', linewidth=2)
        self.merged_polygons = polys
    def add_bank_type(self):
        """
        Add bank_type attribute, either 'left' or 'right'
        """
        n_right_banks = 0
        n_left_banks = 0
        for scroll in self.scrolls:
            if scroll.bank == 'left':
                n_left_banks += 1
            if scroll.bank == 'right':
                n_right_banks += 1
        if n_right_banks > n_left_banks:
            self.bank_type = 'right'
        else:
            self.bank_type = 'left'


class Scroll:
    def __init__(self, number, age, bank, polygon, bar, small_polygons):
        self.number = number
        self.age = age
        self.bank = bank # left or right bank
        self.polygon = polygon
        self.bar = bar
        self.small_polygons = small_polygons

