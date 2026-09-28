"""
Build polygon graphs (quadrilateral-ish depositional polygons between
consecutive lines / radial trajectories) from a line graph.

Node / edge / graph attributes
-------------------------------
create_polygon_graph's output carries, per node: ``poly`` (the polygon,
Polygon or MultiPolygon), ``age``, ``x``/``y`` (from the inner/near
line-graph node), ``length``, ``width``, ``direction`` (+-1), ``timestep``,
``migr_rate``, ``curv``. Edges carry ``edge_type = 'channel'``.
Graph-level: ``cl_start_nodes`` (one polygon-graph node id per
centerline, marking where its channel-edge chain starts).

create_simple_polygon_graph builds a different, lighter-weight graph:
nodes carry ``node_type`` (``'channel'`` or ``'radial'``), ``x``/``y``,
and (radial nodes only, once a polygon is found) ``poly``/``direction``;
edges carry ``edge_type`` (``'channel'`` or ``'radial'``).
"""
from typing import List, Tuple, Union

import numpy as np
import networkx as nx
from tqdm import trange
from shapely.geometry import Polygon, MultiPolygon, MultiLineString, LineString, JOIN_STYLE, GeometryCollection
from shapely.geometry.polygon import LinearRing

from .graph import find_longitudinal_path, find_radial_path, add_edge_directions_to_bank_graph, radial_successor, channel_successor
from .geometry import fix_geometry, compute_distance, directionOfPoint, ensure_multipolygon

__all__ = [
    "create_polygon_graph", "create_channel_polygon_from_centerline",
    "create_channel_polygon_from_banks", "get_channel_banks",
    "one_step_difference_no_plot", "one_step_difference_no_jump",
    "create_simple_polygon_graph",
]

def create_polygon_graph(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Create graph of polygons from centerline / bankline graph.

    Parameters
    ----------
    graph : directed graph
        Center- or bankline graph to be used.

    Returns
    -------
    poly_graph : directed graph
        Graph with polygons at its nodes.
    """
    # add directionality to nodes
    graph = add_edge_directions_to_bank_graph(graph)

    # create polygon graph
    poly_graph = nx.DiGraph()
    cl_start_nodes = []
    age = 0
    for node in trange(graph.graph['number_of_centerlines'] - 1):
        path = find_longitudinal_path(graph, node)

        ### Get curvature series and timestep from the line graph
        curvature = [graph.nodes[node]['curv'] for node in path]
        ts = graph.nodes[node]['timestep']

        for i in range(len(path) - 1):
            node_1 = path[i]
            node_2 = path[i+1]
            node_4 = radial_successor(graph, node_1)
            node_3 = radial_successor(graph, node_2)
            if (not node_3) and (i < len(path) - 2):
                count = 2
                while node_3 is False:
                    node_2 = path[i+count]
                    node_2_children = list(graph.successors(node_2))
                    if len(node_2_children) > 0:
                        node_3 = radial_successor(graph, node_2)
                        if not node_3:
                            count += 1
                    else:
                        break
            if (not node_4) and node_3 and (i < len(path) - 2):
                node_4 = channel_successor(graph, node_3)
            if node_3 and node_4: # only add a new polygon if there is another centerline
                coords = []
                poly1 = False
                x1 = graph.nodes[node_1]['x']
                x2 = graph.nodes[node_2]['x']
                y1 = graph.nodes[node_1]['y']
                y2 = graph.nodes[node_2]['y']
                width_1 = compute_distance(x1, x2, y1, y2)
                try:
                    outer_poly_boundary = nx.shortest_path(graph, source=node_4, target=node_3)
                except (nx.NetworkXNoPath, nx.NodeNotFound): # if there is no path between node 4 and node 3
                    outer_poly_boundary = []
                if (graph.nodes[node_3]['x'] == graph.nodes[node_4]['x']) and (graph.nodes[node_3]['y'] == graph.nodes[node_4]['y']):
                    # sometimes 'node_3' and 'node_4' are the same node
                    x3, y3 = graph.nodes[node_3]['x'], graph.nodes[node_3]['y']
                    x4, y4 = graph.nodes[node_4]['x'], graph.nodes[node_4]['y']
                    coords = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)]
                    width_2 = 0.0
                    length_1 = compute_distance(x1, x4, y1, y4)
                    length_2 = compute_distance(x2, x3, y2, y3)
                elif len(outer_poly_boundary) >= 2:
                    # outer boundary running from node_3 (near node_2) to node_4
                    # (near node_1) -- the reverse of shortest_path's
                    # source=node_4, target=node_3 order:
                    outer_nodes = outer_poly_boundary[::-1]
                    outer_xy = [(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in outer_nodes]
                    x3, y3 = outer_xy[0]
                    x4, y4 = outer_xy[-1]
                    width_2 = sum(
                        compute_distance(xa, xb, ya, yb)
                        for (xa, ya), (xb, yb) in zip(outer_xy[:-1], outer_xy[1:])
                    )
                    length_1 = compute_distance(x1, x4, y1, y4)
                    length_2 = compute_distance(x2, x3, y2, y3)
                    coords = [(x1, y1), (x2, y2)] + outer_xy + [(x1, y1)]
                    if len(outer_nodes) in (2, 3):
                        # for a 2- or 3-node outer boundary, split into two
                        # triangles when the near edge (node_1-node_2) crosses
                        # the outer-boundary segment closest to node_4:
                        (xs, ys), (xe, ye) = outer_xy[-2], outer_xy[-1]
                        line1 = LineString([[x1, y1], [x2, y2]])
                        line_near_4 = LineString([[xs, ys], [xe, ye]])
                        if line1.intersects(line_near_4):
                            x0 = line1.intersection(line_near_4).x
                            y0 = line1.intersection(line_near_4).y
                            poly1 = Polygon(LinearRing([(x1, y1), (x0, y0), (x4, y4)]))
                            poly2 = Polygon(LinearRing([(x0, y0), (x2, y2)] + outer_xy[:-1]))
                if len(coords) > 0:
                    if not poly1:
                        poly = Polygon(LinearRing(coords))
                    else:
                        poly = MultiPolygon((poly1, poly2))
                    width = 0.5*(width_1 + width_2)
                    length = 0.5*(length_1 + length_2)
                    direction_14 = graph[node_1][node_4]['direction']
                    direction_23 = graph[node_2][node_3]['direction']
                    x1 = graph.nodes[node_1]['x']
                    y1 = graph.nodes[node_1]['y']
                    x2 = graph.nodes[node_2]['x']
                    y2 = graph.nodes[node_2]['y']
                    x3 = graph.nodes[node_3]['x']
                    y3 = graph.nodes[node_3]['y']
                    x4 = graph.nodes[node_4]['x']
                    y4 = graph.nodes[node_4]['y']
                    node_1_coords = np.array([x1, y1])
                    node_2_coords = np.array([x2, y2])
                    node_3_coords = np.array([x3, y3])
                    node_4_coords = np.array([x4, y4])
                    dist_14 = np.linalg.norm(node_1_coords - node_4_coords)
                    dist_23 = np.linalg.norm(node_2_coords - node_3_coords)
                    if direction_23 != direction_14:
                        if dist_14 >= dist_23:
                            direction = direction_14
                        else:
                            direction = direction_23
                    else:
                        direction = direction_23
                    curvature_12 = 0.5*(curvature[i] + curvature[i+1])
                    if poly.is_valid: # add node only if polygon is valid
                        poly_graph.add_node(path[i], poly = poly, age = age, x = x1, y = y1, length = length, width = width, direction = direction, timestep = ts, migr_rate = 0.5*(dist_14 + dist_23)/ts, curv = curvature_12)
                    else:
                        poly = fix_geometry(poly)
                        poly_graph.add_node(path[i], poly = poly, age = age, x = x1, y = y1, length = length, width = width, direction = direction, timestep = ts, migr_rate = 0.5*(dist_14 + dist_23)/ts, curv = curvature_12)
                    if i == 0:
                        if poly.is_valid:
                            cl_start_nodes.append(path[i])
                        else:
                            poly = fix_geometry(poly)
                            poly_graph.add_node(path[i], poly = poly, age = age, x = x1, y = y1, length = length, width = width, direction = direction, timestep = ts, migr_rate = 0.5*(dist_14 + dist_23)/ts, curv = curvature_12)
                            cl_start_nodes.append(path[i])
            else:
                if i == 0: # something is needed at the beginning of the centerline even when there is no 'node_3' or 'node_4', so we just make up a polygon
                    x1 = graph.nodes[node_1]['x']
                    y1 = graph.nodes[node_1]['y']
                    x2 = graph.nodes[node_2]['x']
                    y2 = graph.nodes[node_2]['y']
                    x3 = x2
                    y3 = y2 + 1.0
                    x4 = x1
                    y4 = y1 + 1.0
                    coords = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)]
                    poly = Polygon(LinearRing(coords))
                    if not poly.is_valid: # fix poly
                        poly = poly.buffer(0) # fix the invalid polygon
                    poly_graph.add_node(path[i], poly = poly, age = age, x = x1, y = y1,
                                        length = 1.0, width = compute_distance(x1, x2, y1, y2))
                    cl_start_nodes.append(path[i])
        for i in range(len(path) - 2): # add graph edges
            if (path[i] in poly_graph) and (path[i+1] in poly_graph):
                poly_graph.add_edge(path[i], path[i+1], edge_type = 'channel')
        # need to reconnect broken centerline paths in polygon graph:
        path1 = np.array(find_longitudinal_path(graph, node))
        # path2 = find_longitudinal_path(poly_graph, cl_start_nodes[-1])
        sink_nodes = [] # find sink nodes
        for n in poly_graph.nodes:
            if poly_graph.out_degree(n) == 0:
                sink_nodes.append(n)
        source_nodes = [] # find source nodes
        for n in poly_graph.nodes:
            if poly_graph.in_degree(n) == 0:
                source_nodes.append(n)
        source_nodes_in_path = []
        sink_nodes_in_path = []
        for n in source_nodes:
            if n in path1:
                ind = np.where(path1 == n)[0][0]
                source_nodes_in_path.append(ind)
        for n in sink_nodes:
            if n in path1:
                ind = np.where(path1 == n)[0][0]
                sink_nodes_in_path.append(ind)
        inds = np.sort(sink_nodes_in_path + source_nodes_in_path)[1:-1]
        s_nodes = inds[::2]
        e_nodes = inds[1::2]
        for j in range(len(s_nodes)):
            poly_graph.add_edge(path1[s_nodes[j]], path1[e_nodes[j]], edge_type = 'channel')
        age += 1
    poly_graph.graph['cl_start_nodes'] = cl_start_nodes # store start nodes
    return poly_graph


def create_channel_polygon_from_centerline(x: np.ndarray, y: np.ndarray, W: float) -> Polygon:
    """
    Create a channel polygon from the centerline coordinates.

    Parameters
    ----------
    x : 1D array
        x-coordinates of channel centerline.
    y : 1D array
        y-coordinates of channel centerline.
    W : float
        Channel width.

    Returns
    -------
    ch : Polygon
        Shapely polygon that corresponds to the channel.
    """

    xm, ym = get_channel_banks(x, y, W)
    coords = []
    for i in range(len(xm)):
        coords.append((xm[i],ym[i]))
    ch = Polygon(LinearRing(coords))
    if not ch.is_valid:
        ch = fix_geometry(ch)
    return ch


def create_channel_polygon_from_banks(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> Polygon:
    """
    Create a channel polygon from the bankline coordinates.

    Parameters
    ----------
    x1 : 1D array
        x-coordinates of first bank.
    y1 : 1D array
        y-coordinates of first bank.
    x2 : 1D array
        x-coordinates of second bank.
    y2 : 1D array
        y-coordinates of second bank.

    Returns
    -------
    ch : Polygon
        Shapely polygon that corresponds to the channel.
    """

    xm = np.hstack((x1,x2[::-1]))
    ym = np.hstack((y1,y2[::-1]))
    coords = []
    for i in range(len(xm)):
        coords.append((xm[i], ym[i]))
    ch = Polygon(LinearRing(coords))
    if not ch.is_valid:
        ch = fix_geometry(ch)
    return ch


def get_channel_banks(x: np.ndarray, y: np.ndarray, W: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find coordinates of channel banks, given a centerline and a channel width.

    Parameters
    ----------
    x : 1D array
        x-coordinates of centerline.
    y : 1D array
        y-coordinates of centerline.
    W : float
        Channel width.

    Returns
    -------
    xm : 1D array
        x-coordinates of channel (both banks)
    ym : 1D array
        y-coordinates of channel (both banks)
    """

    x1 = x.copy()
    y1 = y.copy()
    x2 = x.copy()
    y2 = y.copy()
    ns = len(x)
    dx = np.diff(x); dy = np.diff(y) 
    ds = np.sqrt(dx**2+dy**2)
    x1[:-1] = x[:-1] + 0.5*W*np.diff(y)/ds
    y1[:-1] = y[:-1] - 0.5*W*np.diff(x)/ds
    x2[:-1] = x[:-1] - 0.5*W*np.diff(y)/ds
    y2[:-1] = y[:-1] + 0.5*W*np.diff(x)/ds
    x1[ns-1] = x[ns-1] + 0.5*W*(y[ns-1]-y[ns-2])/ds[ns-2]
    y1[ns-1] = y[ns-1] - 0.5*W*(x[ns-1]-x[ns-2])/ds[ns-2]
    x2[ns-1] = x[ns-1] - 0.5*W*(y[ns-1]-y[ns-2])/ds[ns-2]
    y2[ns-1] = y[ns-1] + 0.5*W*(x[ns-1]-x[ns-2])/ds[ns-2]
    xm = np.hstack((x1,x2[::-1]))
    ym = np.hstack((y1,y2[::-1]))
    return xm, ym


def one_step_difference_no_plot(ch1: Polygon, ch2: Polygon, cutoff_area: float) -> Tuple[Union[Polygon, MultiPolygon], MultiPolygon, MultiPolygon, Union[Polygon, MultiPolygon], List[Polygon]]:
    """
    Create polygons from one time step of channel migration, as defined by two consecutive channel polygons, without plotting them.

    Parameters
    ----------
    ch1 : Polygon 
        Shapely polygon for first channel.
    ch2 : Polygon
        Shapely polygon for second channel.
    cutoff_area : float
        Maximum continuous area (created through channel bank movement in one timestep) that is still considered a bar and not a cutoff.

    Returns
    -------
    ch1 : Polygon
        First channel that has been updated with any potential 'jump' areas.
    bar : MultiPolygon
        The depositional bars that result from the movement of the channel banks.
    erosion : MultiPolygon
        The erosional areas that result from the movement of the channel banks.
    jump : MultiPolygon
        Gaps between the two channels when they move more than one channel width during one timestep.
    cutoffs : list
        Shapely polygons of cutoffs.
    """

    ch1 = fix_geometry(ch1)
    ch2 = fix_geometry(ch2)
    both_channels = ch1.union(ch2) # union of the two channels
    if type(both_channels) == MultiPolygon or type(both_channels) == GeometryCollection:
        poly = both_channels.geoms[0]
        for j in range(len(both_channels.geoms)):
            if both_channels.geoms[j].area > poly.area:
                poly = both_channels.geoms[j]
        both_channels = poly
    outline = Polygon(LinearRing(list(both_channels.exterior.coords))) # outline of the union
    jump = outline.difference(both_channels) # gaps between the channels
    bar = ch1.difference(ch2) # the (point) bars are the difference between ch1 and ch2
    bar = fix_geometry(bar)
    jump = fix_geometry(jump)
    bar = ensure_multipolygon(bar.union(jump)) # add gaps to bars
    erosion = ensure_multipolygon(ch2.difference(ch1)) # erosion is the difference between ch2 and ch1
    bar_no_cutoff = []
    for geom in bar.geoms:
        if type(geom) == Polygon:
            bar_no_cutoff.append(geom)
    # bar_no_cutoff = list(bar.geoms) # create list of bars (cutoffs will be removed later)
    erosion_no_cutoff = []
    if type(erosion) == MultiPolygon:
        for geom in erosion.geoms:
            if type(geom) == Polygon:
                erosion_no_cutoff.append(geom)
    elif type(erosion) == Polygon:
        erosion_no_cutoff.append(erosion)
    # erosion_no_cutoff = list(erosion.geoms) # create list of eroded areas (cutoffs will be removed later)
    if type(jump)==MultiPolygon: # create list of gap polygons (if there is more than one gap)
        jump_no_cutoff = list(jump.geoms)
    else:
        jump_no_cutoff = jump
    cutoffs = []
    for b in bar.geoms:
        if b.area>cutoff_area: # look for cutoffs
            bar_no_cutoff.remove(b) # remove cutoff from list of bars
            for e in erosion.geoms: # remove 'fake' erosion related to cutoffs
                if b.intersects(e): # if bar intersects erosional area
                    if type(b.intersection(e))==MultiLineString:
                        if e in erosion_no_cutoff:
                            erosion_no_cutoff.remove(e)
            # deal with gaps between channels:
            if type(jump)==MultiPolygon:
                for j in jump.geoms:
                    if b.intersects(j):
                        if (type(j.intersection(b))==Polygon) and (j.area>0.3*cutoff_area):
                            jump_no_cutoff.remove(j) # remove cutoff-related gap from list of gaps
                            cutoffs.append(b.symmetric_difference(b.intersection(j))) # collect cutoff
            if type(jump)==Polygon:
                if b.intersects(jump):
                    if type(jump.intersection(b))==Polygon:
                        jump_no_cutoff = []
                        cutoffs.append(b.symmetric_difference(b.intersection(jump))) # collect cutoff
    bar = MultiPolygon(bar_no_cutoff)
    erosion = MultiPolygon(erosion_no_cutoff)
    if type(jump_no_cutoff)==list:
        jump = MultiPolygon(jump_no_cutoff)
    ch1 = ch1.union(jump)
    eps = 0.1 # this is needed to get rid of 'sliver geometries' - 
    ch1 = ch1.buffer(eps, 1, join_style=JOIN_STYLE.mitre).buffer(-eps, 1, join_style=JOIN_STYLE.mitre)
    return ch1, bar, erosion, jump, cutoffs


def one_step_difference_no_jump(ch1: Polygon, ch2: Polygon, cutoff_area: float) -> Tuple[Polygon, MultiPolygon, MultiPolygon, List[Polygon]]:
    """
    Create polygons from one time step of channel migration, as defined by two consecutive channel polygons, without plotting them.

    Parameters
    ----------
    ch1 : Polygon 
        Shapely polygon for first channel.
    ch2 : Polygon
        Shapely polygon for second channel.
    cutoff_area : float
        Maximum continuous area (created through channel bank movement in one timestep) that is still considered a bar and not a cutoff.

    Returns
    -------
    ch1 : Polygon
        First channel that has been updated with any potential 'jump' areas.
    bar : MultiPolygon
        The depositional bars that result from the movement of the channel banks.
    erosion : MultiPolygon
        The erosional areas that result from the movement of the channel banks.
    jump : MultiPolygon
        Gaps between the two channels when they move more than one channel width during one timestep.
    cutoffs : list
        Shapely polygons of cutoffs.
    """
 
    bar = ensure_multipolygon(ch1.difference(ch2)) # the (point) bars are the difference between ch1 and ch2
    erosion = ensure_multipolygon(ch2.difference(ch1)) # erosion is the difference between ch2 and ch1
    bar_no_cutoff = []
    cutoffs = []
    for geom in bar.geoms:
        if type(geom) == Polygon:
            bar_no_cutoff.append(geom)
    erosion_no_cutoff = []
    for geom in erosion.geoms:
        if type(geom) == Polygon:
            erosion_no_cutoff.append(geom)
    for b in bar.geoms:
        if b.area>cutoff_area: # look for cutoffs
            bar_no_cutoff.remove(b) # remove cutoff from list of bars
            cutoffs.append(b)
            for e in erosion.geoms: # remove 'fake' erosion related to cutoffs
                if b.intersects(e): # if bar intersects erosional area
                    if type(b.intersection(e))==MultiLineString:
                        if e in erosion_no_cutoff:
                            erosion_no_cutoff.remove(e)
            # deal with gaps between channels:
            # if type(jump)==MultiPolygon:
            #     for j in jump.geoms:
            #         if b.intersects(j):
            #             if (type(j.intersection(b))==Polygon) and (j.area>0.3*cutoff_area):
            #                 jump_no_cutoff.remove(j) # remove cutoff-related gap from list of gaps
            #                 cutoffs.append(b.symmetric_difference(b.intersection(j))) # collect cutoff
            # if type(jump)==Polygon:
            #     if b.intersects(jump):
            #         if type(jump.intersection(b))==Polygon:
            #             jump_no_cutoff = []
            #             cutoffs.append(b.symmetric_difference(b.intersection(jump))) # collect cutoff
    bar = MultiPolygon(bar_no_cutoff)
    erosion = MultiPolygon(erosion_no_cutoff)
    eps = 0.1 # this is needed to get rid of 'sliver geometries' - 
    ch1 = ch1.buffer(eps, 1, join_style=JOIN_STYLE.mitre).buffer(-eps, 1, join_style=JOIN_STYLE.mitre)
    return ch1, bar, erosion, cutoffs


def create_simple_polygon_graph(bank_graph: nx.DiGraph, X: list) -> nx.DiGraph:
    """
    Make a polygon graph consisting of nodes and edges

    Parameters
    ----------
    bank_graph : directed graph
        Bankline graph.
    X : list 
        x coordinate arrays.

    Returns
    -------
    graph : directed graph
        Simple polygon graph
    """
    graph = nx.DiGraph(number_of_centerlines = bank_graph.graph['number_of_centerlines']) # directed graph
    graph.add_nodes_from(bank_graph, node_type = 'channel') # add nodes
    # add radial edges:
    path = find_longitudinal_path(bank_graph, bank_graph.graph['start_nodes'][0])
    for node in path:
        radial_path, dummy = find_radial_path(bank_graph, node)
        edges = []
        for i in range(len(radial_path)-1):
            edges.append((radial_path[i], radial_path[i+1]))
        graph.add_edges_from(edges, edge_type = 'radial')
        for n in radial_path:
            graph.nodes[n]['node_type'] = 'radial'

    # add longitudinal edges:
    start_nodes = []
    for node in range(0, bank_graph.graph['number_of_centerlines']):
        path = find_longitudinal_path(bank_graph, node)
        edges = []
        for i in range(len(path)-1):
            edges.append((path[i], path[i+1]))
        graph.add_edges_from(edges, edge_type = 'channel')
        start_nodes.append(node)
    graph.graph['start_nodes'] = start_nodes

    # add x and y coordinates:
    x = []
    y = []
    radial_nodes = []
    for n in graph.nodes:
        graph.nodes[n]['x'] = bank_graph.nodes[n]['x']
        graph.nodes[n]['y'] = bank_graph.nodes[n]['y']
        x.append(bank_graph.nodes[n]['x'])
        y.append(bank_graph.nodes[n]['y'])
        if graph.nodes[n]['node_type'] == 'radial':
            radial_nodes.append(n)
    graph.graph['x'] = np.array(x)
    graph.graph['y'] = np.array(y)

    # create polygons:
    polys = []
    for node in trange(bank_graph.graph['number_of_centerlines']):
        path = find_longitudinal_path(graph, node)
        path1 = [] 
        for n in path:
            if n in radial_nodes:
                path1.append(n)
        path = path1
        for i in range(len(path) - 1):
            node_1 = path[i]
            node_2 = path[i+1]
            node_4 = radial_successor(graph, node_1)
            node_3 = radial_successor(graph, node_2)
            if node_3 and node_4: # nodes 1, 2, 3
                x1 = graph.nodes[node_1]['x']
                y1 = graph.nodes[node_1]['y']
                x2 = graph.nodes[node_2]['x']
                y2 = graph.nodes[node_2]['y']
                x3 = graph.nodes[node_3]['x']
                y3 = graph.nodes[node_3]['y']
                x4 = graph.nodes[node_4]['x']
                y4 = graph.nodes[node_4]['y']
                node_1_coords = np.array([x1, y1])
                node_2_coords = np.array([x2, y2])
                node_3_coords = np.array([x3, y3])
                node_4_coords = np.array([x4, y4])
                dist_14 = np.linalg.norm(node_1_coords - node_4_coords)
                dist_23 = np.linalg.norm(node_2_coords - node_3_coords)
                direction_23 = directionOfPoint(x1, y1, x2, y2, x3, y3)
                direction_14 = directionOfPoint(x1, y1, x2, y2, x4, y4)
                if direction_23 != direction_14:
                    if dist_14 >= dist_23:
                        direction = direction_14
                    else:
                        direction = direction_23
                else:
                    direction = direction_23
                coords = []
                inner_poly_boundary = nx.shortest_path(graph, source=node_1, target=node_2)
                try:
                    outer_poly_boundary = nx.shortest_path(graph, source=node_4, target=node_3)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    outer_poly_boundary = []
                if len(outer_poly_boundary) > 0:
                    x = []
                    y = []
                    for n in inner_poly_boundary:
                        x.append(graph.nodes[n]['x'])
                        y.append(graph.nodes[n]['y'])
                    for n in outer_poly_boundary[::-1]:
                        x.append(graph.nodes[n]['x'])
                        y.append(graph.nodes[n]['y'])
                    coords = []
                    for p in range(len(x)):
                        coords.append((x[p], y[p]))
                    poly = Polygon(LinearRing(coords))
                    polys.append(poly)
                    graph.nodes[node_1]['poly'] = poly
                    graph.nodes[node_1]['direction'] = direction
            else:
                graph.nodes[node_1]['poly'] = None
    return graph


