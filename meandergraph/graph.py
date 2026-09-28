"""
Build and query the directed line graph (channel + radial edges) from
correlated center- or banklines.

Node / edge / graph attributes
-------------------------------
A line graph built by create_graph_from_channel_lines carries:

- Node attributes: ``x``, ``y`` (coordinates), ``age`` (index of the
  center-/bankline this node lies on), ``curv`` (curvature, added by
  add_curvature_to_line_graph), ``timestep`` (added by
  add_timesteps_to_line_graph).
- Edge attributes: ``edge_type`` (``'channel'`` along one line,
  ``'radial'`` across lines through time); radial edges created in the
  initial n_points-spaced pass also carry ``age``; radial edges get a
  ``direction`` (+-1, erosion/deposition side) once
  add_edge_directions_to_bank_graph has run.
- Graph-level attributes (``graph.graph[...]``): ``number_of_centerlines``,
  ``x``/``y`` (coordinate arrays indexed by node id -- must stay in sync
  with the node set; nodes removed by e.g. remove_high_density_nodes
  leave stale entries here), ``start_nodes`` (nodes where a radial path
  begins), ``cutoff_nodes``, and ``sparse_cutoff_nodes`` (added by
  add_sparse_cutoff_nodes).
"""
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.signal import savgol_filter
import networkx as nx
from tqdm import trange, tqdm
from copy import deepcopy

from .correlation import find_indices, compute_curvature, compute_derivatives
from .geometry import directionOfPoint

logger = logging.getLogger(__name__)

__all__ = [
    "find_radial_path", "find_radial_path_backward", "find_longitudinal_path",
    "create_list_of_start_nodes", "create_graph_from_channel_lines",
    "reconnect_nodes_along_centerline", "remove_high_density_nodes",
    "add_curvature_to_line_graph", "add_timesteps_to_line_graph",
    "find_next_node", "add_sparse_cutoff_nodes", "find_radial_path_2",
    "add_edge_directions_to_bank_graph", "find_cutoff_ages",
    "radial_successor", "channel_successor",
]

def radial_successor(graph, node):
    """
    Find the successor of 'node' reached by a 'radial' edge.

    Parameters
    ----------
    graph : directed graph
        Graph with radial edges defined.
    node : int
        Node whose radial successor is wanted.

    Returns
    -------
    successor : int or False
        The radial successor of 'node', or False if it has none.
    """

    successor = False
    for n in graph.successors(node):
        if graph[node][n]['edge_type'] == 'radial':
            successor = n
    return successor

def channel_successor(graph, node):
    """
    Find the successor of 'node' reached by a 'channel' edge.

    Parameters
    ----------
    graph : directed graph
        Graph with channel edges defined.
    node : int
        Node whose channel successor is wanted.

    Returns
    -------
    successor : int or False
        The channel successor of 'node', or False if it has none.
    """

    successor = False
    for n in graph.successors(node):
        if graph[node][n]['edge_type'] == 'channel':
            successor = n
    return successor

def find_radial_path(graph, node):
    """
    Collect the indices of graph nodes that describe a radial path starting from 'node'.

    Parameters
    ----------
    graph : directed graph 
        Graph with radial edges defined.
    node : int
        Start node of radial path.

    Returns
    -------
    path : list
        Nodes that define the radial path.
    path_ages : list
        Ages of the nodes in the path.
    """

    path = []
    path_ages = []
    path.append(node)
    path_ages.append(graph.nodes[node].get('age'))
    edge_types = []
    for successor_node in graph.successors(node):
        edge_types.append(graph[node][successor_node]['edge_type'])
    while 'radial' in edge_types:
        for successor_node in graph.successors(node):
            if graph[node][successor_node]['edge_type'] == 'radial':
                next_node = successor_node
        path.append(next_node)
        path_ages.append(graph.nodes[next_node].get('age'))
        node = next_node
        edge_types = []
        for successor_node in graph.successors(node):
            edge_types.append(graph[node][successor_node]['edge_type'])
    return path, path_ages


def find_radial_path_backward(graph, node):
    """
    Collect the indices of graph nodes that describe a radial path starting from 'node', going backward.

    Parameters
    ----------
    graph : directed graph 
        Graph with radial edges defined.
    node : int
        Start node of the backward path.

    Returns
    -------
    path : list
        Nodes that define the backward radial path.
    path_ages : list
        Ages of the nodes in the path.
    """

    path = []
    path_ages = []
    path.append(node)
    path_ages.append(graph.nodes[node]['age'])
    edge_types = []
    for predecessor_node in graph.predecessors(node):
        edge_types.append(graph[predecessor_node][node]['edge_type'])
    while 'radial' in edge_types:
        for predecessor_node in graph.predecessors(node):
            if graph[predecessor_node][node]['edge_type'] == 'radial':
                next_node = predecessor_node
        path.append(next_node)
        path_ages.append(graph.nodes[next_node]['age'])
        node = next_node
        edge_types = []
        for predecessor_node in graph.predecessors(node):
            edge_types.append(graph[predecessor_node][node]['edge_type'])
    return path, path_ages


def find_longitudinal_path(graph, node):
    """
    Collect the indices of graph nodes that describe a longitudinal path starting from 'node'.

    Parameters
    ----------
    graph : directed graph 
        Graph with longitudinal edges defined.
    node : int
        Start node of the longitudinal path of interest.

    Returns
    -------
    path: list
        Nodes that define the longituidnal path.
    """

    path = []
    path.append(node)
    edge_types = []
    for successor_node in graph.successors(node):
        edge_types.append(graph[node][successor_node]['edge_type'])
    while 'channel' in edge_types:
        for successor_node in graph.successors(node):
            if graph[node][successor_node]['edge_type'] == 'channel':
                next_node = successor_node
        if next_node not in path: # when working with bars, it is possible to form a cycle; this prevents that
            path.append(next_node)
            node = next_node
            edge_types = []
            for successor_node in graph.successors(node):
                edge_types.append(graph[node][successor_node]['edge_type'])
        else:
            path.append(next_node)
            break
    return path


def create_list_of_start_nodes(graph):
    """
    Find all the nodes in a graph that are starting points for radial paths.

    Parameters
    ----------
    graph: directed graph
        Graph with radial paths defined.

    Returns
    -------
    start_nodes: list
        Nodes that are starting points of radial paths.
    """

    start_nodes = []
    for node in graph.nodes:
        parents = graph.predecessors(node)
        edge_types = []
        for parent in parents:
            edge_types.append(graph[parent][node]['edge_type'])
        if 'radial' not in edge_types:
            start_nodes.append(node)
    return start_nodes


def create_graph_from_channel_lines(X, Y, P, Q, n_points, max_dist, smoothing_factor = 51, remove_cutoff_edges = False, timesteps = None, clean_up_centerlines = True):
    """
    Create directed graph from a set of cghannel center- or bank lines.

    Parameters
    ----------
    X : list
        x coordinates of lines.
    Y : list
        y coordinates of lines.
    P : list
        Arrays of indices of correlated successive pairs of curves (for first curve)
    Q : list
        Arrays of indices of correlated successive pairs of curves (for second curve)
    n_points : int
        Every 'n_points'th point on the first centerline is used to start a radial trajectory
    max_dist : int
        Parameter used to determine where cutoffs have occurred
    smoothing_factor : int (Default = 51)
        Window size in savgol filter, used when computing channel curvature
    remove_cutoff_edges: Boolean (Optional, Default = False)
        Parameter that is used to eliminate edges that correspond to cutoffs
    timesteps: list (Optional, Default = None)
        List containing floating point numbers representing the amount of time (years) between successive longitudinal paths.
        If None the timestep defaults to 1.0.
    clean_up_centerlines: Boolean (Optional, Default = True)
        Parameter that is used to remove nodes that are not properly connected along the centerlines
    Returns
    -------
    graph : directed graph
        Graph that contains all the center- or bank lines and radial lines
    """
    graph = nx.DiGraph(number_of_centerlines = len(X)) # directed graph to store nodes and edges
    cl_indices = [] # list of lists to store *centerline* indices that will be part of the graph
    n_centerlines = len(X)
    for i in range(n_centerlines): # initialize 'cl_indices'
        cl_indices.append([])
    # maps (centerline index, point index along that centerline) -> node id,
    # populated as nodes are created below, so channel edges can be added by
    # a dict lookup instead of a linear (x, y) float-equality search. DTW
    # correlation is many-to-one near sequence boundaries, so more than one
    # radial trajectory can land on the same (centerline, point index); when
    # that happens the *first* trajectory to claim a slot stays canonical
    # (matching what the old coordinate-equality search always resolved to,
    # since it scanned nodes in insertion order) -- hence setdefault, not a
    # plain assignment.
    node_lookup = {}
    # add radial nodes and edges:
    print('add radial nodes and edges...')
    for ind1 in trange(0, len(X[0]), n_points):
        indices, x, y = find_indices(ind1, X, Y, P, Q)
        new_node_inds = np.arange(len(graph), len(graph) + len(indices))
        for i in range(len(indices)):
            cl_indices[i].append(indices[i])
            graph.add_node(new_node_inds[i], x = x[i], y = y[i], age = i, curv = 0)
            node_lookup.setdefault((i, indices[i]), new_node_inds[i])
        for i in range(len(indices)-1):
            graph.add_edge(new_node_inds[i], new_node_inds[i+1], edge_type = 'radial', age = i)
    print('add intermediate trajectories...')
    for cl_number in trange(n_centerlines - 1):
        large_gap_inds = np.where(np.diff(cl_indices[cl_number]) > 2*n_points) # find gaps that are longer than 2 x the number of points
        if len(large_gap_inds) > 0:
            large_inds = large_gap_inds[0]
            diffs = np.diff(cl_indices[cl_number])[large_inds]
            large_inds = np.array(cl_indices[cl_number])[large_inds] + np.round(diffs*0.5).astype('int') # indices of new nodes on centerline
            for ind1 in large_inds:
                indices, x, y = find_indices(ind1, X[cl_number:], Y[cl_number:], P[cl_number:], Q[cl_number:]) # find indices on younger centerlines, using DTW correlation
                new_node_inds = np.arange(len(graph), len(graph) + len(indices)) # create node indices for new nodes
                for i in range(len(indices)):
                    cl_indices[cl_number+i].append(indices[i]) # add indices of new nodes to list of indices of centerline nodes
                    graph.add_node(new_node_inds[i], x = x[i], y = y[i], age = cl_number + i) # add new nodes to graph
                    node_lookup.setdefault((cl_number+i, indices[i]), new_node_inds[i])
                for i in range(len(indices)-1):
                    graph.add_edge(new_node_inds[i], new_node_inds[i+1], edge_type = 'radial') # add new edges to graph
            for i in range(cl_number, n_centerlines):
                cl_indices[i].sort() # sort indices of all nodes along current centerline
    # add edges that represent centerlines:
    x = []
    y = []
    for node in graph.nodes:
        x.append(graph.nodes[node]['x'])
        y.append(graph.nodes[node]['y'])
    x = np.array(x)
    y = np.array(y)
    graph.graph['x'] = x
    graph.graph['y'] = y
    print('add centerline edges...')
    for cl_number in trange(n_centerlines):
        cl_nodes = [node_lookup[(cl_number, i)] for i in cl_indices[cl_number]]
        for i in range(len(cl_nodes) - 1):
            graph.add_edge(cl_nodes[i], cl_nodes[i+1], edge_type = 'channel')
    # a few edges are linking nodes to themselves, and they need to be removed:
    edges_to_be_removed = []
    for (s, e) in graph.edges:
        if s == e:
            edges_to_be_removed.append((s, e))       
    for edge in edges_to_be_removed:
        graph.remove_edge(edge[0], edge[1])
    # find nodes from which radial trajectories start:
    start_nodes = create_list_of_start_nodes(graph)
    # remove edges that correspond to cutoffs:
    edges_to_be_removed = []
    cutoff_nodes = []
    print('collect edges to be removed...')
    for node in tqdm(start_nodes):
        path, path_ages = find_radial_path(graph, node)
        ds = [] # distances between consecutive radial nodes
        for i in range(len(path)-1): # compute and store the distances
            ds.append(((x[path[i+1]] - x[path[i]])**2 + (y[path[i+1]] - y[path[i]])**2)**0.5)
        if len(ds) > 1:
            # if there is at least one place where the increase in the distance between nodes is larger than 'max_dist':
            if np.max(np.abs(np.diff(ds))) > max_dist: 
                inds = list(np.where(np.diff(ds) > max_dist)[0] + 1) # indices where the difference is larger than max_dist
                if  ds[0] > max_dist: # if the first distance is larger than 'max_dist'
                    inds = [0] + inds # the first node needs to be added to the list of cutoff nodes
                for ind in inds: # collect cutoff-related edges and cutoff nodes for all indices
                    if (path[ind], path[ind+1]) not in edges_to_be_removed:
                        edges_to_be_removed.append((path[ind], path[ind+1]))
                        cutoff_nodes.append(path[ind+1])
    graph.graph['cutoff_nodes'] = cutoff_nodes
    if remove_cutoff_edges: # only remove cutoff edges if you want to
        for edge in edges_to_be_removed:
            graph.remove_edge(edge[0], edge[1])   
    # redo list of nodes from which radial trajectories start
    start_nodes = create_list_of_start_nodes(graph)
    graph.graph['start_nodes'] = start_nodes
    # clean up a few nodes that are not properly connected up along the centerlines:
    if clean_up_centerlines:
        cl_nodes = []
        for cl_number in trange(len(X)): # collect all nodes that are connected along the centerlines
            path = find_longitudinal_path(graph, cl_number)
            cl_nodes += path
        for node in set(cl_nodes) ^ set(graph.nodes): # if a node is not in 'cl_nodes', remove it from the graph
            graph.remove_node(node)
            if node in graph.graph['start_nodes']: # remove the node from 'start_nodes' as well
                graph.graph['start_nodes'].remove(node)
    # compute curvature and add to graph nodes
    add_curvature_to_line_graph(graph, smoothing_factor=smoothing_factor)
    # add 'timestep' attribute to graph nodes
    if timesteps is not None:
        try:
            add_timesteps_to_line_graph(graph, timesteps)
        except:
            print('Error: timesteps should be a list of datetime objects (YYYYMMDD)')
    else:
        add_timesteps_to_line_graph(graph, np.ones(len(X)))
    return graph


def reconnect_nodes_along_centerline(graph1, graph2, cl_number):
    """
    Reconnect nodes along a centerline in graph2, based on nodes along the same centerline in graph1

    Parameters
    ----------

    graph1 : directed graph
        Graph that contains the intact center- or banklines defined.
    graph2 : dirceted graph
        Graph that has center- or banklines that need to be reconnected.
    cl_number : int
        Index of center- or bankline in 'graph1'.
    """

    path = find_longitudinal_path(graph1, cl_number)
    cl_nodes = []
    for node in path:
        if node in graph2:
            cl_nodes.append(node)
    for i in range(len(cl_nodes) - 1):
        if (cl_nodes[i], cl_nodes[i+1]) not in graph2.edges:
            graph2.add_edge(cl_nodes[i], cl_nodes[i+1], edge_type = 'channel')


def remove_high_density_nodes(graph1, min_dist, max_dist):
    """
    Remove nodes and edges where radial lines are too dense (especially after cutoffs).

    Parameters
    ----------
    graph1 : directed graph
        Graph that has some nodes that are too close to each other, due to cutoffs.
    min_dist : int
        Distances between nodes that are smaller than this will result in removing one of the nodes.
    max_dist : int
        Maximum distance between nodes; a node will be not be removed if it results in a distance larger than this.

    Returns
    -------
    graph2 : directed graph

    Example
    -------
    graph = mg.remove_high_density_nodes(graph, min_dist = 10, max_dist = 30)
    """

    graph2 = deepcopy(graph1)
    for cl_number in trange(graph1.graph['number_of_centerlines']):
        path = find_longitudinal_path(graph2, cl_number)
        # compute distances between nodes along centerline (we need only 'ds')
        dx, dy, ds, s = compute_derivatives(graph2.graph['x'][path], graph2.graph['y'][path])
        small_inds = np.where(ds < min_dist)[0] # indices of distances that are too small
        nodes_to_be_removed = [] # for storing nodes that need to be removed
        if len(small_inds) > 0:
            if small_inds[0] != 0: 
                small_inds = np.hstack((0, small_inds)) # add first index
            if small_inds[-1] != len(ds) - 1:
                small_inds = np.hstack((small_inds, len(ds) - 1)) # add last index
            inds1 = np.where(np.diff(small_inds)>1)[0] + 1 # indices where new segments with short distances start
            inds2 = inds1-1
            inds2 = inds2[1:] # indices where segments with short distances end
            for i in range(len(inds2)):
                if inds1[i] == inds2[i]: # if there is only one node that needs to be removed 
                    nodes_to_be_removed.append(small_inds[inds1[i]])
                else:
                    dist = 0 # cumulative distance along nodes
                    # for each continuous segment with short distances:
                    for small_ind in range(small_inds[inds1[i]]+1, small_inds[inds2[i]]+1):
                        dist += ds[small_ind]
                        if dist < min_dist:
                            nodes_to_be_removed.append(small_ind)
                        else:
                            dist = 0 # reset cumulative distance
            if len(nodes_to_be_removed) > 0:
                nodes = np.array(path)[np.array(nodes_to_be_removed)] # select nodes to be removed from path
                for node in nodes:
                    path1 = find_radial_path_2(graph2, node) # find radial path that starts with current node
                    for n in path1:
                        # compute distance between nodes that are upstream and downstream from current node:
                        n_successor = channel_successor(graph2, n)
                        predecessors = graph2.predecessors(n)
                        for predecessor in predecessors:
                            if graph2[predecessor][n]['edge_type'] == 'channel':
                                n_predecessor = predecessor
                        x1 = graph2.nodes[n_successor]['x']
                        y1 = graph2.nodes[n_successor]['y']
                        x2 = graph2.nodes[n_predecessor]['x']
                        y2 = graph2.nodes[n_predecessor]['y']
                        cl_dist = ((x2-x1)**2 + (y2-y1)**2)**0.5
                        # only remove node if distance between neighboring nodes along centerline is not too large:
                        if cl_dist < max_dist:
                            graph2.remove_node(n)
                            if n in graph2.graph['start_nodes']:
                                graph2.graph['start_nodes'].remove(n)
                        else:
                            break
                # reconnect nodes along every centerline that has been affected by node removal:
                for cln in range(cl_number, graph1.graph['number_of_centerlines']):
                    reconnect_nodes_along_centerline(graph1, graph2, cln)
    return graph2


def add_curvature_to_line_graph(graph, smoothing_factor):
    """
    Add curvature attribute to the nodes of a line graph.

    Parameters
    ----------
    graph : directed graph
        Graph of center- or banklines.
    smoothing_factor : int
        Smoothing factor in the Savitzky-Golay filtering that is applied to the curvature series.
    """

    n_centerlines = graph.graph['number_of_centerlines']
    curvs = []
    for cline in range(0, n_centerlines):
        path = find_longitudinal_path(graph, cline)
        curv = compute_curvature(graph.graph['x'][path], graph.graph['y'][path])
        curv = savgol_filter(curv, smoothing_factor, 2)
        count = 0
        for node in path:
            graph.nodes[node]['curv'] = curv[count]
            count += 1
    for node in graph.nodes:
        if 'curv' not in graph.nodes[node].keys():
            graph.nodes[node]['curv'] = np.nan


def add_timesteps_to_line_graph(graph, timesteps):
    """
    Add timestep attribute to the nodes of a line graph, representing the amount of time between successive longitudinal paths.
    These timesteps could come from Landsat timestamps

    Parameters
    ----------
    graph : directed graph
        Graph of center- or banklines.
    timesteps: list (Optional, Default = None)
        List containing floating point numbers representing the amount of time (years) between successive longitudinal paths.
    """

    n_centerlines = graph.graph['number_of_centerlines']
    for cline in range(0, n_centerlines-1):
        path = find_longitudinal_path(graph, cline)
        for node in path:
            graph.nodes[node]['timestep'] = timesteps[cline]
            
    for node in graph.nodes:
        if 'timestep' not in graph.nodes[node].keys():
            graph.nodes[node]['timestep'] = np.nan


def find_next_node(graph, start_node):
    """
    Find the next node along radial path.

    Parameters
    ----------
    graph : directed graph
        Graph with radial paths defined.

    start_node : int
        Node that is the starting point of radial path.

    Returns
    -------
    nodes : list 
        List of nodes along radial path
    """

    nodes = [start_node]
    while len(list(graph.successors(start_node))) > 0:
        next_node = list(graph.successors(start_node))[0]
        nodes.append(next_node)
        start_node = next_node
    return nodes


def add_sparse_cutoff_nodes(graph, min_dist):
    """
    Add a sparse_cutoff_nodes attribute to bar graph.

    Parameters
    ----------
    graph : directed graph
        Bar graph; graph containing bar objects
    min_dist:
        Minimum allowable distance between nodes (meters)
    """
    cutoff_node_ages = []
    for node in graph.graph['cutoff_nodes']:
        cutoff_node_ages.append(graph.nodes[node]['age'])
    cutoff_ages = np.unique(cutoff_node_ages)
    sparse_cutoff_nodes = []
    for cf_age in range(len(cutoff_ages)):
        path = find_longitudinal_path(graph, cutoff_ages[cf_age])
        ordered_cutoff_nodes = []
        for node in path:
            if node in graph.graph['cutoff_nodes']:
                ordered_cutoff_nodes.append(node)
        path = ordered_cutoff_nodes
        ds = [] # along-path distance
        for i in range(len(path)-1):
            ds.append(((graph.graph['x'][path[i+1]] - graph.graph['x'][path[i]])**2 + 
                       (graph.graph['y'][path[i+1]] - graph.graph['y'][path[i]])**2)**0.5)
        # get rid of the start nodes that are too close to each other:    
        sparse_inds = [0]
        running_sum = 0
        for i in range(len(ds)):
            running_sum += ds[i]
            if running_sum > min_dist:
                sparse_inds.append(i)
                running_sum = 0
        for node in np.array(path)[sparse_inds]:
            sparse_cutoff_nodes.append(node)
    graph.graph['sparse_cutoff_nodes'] = sparse_cutoff_nodes


def find_radial_path_2(graph, node):
    """
    Make a list of the indices of graph nodes that describe a radial path starting from 'node'
    (same as 'find_radial_path', but without the node ages)

    Parameters
    ----------
    graph : directed graph
        Graph of centerline or bankline
    node : int
        Number corresponding to the node from which to start the radial path.

    Returns
    -------
    path : list
        List of nodes along the radial path
    """
    path, path_ages = find_radial_path(graph, node)
    return path


def add_edge_directions_to_bank_graph(graph):
    """
    Add directionality to graph nodes as a 'direction' attribute.

    Parameters
    ----------
    graph : directed graph
        Centerline or bankline graph
    
    Returns
    -------
    graph : directed graph
        Centerline or bankline graph with 'direction' attribute added.
    """

    for node in trange(graph.graph['number_of_centerlines']):
        path = find_longitudinal_path(graph, node)
        for i in range(len(path) - 1):
            node_1 = path[i]
            node_2 = path[i+1]
            node_4 = radial_successor(graph, node_1)
            node_3 = radial_successor(graph, node_2)
            x1 = graph.nodes[node_1]['x']
            y1 = graph.nodes[node_1]['y']
            x2 = graph.nodes[node_2]['x']
            y2 = graph.nodes[node_2]['y']
            node_1_coords = np.array([x1, y1])
            node_2_coords = np.array([x2, y2])
            if node_3:
                x3 = graph.nodes[node_3]['x']
                y3 = graph.nodes[node_3]['y']
                node_3_coords = np.array([x3, y3])
                dist_23 = np.linalg.norm(node_2_coords - node_3_coords)
                direction_23 = directionOfPoint(x1, y1, x2, y2, x3, y3)
                graph[node_2][node_3]['direction'] = direction_23
            if node_4:
                x4 = graph.nodes[node_4]['x']
                y4 = graph.nodes[node_4]['y']
                node_4_coords = np.array([x4, y4])
                dist_14 = np.linalg.norm(node_1_coords - node_4_coords)
                direction_14 = directionOfPoint(x1, y1, x2, y2, x4, y4)
                graph[node_1][node_4]['direction'] = direction_14
    return graph


def find_cutoff_ages(graph):
    """
    Find the timestep corresponding to cutoff event(s).

    Parameters
    ----------
    graph : directed graph
            Centerline graph containing cutoffs.
    
    Returns
    -------
    cutoff_ages : 1D array
                  Array elements are timestep during which a cutoff occurred.
    """

    cutoff_node_ages = []
    for node in graph.graph['cutoff_nodes']:
        try:
            cutoff_node_ages.append(graph.nodes[node]['age'])
        except:
            pass
    cutoff_ages = np.unique(cutoff_node_ages)
    return cutoff_ages


