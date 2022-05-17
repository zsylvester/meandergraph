import meanderpy as mp
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import distance
from librosa.sequence import dtw
from tqdm import trange, tqdm
import networkx as nx
from descartes import PolygonPatch
from shapely.geometry import Polygon, MultiPolygon, Point, MultiLineString, LineString, shape, JOIN_STYLE
from shapely.geometry.polygon import LinearRing
from shapely.ops import snap, unary_union
import random
from copy import deepcopy

def find_next_index(x1, y1, x2, y2, p, q, ind1):
    """ find index 'ind2' of the next point on the next curve (which is defined by (x2, y2)), 
    # if the current index is 'ind1' on curve (x1, y1)
    parameters:
    x1 - x-coordinates of first curve
    y1 - y-coordinates of first curve
    x2 - x-coordinates of second curve
    y2 - y-coordinates of second curve
    p - correlation indices for first curve
    q - correlation indices for second curve
    ind1 - index of point of interest in first curve
    returns:
    ind2 - index of correlated point in second curve"""
    p_index = np.where(p == ind1)[0] # find the location where 'p' equals 'ind1'
    p_index = int(np.median(p_index)) # have to choose only one if there are more than one
    ind2 = q[p_index] # find the equivalent index in 'q'
    return ind2

def correlate_curves(x1,x2,y1,y2):
    # use dynamic time warping to correlate two 2D curves
    X = np.vstack((x1,y1))
    Y = np.vstack((x2,y2))
    sm = distance.cdist(X.T, Y.T) # similarity matrix
    D, wp = dtw(C=sm) # dynamic time warping
    p = wp[:,0] # correlation indices for first curve
    q = wp[:,1] # correlation indices for second curve
    return p, q

def correlate_set_of_curves(X, Y):
    """correlate a set of curves defined by x and y coordinates stored as two lists X and Y
    parameters:
    X - list of x coordinate arrays
    Y - list of y coordinate arrays
    returns:
    P - lists of indices of correlated successive pairs of curves (for first curve)
    Q - lists of indices of correlated successive pairs of curves (for second curve)"""
    P = []
    Q = []
    for i in trange(len(X) - 1):
        p, q = correlate_curves(X[i], X[i+1], Y[i], Y[i+1])
        P.append(p)
        Q.append(q)
    return(P, Q)

def find_indices(ind1, X, Y, P, Q):
    # function for tracing one index through a series of centerlines (stored as lists of coordinates X and Y)
    indices = []
    x = []
    y = []
    indices.append(ind1)
    x.append(X[0][ind1])
    y.append(Y[0][ind1])
    n_centerlines = len(X)
    for i in range(n_centerlines-1):
        ind2 = find_next_index(X[i], Y[i], X[i+1], Y[i+1], P[i], Q[i], ind1)
        indices.append(ind2)
        x.append(X[i+1][ind2])
        y.append(Y[i+1][ind2])
        ind1 = ind2
    x = np.array(x)
    y = np.array(y)
    return indices, x, y

def find_radial_path(graph, node):
    # collect the indices of graph nodes that describe a radial path starting from 'node'
    path = []
    path_ages = []
    path.append(node)
    path_ages.append(graph.nodes[node]['age'])
    edge_types = []
    for successor_node in graph.successors(node):
        edge_types.append(graph[node][successor_node]['edge_type'])
    while 'radial' in edge_types:
        for successor_node in graph.successors(node):
            if graph[node][successor_node]['edge_type'] == 'radial':
                next_node = successor_node
        path.append(next_node)
        path_ages.append(graph.nodes[next_node]['age'])
        node = next_node
        edge_types = []
        for successor_node in graph.successors(node):
            edge_types.append(graph[node][successor_node]['edge_type'])
    return path, path_ages

def find_radial_path_backward(graph, node):
    # collect the indices of graph nodes that describe a radial path starting from 'node'
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
    # collect the indices of graph nodes that describe a longitudinal path starting from 'node'
    path = []
    path.append(node)
    edge_types = []
    for successor_node in graph.successors(node):
        edge_types.append(graph[node][successor_node]['edge_type'])
    while 'channel' in edge_types:
        for successor_node in graph.successors(node):
            if graph[node][successor_node]['edge_type'] == 'channel':
                next_node = successor_node
        path.append(next_node)
        node = next_node
        edge_types = []
        for successor_node in graph.successors(node):
            edge_types.append(graph[node][successor_node]['edge_type'])
    return path

def create_list_of_start_nodes(graph):
    # find all the nodes in a graph that are starting points for radial paths
    start_nodes = []
    for node in graph.nodes:
        parents = graph.predecessors(node)
        edge_types = []
        for parent in parents:
            edge_types.append(graph[parent][node]['edge_type'])
        if 'radial' not in edge_types:
            start_nodes.append(node)
    return start_nodes

def delete_elements_from_list(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)
    return list_object

def create_graph_from_channel_lines(X, Y, P, Q, n_points, max_dist, remove_cutoff_edges = False):
    """inputs:
    n_points = every 'n_points'th point on the first centerline is used to start a radial trajectory
    max_dist = parameter that is used to eliminate edges that correspond to cutoffs
    """
    graph = nx.DiGraph(number_of_centerlines = len(X)) # directed graph to store nodes and edges
    cl_indices = [] # list of lists to store *centerline* indices that will be part of the graph
    n_centerlines = len(X)
    for i in range(n_centerlines): # initialize 'cl_indices'
        cl_indices.append([]) 
    # add radial nodes and edges:
    for ind1 in range(0, len(X[0]), n_points): 
        indices, x, y = find_indices(ind1, X, Y, P, Q)
        new_node_inds = np.arange(len(graph), len(graph) + len(indices))
        for i in range(len(indices)):
            cl_indices[i].append(indices[i])
            graph.add_node(new_node_inds[i], x = x[i], y = y[i], age = i, curv = 0)
        for i in range(len(indices)-1):
            graph.add_edge(new_node_inds[i], new_node_inds[i+1], edge_type = 'radial', age = i)
    for cl_number in trange(n_centerlines - 1): 
        # add 'intermediate' trajectories:
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
    for cl_number in range(n_centerlines):
        cl_nodes = []
        for i in cl_indices[cl_number]:
            node_ind = np.where((x == X[cl_number][i]) & (y == Y[cl_number][i]))[0][0]
            cl_nodes.append(node_ind)
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
    cl_nodes = []
    for cl_number in trange(len(X)): # collect all nodes that are connected along the centerlines
        path = find_longitudinal_path(graph, cl_number)
        cl_nodes += path
    for node in set(cl_nodes) ^ set(graph.nodes): # if a node is not in 'cl_nodes', remove it from the graph
        graph.remove_node(node)
        if node in graph.graph['start_nodes']: # remove the node from 'start_nodes' as well
            graph.graph['start_nodes'].remove(node)
    return graph

def reconnect_nodes_along_centerline(graph1, graph2, cl_number):
    # function for reconnecting nodes along a centerline in graph2, based on nodes along the same centerline in graph1
    path = find_longitudinal_path(graph1, cl_number)
    cl_nodes = []
    for node in path:
        if node in graph2:
            cl_nodes.append(node)
    for i in range(len(cl_nodes) - 1):
        if (cl_nodes[i], cl_nodes[i+1]) not in graph2.edges:
            graph2.add_edge(cl_nodes[i], cl_nodes[i+1], edge_type = 'channel')

def compute_derivatives(x, y):
    """function for computing first derivatives of a curve (centerline)
    x,y are cartesian coodinates of the curve
    outputs:
    dx - first derivative of x coordinate
    dy - first derivative of y coordinate
    ds - distances between consecutive points along the curve
    s - cumulative distance along the curve"""
    dx = np.diff(x) # first derivatives
    dy = np.diff(y)   
    ds = np.sqrt(dx**2+dy**2)
    s = np.hstack((0,np.cumsum(ds)))
    return dx, dy, ds, s
            
def remove_high_density_nodes(graph1, min_dist, max_dist):
    """function for removing nodes and edges where radial lines are too dense (especially after cutoffs)
    example usage:
    graph = mg.remove_high_density_nodes(graph, min_dist = 10, max_dist = 30)"""
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
                        successors = graph2.successors(n)
                        for successor in successors:
                            if graph2[n][successor]['edge_type'] == 'channel':
                                n_successor = successor
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
                            if node in graph2.graph['start_nodes']:
                                graph2.graph['start_nodes'].remove(n)
                        else:
                            break
                # reconnect nodes along every centerline that has been affected by node removal:
                for cln in range(cl_number, graph1.graph['number_of_centerlines']):
                    reconnect_nodes_along_centerline(graph1, graph2, cln)
    return graph2

def plot_graph(graph, ax):
    """function for plotting channel line graphs (does not work with polygon graphs)"""
    cmap = plt.get_cmap("tab10")
    for node in np.arange(graph.graph['number_of_centerlines']):
        path = find_longitudinal_path(graph, node)
        ax.plot(graph.graph['x'][path], graph.graph['y'][path], color = cmap(0), linewidth = 0.5)  
    for node in graph.graph['start_nodes']:
        path, path_ages = find_radial_path(graph, node)
        ax.plot(graph.graph['x'][path], graph.graph['y'][path], color = cmap(2), linewidth = 0.5)
    plt.axis('equal')
        
def create_polygon_graph(graph):
    # create graph of polygons from centerline / bankline graph
    poly_graph = nx.DiGraph()
    cl_start_nodes = []
    age = 0
    for node in trange(graph.graph['number_of_centerlines'] - 1):
        path = find_longitudinal_path(graph, node)
        for i in range(len(path) - 1):
            node_1 = path[i]
            node_2 = path[i+1]
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
            if (not node_3) & (i < len(path) - 2):
                node_2 = path[i+2]
                node_2_children = list(graph.successors(node_2))
                for n in node_2_children:
                    if graph[node_2][n]['edge_type'] == 'radial':
                        node_3 = n
            if node_3 and node_4: # only add a new polygon if there is another centerline
                coords = []
                x1 = graph.nodes[node_1]['x']
                x2 = graph.nodes[node_2]['x']
                y1 = graph.nodes[node_1]['y']
                y2 = graph.nodes[node_2]['y']
                try:
                    outer_poly_boundary = nx.shortest_path(graph, source=node_4, target=node_3)
                except: # if there is no path between node 4 and node 3
                    outer_poly_boundary = []
                # sometimes 'node_3' and 'node_4' are the same node, and this is needed:
                if (graph.nodes[node_3]['x'] == graph.nodes[node_4]['x']) & (graph.nodes[node_3]['y'] == graph.nodes[node_4]['y']):
                    x3 = graph.nodes[node_3]['x']
                    x4 = graph.nodes[node_4]['x']
                    y3 = graph.nodes[node_3]['y']
                    y4 = graph.nodes[node_4]['y']
                    coords = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)]
                if len(outer_poly_boundary) == 2: # 2 nodes on the outer boundary
                    x3 = graph.nodes[node_3]['x']
                    x4 = graph.nodes[node_4]['x']
                    y3 = graph.nodes[node_3]['y']
                    y4 = graph.nodes[node_4]['y']
                    coords = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)]
                if len(outer_poly_boundary) == 3: # 3 nodes on the outer boundary
                    x3 = graph.nodes[outer_poly_boundary[2]]['x']
                    x4 = graph.nodes[outer_poly_boundary[1]]['x']
                    x5 = graph.nodes[outer_poly_boundary[0]]['x']
                    y3 = graph.nodes[outer_poly_boundary[2]]['y']
                    y4 = graph.nodes[outer_poly_boundary[1]]['y']
                    y5 = graph.nodes[outer_poly_boundary[0]]['y']
                    coords = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x1, y1)]
                # this does not work and I am not sure why at this point:
                # if len(outer_poly_boundary) == 4: # 4 nodes on the outer boundary
                #     x3 = graph.nodes[outer_poly_boundary[3]]['x']
                #     x4 = graph.nodes[outer_poly_boundary[2]]['x']
                #     x5 = graph.nodes[outer_poly_boundary[1]]['x']
                #     x6 = graph.nodes[outer_poly_boundary[0]]['x']
                #     y3 = graph.nodes[outer_poly_boundary[3]]['y']
                #     y4 = graph.nodes[outer_poly_boundary[2]]['y']
                #     y5 = graph.nodes[outer_poly_boundary[1]]['y']
                #     y6 = graph.nodes[outer_poly_boundary[0]]['x']
                #     coords = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6), (x1, y1)]
                if len(coords) > 0:
                    poly = Polygon(LinearRing(coords))
                    if poly.is_valid: # add node only if polygon is valid
                        poly_graph.add_node(path[i], poly = poly, age = age, x = x1, y = y1)
                    else:
                        poly = poly.buffer(0) # fix the invalid polygon
                        poly_graph.add_node(path[i], poly = poly, age = age, x = x1, y = y1)
                    if i == 0:
                        if poly.is_valid:
                            cl_start_nodes.append(path[i])
                        else:
                            poly = poly.buffer(0) # fix the invalid polygon
                            poly_graph.add_node(path[i], poly = poly, age = age, x = x1, y = y1)
                            cl_start_nodes.append(path[i])
            else: 
                if i == 0: # something is needed at the beginning of the centerline even when there is no 'node_3' or 'node_4', so we just make up a polygon
                    x3 = x2 
                    y3 = y2 + 1.0
                    x4 = x1 
                    y4 = y1 + 1.0
                    coords = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)]
                    poly = Polygon(LinearRing(coords))
                    if not poly.is_valid: # fix poly
                        poly = poly.buffer(0) # fix the invalid polygon
                    poly_graph.add_node(path[i], poly = poly, age = age, x = x1, y = y1)
                    cl_start_nodes.append(path[i])

        for i in range(len(path) - 2): # add graph edges
            if (path[i] in poly_graph) & (path[i+1] in poly_graph):
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

def plot_bars2(graph, cutoff_area, ax, W):
    # function for creating polygons for 'scroll' bars and plotting them
    n_centerlines = graph.graph['number_of_centerlines']
    X = []
    Y = []
    for node in np.arange(n_centerlines):
        path = find_longitudinal_path(graph, node)
        X.append(graph.graph['x'][path])
        Y.append(graph.graph['y'][path])
    ts = len(X)
    bars = [] # these are 'scroll' bars - shapely MultiPolygon objects that correspond to one time step
    chs = [] # list of channels - shapely Polygon objects
    jumps = [] # gaps between channel polygons that are not cutoffs
    all_chs = [] # list of merged channels (to be used for erosion)
    cutoffs = []
    cmap = mpl.cm.get_cmap('viridis')
    # creating list of channels, jumps, and cutoffs
    for i in trange(ts-1):
        ch1 = create_channel_polygon(X[i], Y[i], W)
        ch2 = create_channel_polygon(X[i+1], Y[i+1], W)
        ch1, bar, erosion, jump, cutoff = one_step_difference_no_plot(ch1, ch2, cutoff_area)
        chs.append(ch1)
        jumps.append(jump)
        for cf in cutoff:
            if type(cf) == MultiPolygon:
                cutoff.remove(cf)
        cutoffs.append(cutoff)
    chs.append(ch2) # append last channel
    # creating list of merged channels
    for i in trange(ts): # create list of merged channels
        if i == 0: 
            all_ch = chs[ts-1]
        else:
            all_ch = all_ch.union(chs[ts-i])
        all_chs.append(all_ch)
    # creating scroll bars and plotting
    for i in trange(ts): # create scroll bars
        bar = chs[i].difference(all_chs[ts-i-1]) # scroll bar defined by difference
        bars.append(bar)
        color = cmap(i/float(ts))
        if type(bar) != Polygon:
            for b in bar:
                if MultiPolygon(cutoffs[i]).is_valid: # sometimes this is invalid
                    if not b.intersects(MultiPolygon(cutoffs[i])):
                        ax.add_patch(PolygonPatch(b,facecolor=color,edgecolor='k'))
                else:
                    ax.add_patch(PolygonPatch(b,facecolor=color,edgecolor='k'))
            else:
                ax.add_patch(PolygonPatch(bar, facecolor=color, edgecolor='k'))
    return bars, chs, all_chs, jumps, cutoffs

def create_channel_polygon(x, y, W):
    # function for reading channel bank shapefiles and creating a polygon
    xm, ym = mp.get_channel_banks(x, y, W)
    coords = []
    for i in range(len(xm)):
        coords.append((xm[i],ym[i]))
    ch = Polygon(LinearRing(coords))
    if not ch.is_valid:
        ch = ch.buffer(0)
    return ch

def one_step_difference_no_plot(ch1, ch2, cutoff_area):
    both_channels = ch1.union(ch2) # union of the two channels
    if type(both_channels) == MultiPolygon:
        poly = both_channels[0]
        for j in range(len(both_channels)):
            if both_channels[j].area > poly.area:
                poly = both_channels[j]
        both_channels = poly
    outline = Polygon(LinearRing(list(both_channels.exterior.coords))) # outline of the union
    jump = outline.difference(both_channels) # gaps between the channels
    bar = ch1.difference(ch2) # the (point) bars are the difference between ch1 and ch2
    bar = bar.union(jump) # add gaps to bars
    erosion = ch2.difference(ch1) # erosion is the difference between ch2 and ch1
    bar_no_cutoff = list(bar.geoms) # create list of bars (cutoffs will be removed later)
    erosion_no_cutoff = list(erosion.geoms) # create list of eroded areas (cutoffs will be removed later)
    if type(jump)==MultiPolygon: # create list of gap polygons (if there is more than one gap)
        jump_no_cutoff = list(jump.geoms)
    else:
        jump_no_cutoff = jump
    cutoffs = []
    for b in bar.geoms:
        if b.area>cutoff_area:
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
                        if (type(j.intersection(b))==Polygon) & (j.area>0.3*cutoff_area):
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

def polygon_width_and_length(graph, node):
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
        if (not node_3) & (len(path) > 2):
            node_2 = path[2]
            node_2_children = list(graph.successors(node_2))
            for n in node_2_children:
                if graph[node_2][n]['edge_type'] == 'radial':
                    node_3 = n
        coords = []
        x1 = graph.nodes[node_1]['x']
        x2 = graph.nodes[node_2]['x']
        y1 = graph.nodes[node_1]['y']
        y2 = graph.nodes[node_2]['y']
        width_1 = compute_distance(x1, x2, y1, y2)
        try:
            outer_poly_boundary = nx.shortest_path(graph, source=node_4, target=node_3)
        except: # if there is no path between node 4 and node 3
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

def add_curvature_to_line_graph(graph, smoothing_factor):
    n_centerlines = graph.graph['number_of_centerlines']
    curvs = []
    for cline in range(0, n_centerlines):
        path = find_longitudinal_path(graph, cline)
        curv = mp.compute_curvature(graph.graph['x'][path], graph.graph['y'][path])
        curv = savgol_filter(curv, smoothing_factor, 2)
        count = 0
        for node in path:
            graph.nodes[node]['curv'] = curv[count]
            count += 1
    for node in graph.nodes:
        if 'curv' not in graph.nodes[node].keys():
            graph.nodes[node]['curv'] = 0

def plot_migration_rate_map(wbar, graph1, graph2, vmin, vmax, dt, saved_ts, ax):
    if wbar.scrolls[-1].bank == 'left':
        graph = graph2
    if wbar.scrolls[-1].bank == 'right':
        graph = graph1
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    m = mpl.cm.ScalarMappable(norm=norm, cmap='viridis')
    time_step = (dt * saved_ts)/(365*24*60*60)
    for node in tqdm(wbar.bar_graph.nodes):
        width, length = polygon_width_and_length(graph, node)
        ax.add_patch(PolygonPatch(wbar.bar_graph.nodes[node]['poly'], facecolor = m.to_rgba(length/time_step), 
                                  edgecolor='k', linewidth=0.25))

def plot_curvature_map(wbar, graph1, graph2, vmin, vmax, W, ax):
    if wbar.scrolls[-1].bank == 'left':
        graph = graph2
    if wbar.scrolls[-1].bank == 'right':
        graph = graph1
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    m = mpl.cm.ScalarMappable(norm=norm, cmap='RdBu')
    for node in wbar.bar_graph.nodes:
            ax.add_patch(PolygonPatch(wbar.bar_graph.nodes[node]['poly'], 
                facecolor=m.to_rgba(W * graph.nodes[node]['curv']), edgecolor='k', linewidth=0.25))

def plot_age_map(wbar, vmin, vmax, W, ax):
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    m = mpl.cm.ScalarMappable(norm=norm, cmap='viridis')
    for node in wbar.bar_graph.nodes:
            ax.add_patch(PolygonPatch(wbar.bar_graph.nodes[node]['poly'], 
                facecolor=m.to_rgba(wbar.bar_graph.nodes[node]['age']), edgecolor='k', linewidth=0.25))

    
def compute_distance(x1, x2, y1, y2):
    dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    return dist

def find_next_node(graph, start_node):
    nodes = [start_node]
    while len(list(graph.successors(start_node))) > 0:
        next_node = list(graph.successors(start_node))[0]
        nodes.append(next_node)
        start_node = next_node
    return nodes

def merge_polygons(graph, nodes, sparse_inds, polys):
    inds = np.arange(len(nodes))
    poly = graph.nodes[nodes[0]]['poly']
    for i in range(len(sparse_inds)-1):
        for j in inds[sparse_inds[i]+1 : sparse_inds[i+1]+1]:
            if (j == sparse_inds[i]+1) & (i!=0):
                poly = graph.nodes[nodes[j]]['poly']
            else:
                poly = poly.union(graph.nodes[nodes[j]]['poly'])
        polys.append(poly)
    return polys

def find_sparse_inds(graph, nodes, min_area):
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

def add_sparse_cutoff_nodes(graph, min_dist):
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

def plot_bar_lines(wbars, bar_ind, bank_graph, ax):
    cmap = plt.get_cmap("tab10")
    source_nodes = [] # source nodes for longitudinal lines
    for node in wbars[bar_ind].bar_graph.nodes:
        if wbars[bar_ind].bar_graph.in_degree(node) == 0:
            source_nodes.append(node)
    for node in source_nodes:
        path = find_longitudinal_path(wbars[bar_ind].bar_graph, node)
        x = bank_graph.graph['x'][path]
        y = bank_graph.graph['y'][path]
        if len(x) > 1:
            line = LineString(np.vstack((x,y)).T).intersection(wbars[bar_ind].polygon)
            if type(line) != MultiLineString:
                x1 = line.xy[0]
                y1 = line.xy[1]
                ax.plot(x1, y1, color=cmap(0), linewidth=0.5)
            else:
                for l in line:
                    x1 = l.xy[0]
                    y1 = l.xy[1]
                    ax.plot(x1, y1, color=cmap(0), linewidth=0.5)
    temp_radial_graph = nx.DiGraph() # temporary radial graph for radial lines
    for node in wbars[bar_ind].bar_graph.nodes:
        temp_radial_graph.add_node(node, x=bank_graph.graph['x'][node], y=bank_graph.graph['y'][node])
    for s in wbars[bar_ind].bar_graph.nodes:
        for e in bank_graph.successors(s):
            if bank_graph[s][e]['edge_type'] == 'radial':
                if e in temp_radial_graph.nodes:
                    temp_radial_graph.add_edge(s, e)
    source_nodes = [] # source nodes for radial lines
    for node in temp_radial_graph.nodes:
        if temp_radial_graph.in_degree(node) == 0:
            source_nodes.append(node)
    path = find_longitudinal_path(bank_graph, bank_graph.graph['start_nodes'][0])
    for node in path:
        radial_path, dummy = find_radial_path(bank_graph, node)
        for common_node in set(radial_path) & set(source_nodes):
            path1, dummy = find_radial_path(bank_graph, common_node)
            x = bank_graph.graph['x'][path1]
            y = bank_graph.graph['y'][path1]
            if len(x) > 1:
                line = LineString(np.vstack((x,y)).T).intersection(wbars[bar_ind].polygon)
                if type(line) != MultiLineString:
                    x1 = line.xy[0]
                    y1 = line.xy[1]
                    ax.plot(x1, y1, color=cmap(1), linewidth=0.5)
                else:
                    for l in line:
                        x1 = l.xy[0]
                        y1 = l.xy[1]
                        ax.plot(x1, y1, color=cmap(1), linewidth=0.5)
    ax.add_patch(PolygonPatch(wbars[bar_ind].polygon, facecolor='none', edgecolor='k', linewidth = 2, zorder = 10000))

def create_scrolls_and_find_connected_scrolls(graph, W, cutoff_area, ax):
    # create scrolls
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    bars, chs, all_chs, jumps, cutoffs = plot_bars2(graph, cutoff_area, ax1, W)
    plt.close(fig)

    # remove cutoffs from list of scrolls of same age:
    new_bars = []
    for bar in bars:
        if type(bar) == MultiPolygon:
            bar = MultiPolygon([P for P in bar if P.area < cutoff_area])
        new_bars.append(bar) 
    bars = new_bars

    n_scrolls = [] # number of scrolls in each 'bar'
    for bar in bars:
        if type(bar) == MultiPolygon:
            n_scrolls.append(len(bar))
        else:
            n_scrolls.append(1)

    scrolls = []
    scroll_ages = []
    count = 0
    single_polygon_count = 0
    for bar in bars:
        if type(bar) == MultiPolygon:
            for scroll in bar:
                scrolls.append(scroll)
                scroll_ages.append(count)
        else:
            scrolls.append(bar)
            scroll_ages.append(count)
            single_polygon_count += 1
        count += 1

    connections = []
    for i in trange(1, len(bars)):
        for j in range(n_scrolls[i]):
            for k in range(n_scrolls[i-1]):
                if (type(bars[i-1]) == MultiPolygon) & (type(bars[i]) == MultiPolygon):
                    if bars[i-1][k].buffer(1.0).overlaps(bars[i][j]):
                        connections.append((sum(n_scrolls[:i]) + j, sum(n_scrolls[:i-1]) + k))
                if (type(bars[i-1]) == Polygon) & (type(bars[i]) == MultiPolygon):
                    if bars[i-1].buffer(1.0).overlaps(bars[i][j]):
                        connections.append((sum(n_scrolls[:i]) + j, sum(n_scrolls[:i-1]) + 1))
                if (type(bars[i-1]) == MultiPolygon) & (type(bars[i]) == Polygon):
                    if bars[i-1][k].buffer(1.0).overlaps(bars[i]):
                        connections.append((sum(n_scrolls[:i]) + 1, sum(n_scrolls[:i-1]) + k))

    all_bars_graph = nx.Graph()
    for i in range(len(connections)):
        all_bars_graph.add_edge(connections[i][0], connections[i][1])

    for component in nx.connected_components(all_bars_graph):
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b, 0.5)
        for i in component:
            if scrolls[i].area > 1.0:
                ax.add_patch(PolygonPatch(scrolls[i], facecolor=color, edgecolor='k'))
    return scrolls, scroll_ages, all_bars_graph, cutoffs

def create_polygon_graphs_and_bar_graphs(graph1, graph2, ts, all_bars_graph, scrolls, scroll_ages, cutoffs, dt, X, Y, W, saved_ts, ax):
    # create polygon graphs for the banks:
    poly_graph_1 = create_polygon_graph(graph1)
    poly_graph_2 = create_polygon_graph(graph2)
    # create list of Bar objects:
    wbars = []
    count = 0
    max_ages = []
    for component in nx.connected_components(all_bars_graph):
        wbar = Bar(count, [])
        ages = []
        for i in component:
            # if current scroll intersects the right bank of the same age:
            if scrolls[i].buffer(1.0).intersects(LineString(np.vstack((X1[scroll_ages[i]], Y1[scroll_ages[i]])).T)):
                bank = 'right'
            else:
                bank = 'left'
            wbar.scrolls.append(Scroll(i, scroll_ages[i], bank, scrolls[i], wbar, [])) 
            ages.append(scroll_ages[i])
        wbar.create_polygon() # create bar polygon
        wbars.append(wbar)
        max_ages.append(max(ages))
    # add polygon graphs to bars:
    for i in range(len(wbars)):
        if wbars[i].scrolls[-1].bank == 'right':
            wbars[i].add_polygon_graphs(poly_graph_1)
        if wbars[i].scrolls[-1].bank == 'left':
            wbars[i].add_polygon_graphs(poly_graph_2)   
    return wbars, poly_graph_1, poly_graph_2

def plot_bar_graphs(graph1, graph2, wbars, ts, cutoffs, dt, X, Y, W, saved_ts, vmin, vmax, plot_type, ax):
    # collect cutoff indices
    cutoff_inds = []
    count = 0
    for cf in cutoffs:
        if len(cf) > 0:
            cutoff_inds.append(count)
        count += 1
    # plotting cutoffs:     
    for wbar in wbars:
        ages = []
        for scroll in wbar.scrolls:
            ages.append(scroll.age)
        for i in cutoff_inds: # cutoffs need to be plotted at the right time
            if max(ages) + 1 == i:
                ax.add_patch(PolygonPatch(cutoffs[i][0], facecolor='lightblue', edgecolor='k'))
    # create polygon for most recent channel and plot it:
    xm, ym = mp.get_channel_banks(X[ts-1], Y[ts-1], W)
    coords = []
    for i in range(len(xm)):
        coords.append((xm[i],ym[i]))
    coords.append((xm[0], ym[0]))
    ch = Polygon(LinearRing(coords))
    ax.add_patch(PolygonPatch(ch, facecolor='lightblue', edgecolor='k'))
    # add polygon graphs to bars and plot them:
    for i in range(len(wbars)):
        if plot_type == 'migration':
            plot_migration_rate_map(wbars[i], graph1, graph2, vmin, vmax, dt, saved_ts, ax)
        if plot_type == 'curvature':
            plot_curvature_map(wbars[i], graph1, graph2, vmin, vmax, W, ax)
        if plot_type == 'age':
            plot_age_map(wbars[i], vmin, vmax, W, ax)
    plt.axis('equal');

def create_simple_polygon_graph(bank_graph, X):
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
    for node in range(0, len(X)):
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
    for node in trange(len(X)):
        path = find_longitudinal_path(graph, node)
        path1 = [] 
        for n in path:
            if n in radial_nodes:
                path1.append(n)
        path = path1
        for i in range(len(path) - 1):
            node_1 = path[i]
            node_2 = path[i+1]
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
                except:
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

def plot_bars_by_bar_number(wbars, ax):
    for wbar in wbars:
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b, 0.5) # random color for each bar
        wbar.plot(ax, color)

# from: https://www.geeksforgeeks.org/direction-point-line-segment/
def directionOfPoint(xa, ya, xb, yb, xp, yp):
    # Subtracting co-ordinates of 
    # point A from B and P, to 
    # make A as origin
    xb -= xa
    yb -= ya
    xp -= xa
    yp -= ya
    # Determining cross Product
    cross_product = xb * yp - yb * xp
    # Return RIGHT if cross product is positive
    if (cross_product > 0):
        return 1  
    # Return LEFT if cross product is negative
    if (cross_product < 0):
        return -1
    # Return ZERO if cross product is zero
    return 0

def find_radial_path_2(graph, node):
    # collect the indices of graph nodes that describe a radial path starting from 'node'
    path = []
    path.append(node)
    edge_types = []
    for successor_node in graph.successors(node):
        edge_types.append(graph[node][successor_node]['edge_type'])
    while 'radial' in edge_types:
        for successor_node in graph.successors(node):
            if graph[node][successor_node]['edge_type'] == 'radial':
                next_node = successor_node
        path.append(next_node)
        node = next_node
        edge_types = []
        for successor_node in graph.successors(node):
            edge_types.append(graph[node][successor_node]['edge_type'])
    return path

def plot_simple_polygon_graph(poly_graph, ax, bank_type):
    cmap = plt.get_cmap("tab10")
    path = find_longitudinal_path(poly_graph, 0)
    for i in trange(len(path)):
        radial_path = find_radial_path_2(poly_graph, path[i])
        count = 0
        for node in radial_path:
            if 'poly' in poly_graph.nodes[node].keys():
                if poly_graph.nodes[node]['poly']:
                    if bank_type == 'left':
                        if poly_graph.nodes[node]['direction'] == -1:
                            ax.add_patch(PolygonPatch(poly_graph.nodes[node]['poly'], facecolor = cmap(1), edgecolor='k', linewidth = 0.3, alpha = 0.5, zorder = count))
                        if poly_graph.nodes[node]['direction'] == 1:
                            ax.add_patch(PolygonPatch(poly_graph.nodes[node]['poly'], facecolor = cmap(0), edgecolor='k', linewidth = 0.3, alpha = 0.5, zorder = count))
                    if bank_type == 'right':
                        if poly_graph.nodes[node]['direction'] == -1:
                            ax.add_patch(PolygonPatch(poly_graph.nodes[node]['poly'], facecolor = cmap(0), edgecolor='k', linewidth = 0.3, alpha = 0.5, zorder = count))
                        if poly_graph.nodes[node]['direction'] == 1:
                            ax.add_patch(PolygonPatch(poly_graph.nodes[node]['poly'], facecolor = cmap(1), edgecolor='k', linewidth = 0.3, alpha = 0.5, zorder = count))
            count += 1

class Bar:
    def __init__(self, number, scrolls):
        self.number = number
        self.scrolls = scrolls
    def plot(self, ax, color):
        ax.add_patch(PolygonPatch(self.polygon, facecolor='w', edgecolor='k', linewidth=2))
        for scroll in self.scrolls:
            ax.add_patch(PolygonPatch(scroll.polygon, facecolor=color, edgecolor='k', linewidth=0.5))
    def create_polygon(self):
        # create bar polygon from component scrolls
        whole_bar = self.scrolls[0].polygon
        for scroll in self.scrolls:
            # using 'union' can result in topological errors
            whole_bar = unary_union([whole_bar, scroll.polygon])
        whole_bar = whole_bar.buffer(0.1, 1, join_style=JOIN_STYLE.mitre).buffer(-0.1, 1, join_style=JOIN_STYLE.mitre)
        self.polygon = whole_bar
    def add_polygon_graphs(self, graph): 
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
                                poly = self.polygon.intersection(graph.nodes[node]['poly'])
                            else:
                                poly = self.polygon.intersection(graph.nodes[node]['poly'].buffer(0))
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
            for node1 in (path1):
                for node2 in path2:
                    if bar_graph.nodes[node1]['poly'].relate(bar_graph.nodes[node2]['poly']) == 'FF2F11212':
                        bar_radial_graph.add_edge(node1, node2, edge_type = 'radial')
                        bar_radial_graph.nodes[node1]['x'] = bar_graph.nodes[node1]['poly'].centroid.x
                        bar_radial_graph.nodes[node1]['y'] = bar_graph.nodes[node1]['poly'].centroid.y
                        bar_radial_graph.nodes[node2]['x'] = bar_graph.nodes[node2]['poly'].centroid.x
                        bar_radial_graph.nodes[node2]['y'] = bar_graph.nodes[node2]['poly'].centroid.y
        self.bar_graph = bar_graph
        self.bar_radial_graph = bar_radial_graph
    def plot_polygons(self, ax, plot_graphs):
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b, 0.5)
        for scroll in self.scrolls: # plot cropped polygons
            for small_polygon in scroll.small_polygons:
                ax.add_patch(PolygonPatch(small_polygon, facecolor=color, edgecolor='k', linewidth = 0.5))
        if plot_graphs:
            for (s, e) in tqdm(self.bar_graph.edges):
                ax.plot([self.bar_graph.nodes[s]['poly'].centroid.x, self.bar_graph.nodes[e]['poly'].centroid.x],
                        [self.bar_graph.nodes[s]['poly'].centroid.y, self.bar_graph.nodes[e]['poly'].centroid.y], 
                        'r', linewidth = 1)
            for (s, e) in tqdm(self.bar_radial_graph.edges):
                ax.plot([self.bar_radial_graph.nodes[s]['x'], self.bar_radial_graph.nodes[e]['x']],
                        [self.bar_radial_graph.nodes[s]['y'], self.bar_radial_graph.nodes[e]['y']], 
                        'g', linewidth = 1)
        ax.add_patch(PolygonPatch(self.polygon, facecolor='none', edgecolor='k', linewidth = 2))      
    def create_merged_polygons(self, ax, min_area):
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
            ax.add_patch(PolygonPatch(poly, facecolor='none', edgecolor='k', linewidth=0.5))
        ax.add_patch(PolygonPatch(self.polygon, facecolor='none', edgecolor='b', linewidth=2))
        self.merged_polygons = polys

class Scroll:
    def __init__(self, number, age, bank, polygon, bar, small_polygons):
        self.number = number
        self.age = age
        self.bank = bank # left or right bank
        self.polygon = polygon
        self.bar = bar
        self.small_polygons = small_polygons