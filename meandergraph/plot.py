"""
Plotting functions for line graphs, channel/bar polygons, and per-bar
attribute maps (migration rate, curvature, age).

'wbar'/'Bar' type hints below refer to bars.Bar; it isn't imported here
(as a real import, only in string form) to avoid a circular import, since
bars.py itself imports compute_bars_from_banks from this module.
"""
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from tqdm import trange
import random
from shapely.geometry import Polygon, MultiPolygon, MultiLineString, LineString

from .graph import find_longitudinal_path, find_radial_path, find_radial_path_2
from .polygons import create_channel_polygon_from_banks, create_channel_polygon_from_centerline, one_step_difference_no_plot
from .geometry import ensure_multipolygon, fix_geometry
from .correlation import find_indices

__all__ = [
    "plot_graph",
    "compute_bars_from_centerline", "plot_bars_from_centerline",
    "compute_bars_from_banks", "plot_bars_from_banks",
    "plot_migration_rate_map", "plot_curvature_map", "plot_age_map",
    "plot_bar_lines", "plot_bar_graphs", "plot_bars_by_bar_number",
    "plot_simple_polygon_graph", "plot_chosen_radial_paths", "fill_polygon",
]

def plot_graph(graph, ax, show_nodes = False, label_nodes = False):
    """
    Plot channel line graphs (does not work with polygon graphs)

    Parameters
    ----------
    graph : directed graph
        Graph to be plotted.
    ax : figure axes
    show_nodes : boolean (Optional)
                 Display the nodes as a point on the graph.
    label_nodes: boolean (Optional)
                 Display the node numbers as text on the graph.
    """

    cmap = plt.get_cmap("tab10")
    for node in np.arange(graph.graph['number_of_centerlines']):
        path = find_longitudinal_path(graph, node)
        ax.plot(graph.graph['x'][path], graph.graph['y'][path], '-', color = cmap(0), linewidth = 0.5)  
    for node in graph.graph['start_nodes']:
        path, path_ages = find_radial_path(graph, node)
        ax.plot(graph.graph['x'][path], graph.graph['y'][path], '-', color = cmap(1), linewidth = 0.5)
    if show_nodes == True or label_nodes == True:
        for node in np.arange(graph.graph['number_of_centerlines']):
            path = find_longitudinal_path(graph, node)
            for path_node in path:
                if show_nodes == True:
                    ax.plot(graph.graph['x'][path_node], graph.graph['y'][path_node], '.k', markersize = 1.0)
                if label_nodes == True:
                    ax.text(graph.graph['x'][path_node], graph.graph['y'][path_node], str(path_node))
    plt.axis('equal')


def compute_bars_from_centerline(graph, cutoff_area, W):
    """
    Compute channel/scroll-bar/cutoff polygons from centerline data (no
    plotting).

    Parameters
    ----------
    graph : directed graph
        Centerline graph.
    cutoff_area : float
        Maximum continuous area (created through channel bank movement in one timestep) that is still considered a bar and not a cutoff.
    W : float
        Channel width.

    Returns
    -------
    bars : list
        Shapely multipolygons representing 'scroll' bars that result from channel migration during one timestep.
    chs : list
        Shapely polygons that represent channels through time.
    all_chs : list
        Shapely polygons that represent merged channels through time.
    jumps : list
        Sometimes there is a gap between two consecutive channels and these gaps are collected into a list of polygons.
    cutoffs ; list
        Shapely polygons that represent cutoffs.
    """

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
    # creating list of channels, jumps, and cutoffs
    for i in trange(ts-1):
        ch1 = create_channel_polygon_from_centerline(X[i], Y[i], W)
        ch2 = create_channel_polygon_from_centerline(X[i+1], Y[i+1], W)
        ch1, bar, erosion, jump, cutoff = one_step_difference_no_plot(ch1, ch2, cutoff_area)
        ch1 = fix_geometry(ch1)
        ch2 = fix_geometry(ch2)
        bar = fix_geometry(bar)
        jump = fix_geometry(jump)
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
    # creating scroll bars
    for i in trange(ts): # create scroll bars
        bar = chs[i].difference(all_chs[ts-i-1]) # scroll bar defined by difference
        bars.append(bar)
    return bars, chs, all_chs, jumps, cutoffs

def plot_bars_from_centerline(graph, cutoff_area, ax, W):
    """
    Create polygons for 'scroll' bars from channel centerline data and plot them.

    Parameters
    ----------
    graph : directed graph
        Centerline graph.
    cutoff_area : float
        Maximum continuous area (created through channel bank movement in one timestep) that is still considered a bar and not a cutoff.
    ax : figure axes
        Axes for plotting.
    W : float
        Channel width.

    Returns
    -------
    bars : list
        Shapely multipolygons representing 'scroll' bars that result from channel migration during one timestep.
    chs : list
        Shapely polygons that represent channels through time.
    all_chs : list
        Shapely polygons that represent merged channels through time.
    jumps : list
        Sometimes there is a gap between two consecutive channels and these gaps are collected into a list of polygons.
    cutoffs ; list
        Shapely polygons that represent cutoffs.
    """

    bars, chs, all_chs, jumps, cutoffs = compute_bars_from_centerline(graph, cutoff_area, W)
    ts = len(bars)
    cmap = mpl.colormaps['viridis']
    for i in range(ts):
        bar = bars[i]
        color = cmap(i/float(ts))
        if type(bar) != Polygon:
            for b in bar.geoms:
                if MultiPolygon(cutoffs[i]).is_valid: # sometimes this is invalid
                    if not b.intersects(MultiPolygon(cutoffs[i])):
                        ax.fill(b.exterior.xy[0], b.exterior.xy[1], facecolor=color, edgecolor='k')
                else:
                    ax.fill(b.exterior.xy[0], b.exterior.xy[1], facecolor=color, edgecolor='k')
        else:
            ax.fill(bar.exterior.xy[0], bar.exterior.xy[1], facecolor=color, edgecolor='k')
    return bars, chs, all_chs, jumps, cutoffs


def compute_bars_from_banks(graph1, graph2, cutoff_area):
    """
    Compute channel/scroll-bar/cutoff polygons from bankline data (no
    plotting).

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
    bars : list
        Shapely multipolygons representing 'scroll' bars that result from channel migration during one timestep.
    chs : list
        Shapely polygons that represent channels through time.
    all_chs : list
        Shapely polygons that represent merged channels through time.
    jumps : list
        Sometimes there is a gap between two consecutive channels and these gaps are collected into a list of polygons.
    cutoffs : list
        Shapely polygons that represent cutoffs.
    """

    n_centerlines = graph1.graph['number_of_centerlines']
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    for node in np.arange(n_centerlines):
        path = find_longitudinal_path(graph1, node)
        X1.append(graph1.graph['x'][path])
        Y1.append(graph1.graph['y'][path])
        path = find_longitudinal_path(graph2, node)
        X2.append(graph2.graph['x'][path])
        Y2.append(graph2.graph['y'][path])
    ts = len(X1)
    bars = [] # these are 'scroll' bars - shapely MultiPolygon objects that correspond to one time step
    chs = [] # list of channels - shapely Polygon objects
    jumps = [] # gaps between channel polygons that are not cutoffs
    all_chs = [] # list of merged channels (to be used for erosion)
    cutoffs = []
    # creating list of channels, jumps, and cutoffs
    for i in trange(ts-1):
        ch1 = create_channel_polygon_from_banks(X1[i], Y1[i], X2[i], Y2[i])
        ch2 = create_channel_polygon_from_banks(X1[i+1], Y1[i+1], X2[i+1], Y2[i+1])
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
    # creating scroll bars
    for i in trange(ts): # create scroll bars
        bar = chs[i].difference(all_chs[ts-i-1]) # scroll bar defined by difference
        bars.append(bar)
    return bars, chs, all_chs, jumps, cutoffs

def plot_bars_from_banks(graph1, graph2, cutoff_area, ax):
    """
    Create polygons for 'scroll' bars from channel bankline data and plot them.

    Parameters
    ----------
    graph1 : directed graph
        Bankline graph.
    graph2 : directed graph
        Bankline graph.
    cutoff_area : float
        Maximum continuous area (created through channel bank movement in one timestep) that is still considered a bar and not a cutoff.
    ax : figure axes
        Axes for plotting.

    Returns
    -------
    bars : list
        Shapely multipolygons representing 'scroll' bars that result from channel migration during one timestep.
    chs : list
        Shapely polygons that represent channels through time.
    all_chs : list
        Shapely polygons that represent merged channels through time.
    jumps : list
        Sometimes there is a gap between two consecutive channels and these gaps are collected into a list of polygons.
    cutoffs : list
        Shapely polygons that represent cutoffs.
    """

    bars, chs, all_chs, jumps, cutoffs = compute_bars_from_banks(graph1, graph2, cutoff_area)
    ts = len(bars)
    cmap = mpl.colormaps['viridis']
    for i in range(ts):
        bar = bars[i]
        color = cmap(i/float(ts))
        if type(bar) != Polygon:
            for b in bar.geoms:
                recreate_cutoff_list = False
                for obj in cutoffs[i]:
                    if type(obj) == MultiPolygon:
                        recreate_cutoff_list = True
                if recreate_cutoff_list:
                    cutoffs_new = []
                    for obj in cutoffs[i]:
                        if type(obj) == MultiPolygon:
                            for obj2 in obj.geoms:
                                cutoffs_new.append(obj2)
                        if type(obj) == Polygon:
                            cutoffs_new.append(obj)
                    cutoffs[i] = cutoffs_new
                if MultiPolygon(cutoffs[i]).is_valid: # sometimes this is invalid
                    if not b.intersects(MultiPolygon(cutoffs[i])):
                        ax.fill(b.exterior.xy[0], b.exterior.xy[1], facecolor=color,edgecolor='k')
                else:
                    ax.fill(b.exterior.xy[0], b.exterior.xy[1], facecolor=color,edgecolor='k')
        else:
            ax.fill(bar.exterior.xy[0], bar.exterior.xy[1], facecolor=color,edgecolor='k')
    plt.axis('equal')
    return bars, chs, all_chs, jumps, cutoffs


def plot_migration_rate_map(wbar, graph1, graph2, vmin, vmax, ax):
    """
    Make a spatial plot of migration rate.

    Parameters
    ----------
    wbar : bar object
    graph1 : directed graph
        Graph of center- or banklines.
    graph2 : directed graph
        Graph of center- or banklines.
    vmin : float
        Minimum value of migration rate (for scaling)
    vmax : float
        Maximum value of migration rate (for scaling)
    ax : int
        Axes for plotting
    """
    if wbar.scrolls[-1].bank == 'left':
        graph = graph2
    if wbar.scrolls[-1].bank == 'right':
        graph = graph1
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    m = mpl.cm.ScalarMappable(norm=norm, cmap='coolwarm')
    #time_step = (dt * saved_ts)/(365*24*60*60)
    for node in wbar.bar_graph.nodes:
        length = wbar.bar_graph.nodes[node]['length']
        time_step = wbar.bar_graph.nodes[node]['timestep']
        if type(wbar.bar_graph.nodes[node]['poly']) == Polygon:
            ax.fill(wbar.bar_graph.nodes[node]['poly'].exterior.xy[0], 
                wbar.bar_graph.nodes[node]['poly'].exterior.xy[1], 
                facecolor = m.to_rgba(length/time_step * wbar.bar_graph.nodes[node]['direction']), 
                edgecolor='k', linewidth=0.25)


def plot_curvature_map(wbar, vmin, vmax, W, ax, cmap='coolwarm'):
    """
    Make a spatial plot of curvature.

    Parameters
    ----------
    wbar : bar object
    vmin : float
        Minimum value of dimensionless curvature (W * curvature), for scaling.
        Use a negative vmin (e.g., -vmax) so that both curvature signs are shown.
    vmax : float
        Maximum value of dimensionless curvature (for scaling)
    W : int/float
        Width (meters)
    ax : int
        Axes for plotting
    cmap: str
        Matplotlib colormap name; a diverging colormap shows the two curvature signs
    """
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    m = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    for node in wbar.bar_graph.nodes:
        if type(wbar.bar_graph.nodes[node]['poly']) == Polygon:
            ax.fill(wbar.bar_graph.nodes[node]['poly'].exterior.xy[0], 
                    wbar.bar_graph.nodes[node]['poly'].exterior.xy[1], 
                    facecolor = m.to_rgba(W * wbar.bar_graph.nodes[node]['curv']), 
                    edgecolor='k', linewidth=0.25)


def plot_age_map(wbar, vmin, vmax, ax):
    """
    Make a spatial plot of age.

    Parameters
    ----------
    wbar : 'bar' object
    vmin : float
        Minimum value of age (for scaling)
    vmax : float
        Maximum value of age (for scaling)
    W : int/float
        Width (meters)
    cmap: str 
        Matplotlib cmap object
    ax : int
        Axes for plotting
    """
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    m = mpl.cm.ScalarMappable(norm=norm, cmap='YlGn_r')
    for node in wbar.bar_graph.nodes:
            poly = wbar.bar_graph.nodes[node]['poly']
            if type(poly) == Polygon:
                ax.fill(poly.exterior.xy[0], poly.exterior.xy[1], 
                    facecolor = m.to_rgba(wbar.bar_graph.nodes[node]['age']), 
                    edgecolor='k', linewidth=0.25)


def plot_bar_lines(wbar, graph1, graph2, ax):
    """
    Create a plot of the outline of the bar polygons.

    Parameters
    ----------
    wbar : 'bar' object
    graph1 : directed graph
        Bar graph for one of the banks
    graph2 : directed graph
        Bar graph for one of the banks
    ax : int
        Axes for plotting
    """
    if wbar.scrolls[-1].bank == 'right':
        bank_graph = graph1 
    else:
        bank_graph = graph2
    cmap = plt.get_cmap("tab10")
    source_nodes = [] # source nodes for longitudinal lines
    for node in wbar.bar_graph.nodes:
        if wbar.bar_graph.in_degree(node) == 0:
            source_nodes.append(node)
    for node in source_nodes:
        path = find_longitudinal_path(wbar.bar_graph, node)
        x = bank_graph.graph['x'][path]
        y = bank_graph.graph['y'][path]
        if len(x) > 1:
            line = LineString(np.vstack((x,y)).T).intersection(wbar.polygon)
            if type(line) != MultiLineString:
                x1 = line.xy[0]
                y1 = line.xy[1]
                ax.plot(x1, y1, color=cmap(0), linewidth=0.5)
            else:
                for l in line.geoms:
                    x1 = l.xy[0]
                    y1 = l.xy[1]
                    ax.plot(x1, y1, color=cmap(0), linewidth=0.5)
    temp_radial_graph = nx.DiGraph() # temporary radial graph for radial lines
    for node in wbar.bar_graph.nodes:
        temp_radial_graph.add_node(node, x=bank_graph.graph['x'][node], y=bank_graph.graph['y'][node])
    for s in wbar.bar_graph.nodes:
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
                line = LineString(np.vstack((x,y)).T).intersection(wbar.polygon)
                if type(line) != MultiLineString:
                    x1 = line.xy[0]
                    y1 = line.xy[1]
                    ax.plot(x1, y1, color=cmap(1), linewidth=0.5)
                else:
                    for l in line.geoms:
                        x1 = l.xy[0]
                        y1 = l.xy[1]
                        ax.plot(x1, y1, color=cmap(1), linewidth=0.5)
    fill_polygon(wbar.polygon, ax, facecolor='none', edgecolor='k', linewidth = 2, zorder = 10000)


def plot_bar_graphs(graph1, graph2, wbars, cutoffs, X1, Y1, X2, Y2, W, vmin, vmax, plot_type, ax):
    """
    Make and plot a graph containing 'bar' objects.

    Parameters
    ----------
    graph1 : directed graph
        Polygon graph representing bankline
    graph2 : directed graph
         Polygon graph representing bankline
    wbars : list
        List of 'bar' objects
    cutoffs : list
        Shapely polygons that represent cutoffs.
    X1 : list 
        x coordinate arrays.
    Y1 : list
        y coordinate arrays.
    X2 : list 
        x coordinate arrays.
    Y2 : list
        y coordinate arrays.
    W : float
        Channel width (meters)
    vmin : float
        Minimum value of parameter of interest (for scaling)
    vmax : float
        Maximum value of parameter of interest (for scaling)
    plot_type : str
        Type of plot to product; available options 'migration', 'curvature', 'age'
    ax : int
        Axes for plotting

    Returns
    -------
    graph : directed graph
        Bar graph
    """
    # collect cutoff indices
    cutoff_inds = []
    count = 0
    for cf in cutoffs:
        if len(cf) > 0:
            cutoff_inds.append(count)
        count += 1
    # plotting cutoffs     
    for wbar in wbars:
        ages = []
        for scroll in wbar.scrolls:
            ages.append(scroll.age)
        for i in cutoff_inds: # cutoffs need to be plotted at the right time
            if max(ages) + 1 == i:
                ax.fill(cutoffs[i][0].exterior.xy[0], cutoffs[i][0].exterior.xy[1], facecolor='lightblue', edgecolor='k')
    # create polygon for most recent channel and plot it
    ch = create_channel_polygon_from_banks(X1[-1], Y1[-1], X2[-1], Y2[-1])
    ax.fill(ch.exterior.xy[0], ch.exterior.xy[1], facecolor='lightblue', edgecolor='k')
    # add polygon graphs to bars and plot them (based on plot_type)
    for i in trange(len(wbars)):
        if plot_type == 'migration':
            plot_migration_rate_map(wbars[i], graph1, graph2, vmin, vmax, ax)
        if plot_type == 'curvature':
            plot_curvature_map(wbars[i], vmin, vmax, W, ax)
        if plot_type == 'age':
            plot_age_map(wbars[i], vmin, vmax, ax)
    plt.axis('equal')


def plot_bars_by_bar_number(wbars, ax):
    """
    Make a plot of the bar objects. 

    Parameters
    ----------
    wbars : list of 'wbar' objects
    ax : int
        Axes for plotting
    """
    for wbar in wbars:
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b, 0.5) # random color for each bar
        wbar.plot(ax, color)

# from: https://www.geeksforgeeks.org/direction-point-line-segment/


def plot_simple_polygon_graph(poly_graph, ax, bank_type):
    """
    Make a plot of a simple polygon graph

    Parameters
    ----------
    graph : directed graph
        Simple polygon graph
    ax : int
        Axes for plotting
    bank_type : str
        String specifying the bank as either 'right' or 'left'
    """  
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
                            ax.fill(poly_graph.nodes[node]['poly'].exterior.xy[0], poly_graph.nodes[node]['poly'].exterior.xy[1], facecolor = cmap(1), edgecolor='k', linewidth = 0.3, alpha = 0.5, zorder = count)
                        if poly_graph.nodes[node]['direction'] == 1:
                            ax.fill(poly_graph.nodes[node]['poly'].exterior.xy[0], poly_graph.nodes[node]['poly'].exterior.xy[1], facecolor = cmap(0), edgecolor='k', linewidth = 0.3, alpha = 0.5, zorder = count)
                    if bank_type == 'right':
                        if poly_graph.nodes[node]['direction'] == -1:
                            ax.fill(poly_graph.nodes[node]['poly'].exterior.xy[0], poly_graph.nodes[node]['poly'].exterior.xy[1], facecolor = cmap(0), edgecolor='k', linewidth = 0.3, alpha = 0.5, zorder = count)
                        if poly_graph.nodes[node]['direction'] == 1:
                            ax.fill(poly_graph.nodes[node]['poly'].exterior.xy[0], poly_graph.nodes[node]['poly'].exterior.xy[1], facecolor = cmap(1), edgecolor='k', linewidth = 0.3, alpha = 0.5, zorder = count)
            count += 1


def plot_chosen_radial_paths(graph, X, Y, P, Q, num_paths, cutoff_index = False):
    """
    Produce a plot along a primary radial path for a user-specified region of the graph.
    Location is defined with the cursor.

    Parameters
    ----------
    graph : directed graph
            Centerline graph.
    X : list
        x coordinates of lines.
    Y : list
        y coordinates of lines.
    P : list
        Arrays of indices of correlated successive pairs of curves (for first curve).
    Q : list
        Arrays of indices of correlated successive pairs of curves (for second curve)
    num_paths : int
                The number of desired primary radial paths.
    cutoff_index = int (Optional)
                   Number corresponding to the cutoff year. 
    """

    from matplotlib.widgets import Cursor

    # plot a figure of the directed graph
    fig,ax = plt.subplots()
    plot_graph(graph, ax, show_nodes = False, label_nodes = False)
    
    # user selects (clicks) location where they want a primary radial path
    cursor = Cursor(ax, useblit=True, color='k', linewidth=1)
    zoom_ok = False
    print('\nZoom or pan to view, \npress spacebar when ready to click:\n')
    while not zoom_ok:
        zoom_ok = plt.waitforbuttonpress()
    user_loc = plt.ginput(n=num_paths)
    plt.close(fig)

    # find the x,y and centerline number for the nearest to the point that has been clicked
    last_cl_points = np.vstack((X[-1], Y[-1])).T # coordinates of last centerlines
    tree = KDTree(last_cl_points)
    
    X_flip, Y_flip, P_flip, Q_flip = np.flip(X), np.flip(Y), np.flip(P), np.flip(Q)

    # find x,y for points along the nearest radial path
    user_indices = []
    user_xs = []
    user_ys = []
    for i in range(len(user_loc)):
        user_ind = tree.query(user_loc[i])[1]
        indices, x, y = find_indices(user_ind, X_flip, Y_flip, Q_flip, P_flip)
        user_indices.append(indices)
        user_xs.append(x)
        user_ys.append(y)    
    
    # calculate distance between successive nodes along the radial path
    user_dists = []
    for inds in range(len(user_indices)):
        dist = []
        for i in range(len(user_indices[inds])-1):
            dist.append(((user_xs[inds][i]-user_xs[inds][i+1])**2 + (user_ys[inds][i]-user_ys[inds][i+1])**2)**0.5)
        user_dists.append(dist)
    
    # find the maximum distance (necessary for scaling the y-axis)
    max_dist = 0
    for item in user_dists:
        for dist in item:
            if dist>max_dist:
                max_dist = dist
            else:
                dist+=1

    # define color scheme
    colors = plt.cm.inferno(np.linspace(0, 1, num_paths))
    
    # re-plot graph with the paths drawn and colored
    fig, ax = plt.subplots(figsize=(20,15))
    plot_graph(graph, ax, show_nodes = False, label_nodes = False)
    for i in range(num_paths):
        ax.plot(user_xs[i], user_ys[i], 'o', color=colors[i], markersize = 3)
    ax.axis('equal')
    
    # x-plot migration rate vs age for the chosen paths
    fig, ax = plt.subplots(figsize=(7,5))
    for i in range(len(user_dists)):
        ages = np.arange(len(user_dists[i]))
        user_dists[i].reverse()
        
        # plot the data
        ax.plot(ages,user_dists[i], color=colors[i])
        
    # add the cutoffs years to the plot as vertical lines
    ax.axvline(x=cutoff_index, ymin=0, ymax=len(X), color='gray', linestyle='--')

    # add plot elements
    ax.set_xlabel('Time (year)')
    ax.set_ylabel('Migration Rate (m/yr)')
    ax.set_ylim(0, max_dist+1)


def fill_polygon(poly, ax, **kwargs):
    """
    Plot a filled polygon on 'ax'; works whether 'poly' is a Polygon or a MultiPolygon.
    Keyword arguments are passed on to 'ax.fill'.
    """

    for geom in ensure_multipolygon(poly).geoms:
        ax.fill(geom.exterior.xy[0], geom.exterior.xy[1], **kwargs)


