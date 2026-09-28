import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
from shapely.geometry import Polygon, LinearRing, MultiPolygon, Point, MultiLineString, LineString, shape, JOIN_STYLE
import meandergraph as mg
import matplotlib as mpl

def plot_meander_graph_in_3D(graph, nodes, z_factor):
    x = []
    y = []
    z = []
    triangles = []
    count = 0
    for node in nodes:
        if type(graph.nodes[node]['poly']) == Polygon:
            x1 = graph.nodes[node]['poly'].boundary.xy[0]
            y1 = graph.nodes[node]['poly'].boundary.xy[1]
            if len(x1) == 5:
                for i in range(4):
                    x.append(x1[i])
                    y.append(y1[i])
                    if (i == 0) or (i == 1):
                        z.append(z_factor * graph.nodes[node]['age'] - z_factor*0.5)
                    else:
                        z.append(z_factor * graph.nodes[node]['age'] + z_factor*0.5)
                triangles.append([count, count+1, count+3])
                triangles.append([count+1, count+2, count+3])
                count += 4
            elif len(x1) == 6:
                for i in range(5):
                    x.append(x1[i])
                    y.append(y1[i])
                    if (i == 0) or (i == 1):
                        z.append(z_factor * graph.nodes[node]['age'] - z_factor*0.5)
                    else:
                        z.append(z_factor * graph.nodes[node]['age'] + z_factor*0.5)
                triangles.append([count, count+3, count+4])
                triangles.append([count, count+1, count+3])
                triangles.append([count+1, count+2, count+3])
                count += 5
            elif len(x1) == 4:
                for i in range(3):
                    x.append(x1[i])
                    y.append(y1[i])
                    if (i == 0) or (i == 1):
                        z.append(z_factor * graph.nodes[node]['age'] - z_factor*0.5)
                    else:
                        z.append(z_factor * graph.nodes[node]['age'] + z_factor*0.5)
                triangles.append([count, count+1, count+2])
                count += 3
        elif type(graph.nodes[node]['poly']) == MultiPolygon:
            if len(graph.nodes[node]['poly'].geoms[1].boundary.xy[0]) == 4:
                x1 = graph.nodes[node]['poly'].geoms[0].boundary.xy[0][0]
                y1 = graph.nodes[node]['poly'].geoms[0].boundary.xy[1][0]
                x2 = graph.nodes[node]['poly'].geoms[1].boundary.xy[0][2]
                y2 = graph.nodes[node]['poly'].geoms[1].boundary.xy[1][2]
                x3 = graph.nodes[node]['poly'].geoms[1].boundary.xy[0][1]
                y3 = graph.nodes[node]['poly'].geoms[1].boundary.xy[1][1]
                x4 = graph.nodes[node]['poly'].geoms[0].boundary.xy[0][2]
                y4 = graph.nodes[node]['poly'].geoms[0].boundary.xy[1][2]
                x.append(x1)
                y.append(y1)
                z.append(z_factor * graph.nodes[node]['age'] - z_factor*0.5)
                x.append(x2)
                y.append(y2)
                z.append(z_factor * graph.nodes[node]['age'] - z_factor*0.5)
                x.append(x3)
                y.append(y3)
                z.append(z_factor * graph.nodes[node]['age'] + z_factor*0.5)
                x.append(x4)
                y.append(y4)
                z.append(z_factor * graph.nodes[node]['age'] + z_factor*0.5)
                triangles.append([count, count+1, count+3])
                triangles.append([count+1, count+2, count+3])
                count += 4
            if len(graph.nodes[node]['poly'].geoms[1].boundary.xy[0]) == 5:
                x1 = graph.nodes[node]['poly'].geoms[0].boundary.xy[0][0]
                y1 = graph.nodes[node]['poly'].geoms[0].boundary.xy[1][0]
                x2 = graph.nodes[node]['poly'].geoms[1].boundary.xy[0][1]
                y2 = graph.nodes[node]['poly'].geoms[1].boundary.xy[1][1]
                x3 = graph.nodes[node]['poly'].geoms[1].boundary.xy[0][2]
                y3 = graph.nodes[node]['poly'].geoms[1].boundary.xy[1][2]
                x4 = graph.nodes[node]['poly'].geoms[1].boundary.xy[0][3]
                y4 = graph.nodes[node]['poly'].geoms[1].boundary.xy[1][3]
                x5 = graph.nodes[node]['poly'].geoms[0].boundary.xy[0][2]
                y5 = graph.nodes[node]['poly'].geoms[0].boundary.xy[1][2]
                x.append(x1)
                y.append(y1)
                z.append(z_factor * graph.nodes[node]['age'] - z_factor*0.5)
                x.append(x2)
                y.append(y2)
                z.append(z_factor * graph.nodes[node]['age'] - z_factor*0.5)
                x.append(x3)
                y.append(y3)
                z.append(z_factor * graph.nodes[node]['age'] + z_factor*0.5)
                x.append(x4)
                y.append(y4)
                z.append(z_factor * graph.nodes[node]['age'] + z_factor*0.5)
                x.append(x5)
                y.append(y5)
                z.append(z_factor * graph.nodes[node]['age'] + z_factor*0.5)
                triangles.append([count, count+3, count+1]) # x1, x4, x2
                triangles.append([count, count+3, count+4]) # x1, x4, x5
                triangles.append([count+1, count+3, count+2]) # x2, x4, x3
                count += 5
    x = np.array(x)/1000; y = np.array(y)/1000; z = np.array(z)/1000
    mlab.triangular_mesh(x, y, z, np.array(triangles), representation='surface', colormap = 'viridis', 
                         vmin=0, vmax=140*z_factor/1000, opacity=1.0)
    
def get_xy_and_triangles_for_channel(graph1, graph2, ts, nodes1, nodes2):
    rb_inds = np.array(mg.find_longitudinal_path(graph1, ts))
    lb_inds = np.array(mg.find_longitudinal_path(graph2, ts))
    x1 = []
    y1 = []
    for i in range(len(rb_inds)):
        if rb_inds[i] in nodes1:
            x1.append(graph1.nodes[rb_inds[i]]['x'])
            y1.append(graph1.nodes[rb_inds[i]]['y'])
    x2 = []
    y2 = []
    for i in range(len(lb_inds)):
        if lb_inds[i] in nodes2:
            x2.append(graph2.nodes[lb_inds[i]]['x'])
            y2.append(graph2.nodes[lb_inds[i]]['y'])
    p, q, cost = mg.correlate_curves(x1, x2, y1, y2)
    triangles = []
    x = []
    y = []
    count = 0
    for i in range(0, len(p)-1):
        if p[i] == p[i+1]:
            x.append(x1[p[i]])
            x.append(x2[q[i]])
            x.append(x2[q[i+1]])
            y.append(y1[p[i]])
            y.append(y2[q[i]])
            y.append(y2[q[i+1]])
            triangles.append([count, count+1, count+2])
            count += 3
        if q[i] == q[i+1]:
            x.append(x1[p[i]])
            x.append(x2[q[i]])
            x.append(x1[p[i+1]])
            y.append(y1[p[i]])
            y.append(y2[q[i]])
            y.append(y1[p[i+1]])
            triangles.append([count, count+1, count+2])
            count += 3
        if (p[i] != p[i+1]) & (q[i] != q[i+1]):
            x.append(x1[p[i]])
            x.append(x2[q[i]])
            x.append(x1[p[i+1]])
            y.append(y1[p[i]])
            y.append(y2[q[i]])
            y.append(y1[p[i+1]])
            x.append(x2[q[i]])
            x.append(x2[q[i+1]])
            x.append(x1[p[i+1]])
            y.append(y2[q[i]])
            y.append(y2[q[i+1]])
            y.append(y1[p[i+1]])
            triangles.append([count, count+1, count+2])
            triangles.append([count+3, count+4, count+5])
            count += 6
    x = np.array(x)/1000; y = np.array(y)/1000
    return x, y, triangles
    
def get_xy_and_triangles_for_channel_stack(graph1, graph2, node1, node2, end_ind, z_factor):
    rb_inds = mg.find_radial_path_2(graph1, node1)
    lb_inds = mg.find_radial_path_2(graph2, node2)
    x1 = []
    y1 = []
    z1 = []
    for i in range(len(rb_inds)):
        if graph1.nodes[rb_inds[i]]['age'] <= end_ind:
            x1.append(graph1.nodes[rb_inds[i]]['x'])
            y1.append(graph1.nodes[rb_inds[i]]['y'])
            z1.append(z_factor*graph1.nodes[rb_inds[i]]['age'] - z_factor*0.5)
    x2 = []
    y2 = []
    z2 = []
    for i in range(len(lb_inds)):
        if graph2.nodes[lb_inds[i]]['age'] <= end_ind:
            x2.append(graph2.nodes[lb_inds[i]]['x'])
            y2.append(graph2.nodes[lb_inds[i]]['y'])
            z2.append(z_factor*graph2.nodes[lb_inds[i]]['age'] - z_factor*0.5)
    triangles = []
    x = []
    y = []
    z = []
    count = 0
    for i in range(0, len(x1)-1):
        x.append(x1[i])
        x.append(x2[i])
        x.append(x2[i+1])
        y.append(y1[i])
        y.append(y2[i])
        y.append(y2[i+1])
        z.append(z1[i])
        z.append(z2[i])
        z.append(z2[i+1])
        triangles.append([count, count+1, count+2])
        count += 3
        x.append(x2[i+1])
        x.append(x1[i+1])
        x.append(x1[i])
        y.append(y2[i+1])
        y.append(y1[i+1])
        y.append(y1[i])
        z.append(z2[i+1])
        z.append(z1[i+1])
        z.append(z1[i])
        triangles.append([count, count+1, count+2])
        count += 3
    x = np.array(x)/1000; y = np.array(y)/1000; z = np.array(z)/1000
    return x, y, z, triangles

def get_cutoff_nodes(graph1, graph2, dead_ends1, dead_ends2, age):
    rb_inds = np.array(mg.find_longitudinal_path(graph1, age))
    lb_inds = np.array(mg.find_longitudinal_path(graph2, age))
    rb_inds_cf = []
    for i in range(len(rb_inds)):
        if rb_inds[i] in dead_ends1:
            rb_inds_cf.append(rb_inds[i])
    ind1 = np.where(rb_inds == rb_inds_cf[0])[0][0]
    ind2 = np.where(rb_inds == rb_inds_cf[-1])[0][0]
    rb_node1 = rb_inds[ind1 - 1]
    rb_node2 = rb_inds[min(ind2 + 1, len(rb_inds)-1)]
    # this is needed so that extra nodes are added at the beginning and the end of cutoffs:
    rb_inds_cf = [rb_node1] + rb_inds_cf + [rb_node2]
    lb_inds_cf = []
    for i in range(len(lb_inds)):
        if lb_inds[i] in dead_ends2:
            lb_inds_cf.append(lb_inds[i])
    ind1 = np.where(lb_inds == lb_inds_cf[0])[0][0]
    ind2 = np.where(lb_inds == lb_inds_cf[-1])[0][0]
    lb_node1 = lb_inds[ind1 - 1]
    lb_node2 = lb_inds[min(ind2 + 1, len(lb_inds)-1)]
    lb_inds_cf = [lb_node1] + lb_inds_cf + [lb_node2]
    return rb_inds_cf, lb_inds_cf, rb_node1, rb_node2, lb_node1, lb_node2

def add_patches_between_top_and_base_of_cutoff(graph1, graph2, rb_node1, rb_node2, lb_node1, lb_node2, z_factor):
    norm = mpl.colors.Normalize(vmin=0, vmax=140*z_factor/1000)
    m = mpl.cm.ScalarMappable(norm=norm, cmap='viridis')
    rb_node1_child = mg.find_radial_path_2(graph1, rb_node1)[1]
    rb_node2_child = mg.find_radial_path_2(graph1, rb_node2)[1]
    lb_node1_child = mg.find_radial_path_2(graph2, lb_node1)[1]
    lb_node2_child = mg.find_radial_path_2(graph2, lb_node2)[1]
    x1 = []
    y1 = []
    z1 = []
    x1.append(graph1.nodes[rb_node1]['x'])
    y1.append(graph1.nodes[rb_node1]['y'])
    z1.append((z_factor*graph1.nodes[rb_node1]['age'] - z_factor*0.5)/1000)
    x1.append(graph1.nodes[rb_node1_child]['x'])
    y1.append(graph1.nodes[rb_node1_child]['y'])
    z1.append((z_factor*graph1.nodes[rb_node1_child]['age'] - z_factor*0.5)/1000)
    x1.append(graph2.nodes[lb_node1_child]['x'])
    y1.append(graph2.nodes[lb_node1_child]['y'])
    z1.append((z_factor*graph2.nodes[lb_node1_child]['age'] - z_factor*0.5)/1000)
    x1.append(graph2.nodes[lb_node1]['x'])
    y1.append(graph2.nodes[lb_node1]['y'])
    z1.append((z_factor*graph2.nodes[lb_node1]['age'] - z_factor*0.5)/1000)
    x1 = np.array(x1)/1000; y1 = np.array(y1)/1000
    triangles = [[0,1,2],[2,0,3]]
    mlab.triangular_mesh(x1,y1,z1, triangles, color = m.to_rgba(graph1.nodes[rb_node1]['age']*z_factor/1000)[:3], 
                             vmin=0, vmax=140*z_factor/1000)
    x1 = []
    y1 = []
    z1 = []
    x1.append(graph1.nodes[rb_node2]['x'])
    y1.append(graph1.nodes[rb_node2]['y'])
    z1.append((z_factor*graph1.nodes[rb_node2]['age'] - z_factor*0.5)/1000)
    x1.append(graph1.nodes[rb_node2_child]['x'])
    y1.append(graph1.nodes[rb_node2_child]['y'])
    z1.append((z_factor*graph1.nodes[rb_node2_child]['age'] - z_factor*0.5)/1000)
    x1.append(graph2.nodes[lb_node2_child]['x'])
    y1.append(graph2.nodes[lb_node2_child]['y'])
    z1.append((z_factor*graph2.nodes[lb_node2_child]['age'] - z_factor*0.5)/1000)
    x1.append(graph2.nodes[lb_node2]['x'])
    y1.append(graph2.nodes[lb_node2]['y'])
    z1.append((z_factor*graph2.nodes[lb_node2]['age'] - z_factor*0.5)/1000)
    x1 = np.array(x1)/1000; y1 = np.array(y1)/1000
    triangles = [[0,1,2],[2,0,3]]
    mlab.triangular_mesh(x1,y1,z1, triangles, color = m.to_rgba(graph1.nodes[rb_node2]['age']*z_factor/1000)[:3],
                             vmin=0, vmax=140*z_factor/1000)
    
def get_base_of_cutoff_nodes(graph1, graph2, age):
    rb_inds = np.array(mg.find_longitudinal_path(graph1, age)[:-4])
    lb_inds = np.array(mg.find_longitudinal_path(graph2, age)[:-4])
    rb_inds_cf = []
    for i in range(len(rb_inds)):
        if rb_inds[i] in graph1.graph['cutoff_nodes']:
            rb_inds_cf.append(rb_inds[i])
    ind1 = np.where(rb_inds == rb_inds_cf[0])[0][0]
    ind2 = np.where(rb_inds == rb_inds_cf[-1])[0][0]
    rb_node1 = rb_inds[ind1 - 1]
    rb_node2 = rb_inds[ind2 + 1]
    # this is needed so that extra nodes are added at the beginning and the end of cutoffs:
    rb_inds_cf = [rb_node1] + rb_inds_cf + [rb_node2]
    lb_inds_cf = []
    for i in range(len(lb_inds)):
        if lb_inds[i] in graph2.graph['cutoff_nodes']:
            lb_inds_cf.append(lb_inds[i])
    ind1 = np.where(lb_inds == lb_inds_cf[0])[0][0]
    ind2 = np.where(lb_inds == lb_inds_cf[-1])[0][0]
    lb_node1 = lb_inds[ind1 - 1]
    lb_node2 = lb_inds[ind2 + 1]
    lb_inds_cf = [lb_node1] + lb_inds_cf + [lb_node2]
    return rb_inds_cf, lb_inds_cf, rb_node1, rb_node2, lb_node1, lb_node2

def add_patches_between_top_and_base_of_cutoff_specific_nodes(graph1, graph2, z_factor, 
                                            top_node_1, top_node_2, base_node_1, base_node_2):
    norm = mpl.colors.Normalize(vmin=0, vmax=140*z_factor/1000)
    m = mpl.cm.ScalarMappable(norm=norm, cmap='viridis')
    x = []; y = []; z = []
    x.append(graph1.nodes[top_node_1]['x'])
    y.append(graph1.nodes[top_node_1]['y'])
    z.append((z_factor*graph1.nodes[top_node_1]['age'] - z_factor*0.5)/1000)
    x.append(graph1.nodes[base_node_1]['x'])
    y.append(graph1.nodes[base_node_1]['y'])
    z.append((z_factor*graph1.nodes[base_node_1]['age'] - z_factor*0.5)/1000)
    x.append(graph2.nodes[base_node_2]['x'])
    y.append(graph2.nodes[base_node_2]['y'])
    z.append((z_factor*graph2.nodes[base_node_2]['age'] - z_factor*0.5)/1000)
    x.append(graph2.nodes[top_node_2]['x'])
    y.append(graph2.nodes[top_node_2]['y'])
    z.append((z_factor*graph2.nodes[top_node_2]['age'] - z_factor*0.5)/1000)
    x = np.array(x)/1000; y = np.array(y)/1000
    triangles = [[0,1,2],[2,0,3]]
    mlab.triangular_mesh(x, y, z, triangles, color = m.to_rgba(graph1.nodes[top_node_1]['age']*z_factor/1000)[:3], 
                             vmin=0, vmax=140*z_factor/1000)

def remove_close_elements_repeatedly(arr, threshold=10):
    while len(arr) > 1:
        # Flag to check if any element was removed in this iteration
        removed = False
        # Check the difference between the first two elements
        if abs(arr[1] - arr[0]) < threshold:
            arr = arr[1:]
            removed = True
        # Check the difference between the last two elements
        if len(arr) > 1 and abs(arr[-1] - arr[-2]) < threshold:
            arr = arr[:-1]
            removed = True
        # If no elements were removed, break the loop
        if not removed:
            break
    return arr

def get_cutoff_indices(graph1, graph2, age, rb_inds_cf, lb_inds_cf, rb_start_ind, rb_end_ind, lb_start_ind, lb_end_ind):
    rb_inds_cf1 = rb_inds_cf[rb_start_ind:rb_end_ind] # right bank indices
    lb_inds_cf1 = lb_inds_cf[lb_start_ind:lb_end_ind] # left bank indices
    rb_inds = np.array(mg.find_longitudinal_path(graph1, age))
    lb_inds = np.array(mg.find_longitudinal_path(graph2, age))
    ind = np.where(rb_inds == rb_inds_cf1[-1])[0][0]
    rb_node1 = rb_inds_cf1[0]
    rb_node2 = rb_inds[min(ind + 1, len(rb_inds)-1)]
    ind = np.where(lb_inds == lb_inds_cf1[-1])[0][0]
    lb_node1 = lb_inds_cf1[0]
    lb_node2 = lb_inds[min(ind + 1, len(lb_inds)-1)]
    rb_inds_cf1 = rb_inds_cf1 + [rb_node2] # add another node at the end
    lb_inds_cf1 = lb_inds_cf1 + [lb_node2] # add another node at the end
    return rb_inds_cf1, lb_inds_cf1

def get_bank_coords(graph1, graph2, age):
    rb_inds = np.array(mg.find_longitudinal_path(graph1, age))
    lb_inds = np.array(mg.find_longitudinal_path(graph2, age))
    xrb = graph1.graph['x'][rb_inds]
    yrb = graph1.graph['y'][rb_inds]
    xlb = graph2.graph['x'][lb_inds]
    ylb = graph2.graph['y'][lb_inds]
    return xrb, yrb, xlb, ylb