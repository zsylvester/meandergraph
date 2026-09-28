"""
Correlate successive center- or banklines with dynamic time warping (via
librosa.sequence.dtw), and the curve-resampling / curvature / timestep
helpers built on top of it.
"""
import numpy as np
from scipy.spatial import distance, KDTree
from scipy import interpolate
from librosa.sequence import dtw
from tqdm import trange

__all__ = [
    "find_next_index", "correlate_curves", "correlate_set_of_curves",
    "find_indices", "restrict_and_correlate_lines", "compute_derivatives",
    "compute_curvature", "resample_centerline", "get_timesteps",
]

def find_next_index(p, q, ind1):
    """
    Find index 'ind2' of the next point on the next curve if the current index is 'ind1'.

    Parameters
    ----------
    p : 1D array
        Correlation indices for first curve.
    q : 1D array
        Correlation indices for second curve.
    ind1 : int
        Index of point of interest in first curve.

    Returns
    -------
    ind2 : int
        Index of correlated point in second curve.
    """

    p_index = np.where(p == ind1)[0] # find the location where 'p' equals 'ind1'
    p_index = int(np.median(p_index)) # have to choose only one if there are more than one
    ind2 = q[p_index] # find the equivalent index in 'q'
    return ind2


def correlate_curves(x1,x2,y1,y2,band_rad=None):
    """ 
    Use dynamic time warping to correlate two 2D curves.

    Parameters
    ----------
    x1 : 1D array
        x-coordinates of first curve.
    x2 : 1D array
        x-coordinates of second curve.
    y1 : 1D array
        y-coordinates of first curve.
    y2 : 1D array
        y-coordinates of second curve.

    Returns
    -------
    p : 1D array
        Correlation indices for first curve.
    q : 1D array
        Correlation indices for second curve.
    cost : float
        Total dynamic time warping cost of the correlation.
    """

    X = np.vstack((x1,y1))
    Y = np.vstack((x2,y2))
    # sm = distance.cdist(X.T, Y.T) # similarity matrix
    # D, wp = dtw(C=sm) # dynamic time warping
    # if band_rad:
    D, wp = dtw(X, Y, band_rad=band_rad)
    # else:
    #     D, wp = dtw(X, Y)
    p = wp[:,0] # correlation indices for first curve
    q = wp[:,1] # correlation indices for second curve
    return p, q, D[-1,-1]

# def correlate_curves_fdtw(x1,x2,y1,y2):
#     X = np.vstack((x1,y1)).T
#     Y = np.vstack((x2,y2)).T
#     distance, path = fastdtw(X, Y)
#     path = np.array(path)
#     return path[::-1,0], path[::-1,1], distance


def correlate_set_of_curves(X, Y):
    """
    Correlate a set of curves defined by x and y coordinates stored as two lists X and Y.

    Parameters
    ----------
    X : list 
        x coordinate arrays.
    Y : list
        y coordinate arrays.

    Returns
    -------
    P : list
        Arrays of indices of correlated successive pairs of curves (for first curve)
    Q : list
        Arrays of indices of correlated successive pairs of curves (for second curve)
    """

    P = []
    Q = []
    costs = []
    for i in trange(len(X) - 1):
        p, q, cost = correlate_curves(X[i], X[i+1], Y[i], Y[i+1])
        P.append(p)
        Q.append(q)
        costs.append(cost)
    return(P, Q, costs)


def find_indices(ind1, X, Y, P, Q):
    """
    Tracks one index through a series of centerlines (stored as lists of coordinates X and Y) 

    Parameters
    ----------
    ind1 : int
        Index of point of interest in first curve.
    X : list 
        x coordinate arrays.
    Y : list
        y coordinate arrays.
    P : list 
        Arrays of indices of correlated successive pairs of curves (for first curve)
    Q : list
        Arrays of indices of correlated successive pairs of curves (for second curve)

    Returns
    -------
    indices : list
        Indices that define the correlation path (includes 'ind1'); has same length as 'X'
    x : 1D array
        x-coordinates of correlation path
    y : 1D array
        y-coordinates of correlation path
    """

    indices = []
    x = []
    y = []
    indices.append(ind1)
    x.append(X[0][ind1])
    y.append(Y[0][ind1])
    n_centerlines = len(X)
    for i in range(n_centerlines-1):
        # ind2 = find_next_index(X[i], Y[i], X[i+1], Y[i+1], P[i], Q[i], ind1)
        ind2 = find_next_index(P[i], Q[i], ind1)
        indices.append(ind2)
        x.append(X[i+1][ind2])
        y.append(Y[i+1][ind2])
        ind1 = ind2
    x = np.array(x)
    y = np.array(y)
    return indices, x, y


def restrict_and_correlate_lines(X, Y, points, delta_s=2.0):
    """
    Restrict centerlines or banklines to a specified segment and correlate them across time.
    
    This function takes a set of centerlines or banklines and restricts them to a segment 
    defined by two points, then resamples and correlates the restricted 
    centerlines to establish correspondence between points across different 
    time steps.
    
    Parameters
    ----------
    X : list of array-like
        List of x-coordinates for each centerline / bankline. Each element is an array
        containing the x-coordinates of points along one centerline / bankline.
    Y : list of array-like
        List of y-coordinates for each centerline / bankline. Each element is an array
        containing the y-coordinates of points along one centerline.
        Must have the same length as X.
    points : array-like of shape (2, 2)
        Two points defining the segment boundaries. Each point should be 
        [x, y] coordinates. The centerlines will be restricted to the 
        segment between these two points.
    delta_s : float, optional
        Target spacing for resampling the centerlines, by default 2.0.
        Units should match the coordinate system of X and Y. If
    
    Returns
    -------
    X : list of ndarray
        Restricted and resampled x-coordinates for each centerline / bankline.
    Y : list of ndarray
        Restricted and resampled y-coordinates for each centerline / bankline.
    P : list of ndarray
        Correlation indices from first to second centerline / bankline for each 
        consecutive pair. P[i] contains indices mapping points from 
        centerline i to centerline i+1.
    Q : list of ndarray
        Correlation indices from second to first centerline for each
        consecutive pair. Q[i] contains indices mapping points from
        centerline i+1 to centerline i.
    costs : ndarray
        Correlation costs between consecutive centerline pairs, representing
        the quality of the correlation match.
    
    Notes
    -----
    The function performs the following steps:
    1. Finds the closest points on the first centerline to the specified 
       boundary points using a KDTree for efficient nearest neighbor search.
    2. Performs pairwise correlation between consecutive centerlines to 
       establish point correspondence.
    3. Restricts all centerlines to the segment defined by the boundary points.
    4. Resamples each restricted centerline with the specified spacing.
    5. Correlates the entire set of restricted and resampled centerlines.
    
    The correlation establishes which points on different centerlines 
    correspond to the same physical location along the river channel,
    enabling temporal analysis of channel migration.
    
    Examples
    --------
    >>> # Define boundary points for restriction
    >>> boundary_points = [[1000, 2000], [1500, 2500]]
    >>> 
    >>> # Restrict and correlate centerlines
    >>> X_res, Y_res, P, Q, costs = restrict_and_correlate_lines(
    ...     X_centerlines, Y_centerlines, boundary_points, delta_s=5.0
    ... )
    >>> 
    >>> # X_res and Y_res now contain restricted, resampled centerlines
    >>> # P and Q contain correlation indices between consecutive pairs
    """
    X = list(X) # shallow copies, so that the input lists are not modified in place
    Y = list(Y)
    cl_points = np.vstack((X[0], Y[0])).T # coordinates of first centerline
    tree = KDTree(cl_points)
    first_index = tree.query(np.array(points[0]).reshape(1, -1))[1][0]
    last_index = tree.query(np.array(points[1]).reshape(1, -1))[1][0]
    if first_index > last_index:
        first_index, last_index = last_index, first_index
    P = []
    Q = []
    for i in trange(len(X) - 1):
        p, q, dist = correlate_curves(X[i], X[i+1], Y[i], Y[i+1])
        P.append(p)
        Q.append(q)
    indices1, x, y = find_indices(first_index, X, Y, P, Q)
    indices2, x, y = find_indices(last_index, X, Y, P, Q)
    for i in range(len(X)):
        X[i] = X[i][indices1[i] : indices2[i]+1]
        Y[i] = Y[i][indices1[i] : indices2[i]+1]
    for i in range(len(X)):
        x,y,dx,dy,ds,s = resample_centerline(X[i], Y[i], delta_s)
        X[i] = x
        Y[i] = y
    P, Q, costs = correlate_set_of_curves(X, Y)
    return X, Y, P, Q, costs


def compute_derivatives(x, y):
    """
    Compute first derivatives of a curve (centerline).

    Parameters
    ----------
    x : 1D array
        x coodinates of the curve
    y : 1D array
        y coordinates of the curve

    Returns
    -------
    dx : 1D array
        First derivative of the x coordinate.
    dy : 1D array
        First derivative of the y coordinate.
    ds : 1D array
        Distances between consecutive points along the curve.
    s : 1D array
        Cumulative distance along the curve.
    """

    dx = np.diff(x) # first derivatives
    dy = np.diff(y)   
    ds = np.sqrt(dx**2+dy**2)
    s = np.hstack((0,np.cumsum(ds)))
    return dx, dy, ds, s


def compute_curvature(x,y):
    """function for computing first derivatives and curvature of a curve (centerline)
    x,y are cartesian coodinates of the curve
    outputs:
    dx - first derivative of x coordinate
    dy - first derivative of y coordinate
    ds - distances between consecutive points along the curve
    s - cumulative distance along the curve
    curvature - curvature of the curve (in 1/units of x and y)"""
    dx = np.gradient(x) # first derivatives
    dy = np.gradient(y)      
    ddx = np.gradient(dx) # second derivatives 
    ddy = np.gradient(dy) 
    curvature = (dx*ddy-dy*ddx)/((dx**2+dy**2)**1.5)
    return curvature


def resample_centerline(x, y, deltas):
    '''resample centerline so that 'deltas' is roughly constant, using parametric 
    spline representation of curve; note that there is *no* smoothing

    :param x: x-coordinates of centerline
    :param y: y-coordinates of centerline
    :param z: z-coordinates of centerline
    :param deltas: distance between points on centerline
    :return x: x-coordinates of resampled centerline
    :return y: y-coordinates of resampled centerline
    :return z: z-coordinates of resampled centerline
    :return dx: dx of resampled centerline
    :return dy: dy of resampled centerline
    :return dz: dz of resampled centerline
    :return s: s-coordinates of resampled centerline'''

    dx, dy, ds, s = compute_derivatives(x,y) # compute derivatives
    tck, u = interpolate.splprep([x,y],s=0) 
    unew = np.linspace(0,1,1+int(round(s[-1]/deltas))) # vector for resampling
    out = interpolate.splev(unew,tck) # resampling
    x, y = out[0], out[1] # assign new coordinate values
    dx, dy, ds, s = compute_derivatives(x,y) # recompute derivatives
    return x,y,dx,dy,ds,s


def get_timesteps(dates_list):
    """
    Make a list whose elements correspond to the amount of time between the successive longitudinal paths.

    Parameters
    ----------

    dates_list : 1D list
                 Date corresponding to each longitudinal path. Elements should be datetime objects.

    Returns
    -------
    timesteps : 1D list
                The amount of time between successive longitudinal path with units of years.
    """
    timesteps = []
    for i in range(len(dates_list)-1):
        timestep = abs(dates_list[i]-dates_list[i+1]).days
        timesteps.append(timestep/365)
    return timesteps


