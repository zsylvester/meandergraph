"""
Shared low-level geometry helpers used across the correlation, graph,
polygon, and bar-building submodules.
"""
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection

__all__ = ["fix_geometry", "compute_distance", "directionOfPoint", "ensure_multipolygon"]

def fix_geometry(geom):
    """Attempt to fix invalid geometries"""
    
    if not geom.is_valid:

        # Try buffer(0) - often fixes self-intersections and topology issues
        fixed = geom.buffer(0)
        
        if not fixed.is_valid:
            # Try more aggressive fixes
            try:
                # For polygons, try to extract exterior only
                if hasattr(geom, 'exterior'):
                    fixed = Polygon(geom.exterior.coords)
                
                # If still invalid, try very small buffer
                if not fixed.is_valid:
                    fixed = geom.buffer(1e-10)
                    
            except Exception as e:
                print(f"Could not fix geometry: {e}")
                return None
        
        return fixed
    return geom


def compute_distance(x1, x2, y1, y2):
    """
    Compute the distance between two nodes.

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
    dist : flaot
        Distance (meters)
    """

    dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    return dist


def directionOfPoint(xa, ya, xb, yb, xp, yp):
    """
    Compute the directionality between two points. 

    Parameters
    ----------
    xa : float
        x coordinate of point 1
    ya : float
        y coordinate of point 1
    xb : float
        x coordinate of point 2
    yb : float
        y coordinate of point 2    
    xp : float
        x coordinate of point 3
    yp : float
        y coordinate of point 3
    Returns
    -------
    value : int
        Returns value based on cross product    
    """
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


def ensure_multipolygon(geom):
    """
    Return 'geom' as a MultiPolygon: wrap a single Polygon, and drop any
    non-polygon parts of a GeometryCollection (points and lines from
    degenerate intersections).
    """

    if type(geom) == MultiPolygon:
        return geom
    if type(geom) == Polygon:
        if geom.is_empty:
            return MultiPolygon([])
        return MultiPolygon([geom])
    return MultiPolygon([g for g in geom.geoms if type(g) == Polygon])


