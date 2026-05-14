import numpy as np

def perpendicular(x, a, b):
    d_ab = np.linalg.norm(a - b)
    d_ax = np.linalg.norm(a - x)
    d_bx = np.linalg.norm(b - x)
    if d_ab != 0:
        if np.dot(a - b, x - b) * np.dot(b - a, x - a) >= 0:
            px, py = b[0] - a[0], b[1] - a[1]
            dAB = px * px + py * py
            u = ((x[0] - a[0]) * px + (x[1] - a[1]) * py) / dAB
            p = np.array([a[0] + u * px, a[1] + u * py])
        else:
            p = a if d_ax < d_bx else b
    else:
        p = a
    return p

def nearest_point_to_obstacle(pose, obstacle):
    nearest = None
    min_d = float('inf')
    for i in range(len(obstacle)):
        pt = perpendicular(pose, np.array(obstacle[i]), np.array(obstacle[(i+1)%len(obstacle)]))
        d = np.linalg.norm(pose - pt)
        if d < min_d:
            min_d = d
            nearest = pt
    return nearest

def polygon_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x,1)))

def polygon_centroid(poly):
    x, y = poly[:, 0], poly[:, 1]
    A = polygon_area(poly)
    cx = np.sum((x + np.roll(x,1)) * (x*np.roll(y,1) - np.roll(x,1)*y)) / (6*A)
    cy = np.sum((y + np.roll(y,1)) * (x*np.roll(y,1) - np.roll(x,1)*y)) / (6*A)
    return np.array([cx, cy])

def nearest_point_on_polygon(point, poly):
    dmin = float('inf')
    nearest = None
    for i in range(len(poly)):
        a, b = poly[i], poly[(i+1)%len(poly)]
        p = perpendicular(point, a, b)
        d = np.linalg.norm(point - p)
        if d < dmin:
            dmin = d
            nearest = p
    return nearest

def point_inside_polygon(point, poly):
    x, y = point
    inside = False
    p1x, p1y = poly[0]
    n = len(poly)
    for i in range(n+1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside
