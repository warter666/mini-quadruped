"""Static/dynamic stability tests and contact-force allocation, pure numpy.

Two questions a legged controller has to answer:
    1. Is the current set of footholds enough to keep the robot upright?
       -> support polygon + zero-moment point
    2. Given the wrench the body needs, how do we split it over the feet?
       -> least-squares allocation followed by a friction-cone projection
"""

import numpy as np

GRAVITY = 9.81


# ------------------------------------------------------------------- geometry

def convex_hull(points):
    """Counter-clockwise convex hull (monotone chain), points as (N, 2)."""
    pts = sorted({(float(p[0]), float(p[1])) for p in np.asarray(points)})
    if len(pts) <= 2:
        return np.array(pts)
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def point_in_convex(poly, point):
    """True when the point is inside the CCW convex polygon (boundary counts)."""
    poly = np.asarray(poly, dtype=np.float64)
    if len(poly) == 0:
        return False
    if len(poly) == 1:
        return bool(np.allclose(poly[0], point))
    if len(poly) == 2:
        a, b = poly
        ab, ap = b - a, np.asarray(point) - a
        if abs(ab[0] * ap[1] - ab[1] * ap[0]) > 1e-9:
            return False
        t = float(ap @ ab) / float(ab @ ab)
        return -1e-9 <= t <= 1.0 + 1e-9
    p = np.asarray(point, dtype=np.float64)
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < -1e-9:
            return False
    return True


def stability_margin(poly, point):
    """Signed distance from the point to the polygon edge; >0 means inside.

    This is the standard static stability margin: the distance you would have
    to shove the centre of mass before the robot tips over.
    """
    poly = np.asarray(poly, dtype=np.float64)
    p = np.asarray(point, dtype=np.float64)
    if len(poly) < 3:
        return -float("inf")
    best = float("inf")
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        edge = b - a
        n = np.array([-edge[1], edge[0]])
        n = n / (np.linalg.norm(n) + 1e-15)
        best = min(best, float(n @ (p - a)))
    return best


def support_polygon(feet_world):
    """Hull of the stance feet, projected on the ground plane."""
    pts = np.asarray(feet_world, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    return convex_hull(pts[:, :2])


def triangle_incenter(a, b, c):
    """Incentre of a triangle: the point equidistant from all three edges."""
    a, b, c = (np.asarray(p, dtype=np.float64) for p in (a, b, c))
    la = np.linalg.norm(b - c)
    lb = np.linalg.norm(a - c)
    lc = np.linalg.norm(a - b)
    s = la + lb + lc
    if s < 1e-15:
        return a
    return (la * a + lb * b + lc * c) / s


def chebyshev_center(poly):
    """Deepest interior point of a convex polygon and its distance to the edge.

    This is the optimal place to put the CoM projection for a given set of
    footholds, i.e. exactly the target a body-shift planner aims at. For a
    polygon with up to a handful of vertices it is cheapest to try every
    triple of edges: the deepest point of a convex polygon is always the
    incentre of some three of its supporting lines.
    """
    poly = np.asarray(poly, dtype=np.float64)
    n = len(poly)
    if n < 3:
        return None, -float("inf")
    best_p, best_m = None, -float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p = triangle_incenter(poly[i], poly[j], poly[k])
                m = stability_margin(poly, p)
                if m > best_m:
                    best_p, best_m = p, m
    return best_p, float(best_m)


# ------------------------------------------------------------------ criterion

def zmp(com_xy, com_acc_xy, height):
    """Zero-moment point for a point mass at `height`.

    Static stability is the special case acc = 0: the ZMP is the vertical
    projection of the CoM. With acceleration the CoM may sit outside the
    support polygon as long as the ZMP does not -- which is exactly why trot
    (two diagonal feet down) can be stable while walking fast.
    """
    com_xy = np.asarray(com_xy, dtype=np.float64)
    com_acc_xy = np.asarray(com_acc_xy, dtype=np.float64)
    return com_xy - (float(height) / GRAVITY) * com_acc_xy


# -------------------------------------------------------------------- forces

def contact_wrench_matrix(contacts, com):
    """Maps stacked foot forces (3N,) to a 6D wrench about the CoM.

    Row order: (fx, fy, fz, mx, my, mz).
    """
    contacts = np.asarray(contacts, dtype=np.float64).reshape(-1, 3)
    com = np.asarray(com, dtype=np.float64).reshape(3)
    A = np.zeros((6, 3 * len(contacts)))
    for i, p in enumerate(contacts):
        r = p - com
        A[:3, 3 * i:3 * i + 3] = np.eye(3)
        A[3:6, 3 * i:3 * i + 3] = np.array([[0.0, -r[2], r[1]],
                                            [r[2], 0.0, -r[0]],
                                            [-r[1], r[0], 0.0]])
    return A


def solve_forces(contacts, com, force, torque, weights=None):
    """Minimum-norm (optionally weighted) force split satisfying the wrench."""
    A = contact_wrench_matrix(contacts, com)
    b = np.concatenate([np.asarray(force, dtype=np.float64).reshape(3),
                        np.asarray(torque, dtype=np.float64).reshape(3)])
    n = A.shape[1]
    W = np.eye(n) if weights is None else np.diag(np.asarray(weights).reshape(-1))
    W2 = W @ W
    # min ||W x||  s.t.  A x = b
    x = W2 @ A.T @ np.linalg.solve(A @ W2 @ A.T + 1e-12 * np.eye(6), b)
    return x.reshape(-1, 3)


def project_friction_cone(forces, mu=0.6):
    """Scale the tangential part of any force that escapes the cone."""
    f = np.array(forces, dtype=np.float64).reshape(-1, 3)
    for i in range(len(f)):
        tangential = float(np.hypot(f[i, 0], f[i, 1]))
        limit = mu * max(f[i, 2], 0.0)
        if tangential > limit and tangential > 1e-12:
            f[i, 0] *= limit / tangential
            f[i, 1] *= limit / tangential
    return f


def gravity_wrench(mass):
    return np.array([0.0, 0.0, mass * GRAVITY]), np.zeros(3)


def distribute_gravity(contacts, com, mass, mu=0.6):
    """Split a pure gravity load and enforce the friction cone."""
    force, torque = gravity_wrench(mass)
    raw = solve_forces(contacts, com, force, torque)
    return project_friction_cone(raw, mu)
