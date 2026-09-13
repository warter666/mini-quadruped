"""Leg kinematics for a 3-DOF (roll, pitch, knee) quadruped leg -- pure numpy.

Frames
    body    x forward, y left, z up; origin at the trunk centre
    leg     u forward, w downward, so the foot sits at z = -w when the leg
            hangs straight down
    joints  gamma  hip roll  about body x (positive swings the leg left)
            alpha  hip pitch about body y (positive swings the thigh forward)
            beta   knee      about body y, negative bends the knee backwards
                   (the dog-leg configuration: thigh forward, shank back)

Closed-form inverse kinematics, so it can run at kilohertz without an
iterative solver -- this is the same structure used by champ / SpotMicroAI.
"""

import numpy as np

LEG_NAMES = ("FL", "FR", "RL", "RR")
# Chosen so that an unrotated body faces +x and the legs splay outward.
HIP_SIGN = {"FL": (+1.0, +1.0), "FR": (+1.0, -1.0),
            "RL": (-1.0, +1.0), "RR": (-1.0, -1.0)}


class QuadrupedModel:
    """Kinematic parameters and the FK/IK pair for all four legs."""

    def __init__(self, hip_x=0.19, hip_y=0.085, thigh=0.20, shank=0.20,
                 stand_height=0.30):
        self.hip_x = float(hip_x)
        self.hip_y = float(hip_y)
        self.thigh = float(thigh)
        self.shank = float(shank)
        self.stand_height = float(stand_height)
        self.hip = {name: np.array([sx * hip_x, sy * hip_y, 0.0])
                    for name, (sx, sy) in HIP_SIGN.items()}

    # ------------------------------------------------------------------ limits
    @property
    def reach_min(self):
        return abs(self.thigh - self.shank) * 1.0

    @property
    def reach_max(self):
        return self.thigh + self.shank

    def nominal_foot(self, name):
        """Standing foot position in the body frame (directly below the hip)."""
        return self.hip[name] + np.array([0.0, 0.0, -self.stand_height])

    # ------------------------------------------------------------- forward kin
    def leg_fk(self, q):
        """Joint angles (gamma, alpha, beta) -> foot offset from the hip."""
        gamma, alpha, beta = np.asarray(q, dtype=np.float64)
        u = self.thigh * np.sin(alpha) + self.shank * np.sin(alpha + beta)
        w = self.thigh * np.cos(alpha) + self.shank * np.cos(alpha + beta)
        return np.array([u, w * np.sin(gamma), -w * np.cos(gamma)])

    def foot_body(self, name, q):
        """Foot position in the body frame."""
        return self.hip[name] + self.leg_fk(q)

    def feet_body(self, joints):
        """All four feet at once; joints is {name: q} or a (4, 3) array."""
        if isinstance(joints, dict):
            return {n: self.foot_body(n, joints[n]) for n in LEG_NAMES}
        return {n: self.foot_body(n, joints[i]) for i, n in enumerate(LEG_NAMES)}

    # ------------------------------------------------------------- inverse kin
    def leg_ik(self, foot_offset, knee_backward=True):
        """Foot offset from the hip -> (gamma, alpha, beta).

        Raises ValueError when the target is outside the reachable annulus;
        callers that command slightly over-extended targets should clamp with
        `clamp_reachable` first.
        """
        u, dy, dz = np.asarray(foot_offset, dtype=np.float64)
        w = float(np.hypot(dy, dz))
        gamma = float(np.arctan2(dy, -dz)) if w > 1e-12 else 0.0
        d = float(np.hypot(u, w))
        if d > self.reach_max + 1e-9 or d < self.reach_min - 1e-9:
            raise ValueError("foot target %.4f m outside [%.3f, %.3f]"
                             % (d, self.reach_min, self.reach_max))
        c = (d * d - self.thigh ** 2 - self.shank ** 2) / (2.0 * self.thigh * self.shank)
        c = float(np.clip(c, -1.0, 1.0))
        beta = -np.arccos(c) if knee_backward else np.arccos(c)
        alpha = (np.arctan2(u, w)
                 - np.arctan2(self.shank * np.sin(beta),
                              self.thigh + self.shank * np.cos(beta)))
        return np.array([gamma, alpha, beta])

    def foot_ik(self, name, foot_body_pos, knee_backward=True):
        """Body-frame foot position -> joint angles for one leg."""
        return self.leg_ik(np.asarray(foot_body_pos) - self.hip[name],
                           knee_backward=knee_backward)

    def feet_ik(self, targets, knee_backward=True):
        """{name: body-frame foot position} -> {name: joints}."""
        return {n: self.foot_ik(n, p, knee_backward) for n, p in targets.items()}

    def clamp_reachable(self, foot_offset, margin=0.99):
        """Scale a foot offset so it lies just inside the reachable sphere."""
        v = np.asarray(foot_offset, dtype=np.float64)
        d = float(np.linalg.norm(v))
        limit = self.reach_max * margin
        if d > limit:
            return v * (limit / d)
        return v

    # ------------------------------------------------------------------ jacobi
    def leg_jacobian(self, q):
        """d(foot offset)/d(gamma, alpha, beta), 3x3."""
        gamma, alpha, beta = np.asarray(q, dtype=np.float64)
        du_da = self.thigh * np.cos(alpha) + self.shank * np.cos(alpha + beta)
        du_db = self.shank * np.cos(alpha + beta)
        w = self.thigh * np.cos(alpha) + self.shank * np.cos(alpha + beta)
        dw_da = -self.thigh * np.sin(alpha) - self.shank * np.sin(alpha + beta)
        dw_db = -self.shank * np.sin(alpha + beta)
        sg, cg = np.sin(gamma), np.cos(gamma)
        J = np.zeros((3, 3))
        J[0, 1], J[0, 2] = du_da, du_db
        J[1, 0] = w * cg
        J[1, 1], J[1, 2] = dw_da * sg, dw_db * sg
        J[2, 0] = w * sg
        J[2, 1], J[2, 2] = -dw_da * cg, -dw_db * cg
        return J

    # --------------------------------------------------------------- utilities
    def default_pose(self, height=None):
        """{name: joints} for a symmetric standing pose."""
        h = self.stand_height if height is None else height
        out = {}
        for name in LEG_NAMES:
            out[name] = self.leg_ik(np.array([0.0, 0.0, -h]))
        return out

    def foot_heights(self, joints):
        return {n: float(self.foot_body(n, joints[n])[2]) for n in LEG_NAMES}


def rotate_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotate_xyz(rpy):
    """Intrinsic x-y-z rotation matrix."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def body_to_world(points_body, position, rpy):
    """Map body-frame points into the world frame."""
    R = rotate_xyz(rpy)
    return np.asarray(points_body) @ R.T + np.asarray(position)


def world_to_body(points_world, position, rpy):
    R = rotate_xyz(rpy)
    return (np.asarray(points_world) - np.asarray(position)) @ R
