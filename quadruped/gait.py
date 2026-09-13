"""Gait scheduling and swing/stance foot trajectories, pure numpy.

The stride bookkeeping is worth spelling out, because it is where most
quadruped implementations get the sign wrong.

    T          gait period,  duty = fraction of T spent in stance
    v          forward speed of the body
    stride     relative displacement of the foot during stance = v * duty * T

During stance the foot is glued to the world, so it slides backwards through
the body by exactly `v * duty * T`. During swing the foot must recover that
plus whatever ground the body covered meanwhile, i.e. the foot's world
position advances by v * T per cycle. Both phases are expressed below as
offsets from the standing foot position in the body frame.
"""

import numpy as np

from .kinematics import LEG_NAMES, rotate_z

# Diagonal pairs move together: this is what makes trot stable at speed.
TROT_OFFSETS = {"FL": 0.0, "FR": 0.5, "RL": 0.5, "RR": 0.0}
# Four-beat crawl: always at least three feet down, statically stable.
CRAWL_OFFSETS = {"FL": 0.0, "RR": 0.25, "FR": 0.5, "RL": 0.75}
PACE_OFFSETS = {"FL": 0.0, "FR": 0.5, "RL": 0.0, "RR": 0.5}
BOUND_OFFSETS = {"FL": 0.0, "FR": 0.0, "RL": 0.5, "RR": 0.5}


class Gait:
    """Phase generator: which legs are down, and how far along they are."""

    def __init__(self, period=0.40, duty=0.5, offsets=None):
        self.period = float(period)
        self.duty = float(duty)
        self.offsets = dict(offsets or TROT_OFFSETS)

    def phases(self, t):
        """{leg: phase in [0, 1)}, phase 0 = touchdown."""
        base = (float(t) / self.period) % 1.0
        return {n: (base + self.offsets[n]) % 1.0 for n in LEG_NAMES}

    def stance(self, phase):
        return phase < self.duty

    def stride(self, speed):
        """Relative foot travel during stance."""
        return float(speed) * self.duty * self.period

    def swing_time(self):
        return (1.0 - self.duty) * self.period

    def stance_time(self):
        return self.duty * self.period


# ------------------------------------------------------------------ profiles

def _quintic(p0, p1, v0, v1, tau):
    """Hermite quintic with matched endpoint velocities and zero acceleration."""
    t2, t3, t4, t5 = tau ** 2, tau ** 3, tau ** 4, tau ** 5
    h0 = 1.0 - 10.0 * t3 + 15.0 * t4 - 6.0 * t5
    h1 = tau - 6.0 * t3 + 8.0 * t4 - 3.0 * t5
    h2 = 0.5 * t2 - 1.5 * t3 + 1.5 * t4 - 0.5 * t5
    h3 = 10.0 * t3 - 15.0 * t4 + 6.0 * t5
    h4 = -4.0 * t3 + 7.0 * t4 - 3.0 * t5
    h5 = 0.5 * t2 - t3 + 0.5 * t4
    return h0 * p0 + h1 * v0 + h3 * p1 + h4 * v1


def lift_profile(tau):
    """Smooth bump, zero value and zero slope at both ends."""
    return 16.0 * tau * tau * (1.0 - tau) ** 2


def foot_cycle(phase, duty, stride, lift, speed=0.0, matched_velocity=True):
    """Foot offset from the standing position: returns (dx, dz).

    dx is along the body x axis, dz is vertical (positive = lifted).
    `speed` is the body speed along x, used to match the horizontal foot
    velocity across the stance/swing transition; without it the swing ends
    with a velocity discontinuity, which shows up as a leg jerk at touchdown.
    """
    if stride <= 0.0 and lift <= 0.0:
        return 0.0, 0.0
    half = stride / 2.0
    if phase < duty:
        s = phase / duty
        dx = half - stride * s
        dz = 0.0
    else:
        tau = (phase - duty) / (1.0 - duty)
        if matched_velocity and speed > 0.0 and stride > 0.0:
            # Hermite velocity terms are d/dtau, so scale the physical foot
            # speed by the swing duration to land in the same units.
            swing = stride / (duty * speed) * (1.0 - duty)
            v_hat = -speed * swing
            dx = _quintic(-half, half, v_hat, v_hat, tau)
        else:
            dx = -half + stride * (3.0 * tau ** 2 - 2.0 * tau ** 3)
        dz = lift * lift_profile(tau)
    return float(dx), float(dz)


# ------------------------------------------------------------------ footholds

def raibert_foothold(v_cmd, v_meas, stance_time, gain=0.10, bias=0.0):
    """Raibert's heuristic: land ahead of the hip by half a stance of travel,
    corrected by the current velocity error.

    The velocity term is what turns a fixed open-loop gait into a stabilising
    controller: when the body is running faster than commanded (v_meas > v_cmd)
    the next foothold moves *forward*, so the body tips backwards over the
    stance foot and brakes. The reverse holds when it lags.
    """
    return 0.5 * v_cmd * stance_time + bias + gain * (v_meas - v_cmd)


def turn_cycle(phase, duty, sweep):
    """Body yaw this leg has accumulated since its own touchdown.

    theta = 0 while the foot lands, rising linearly to `sweep` at liftoff, then
    returning smoothly to 0 through the swing so the foot is re-anchored at its
    nominal orientation at the next touchdown. `sweep = wz * stance_time` is
    the yaw the body gains while this foot is planted.

    Being *zero-based* is the whole point, and it is not cosmetic. The body
    yaw implied by a planted leg is yaw_touchdown + theta. Writing
    t - t_touchdown = phi*T and yaw(t - phi*T) = yaw(t) - wz*phi*T, that
    implied yaw equals

        yaw(t) - wz*phi*T + theta(phi)

    so it collapses onto the single value yaw(t) for *every* planted leg only
    if theta(phi) = wz*phi*T exactly. Any constant added to theta -- for
    instance a symmetric [-half, +half] sweep -- cancels only when all planted
    legs share a phase. Trot, pace and bound do; a crawl does not, and the
    three-way disagreement shows up as a least-squares yaw lag (measured:
    1.2 deg of lost turn per second).
    """
    if sweep == 0.0:
        return 0.0
    if phase < duty:
        return sweep * (phase / duty)
    # Match the angular rate across both handovers, exactly as foot_cycle does
    # for translation. The stance turns at d(theta)/d(phi) = sweep/duty, which
    # in swing-time units tau is v = sweep*(1-duty)/duty; without it the rate
    # steps from wz to 0 the instant the foot leaves, and any leg still being
    # used as a constraint one tick into its swing --- which the staggered
    # crawl guarantees --- drags the body yaw down with it.
    v = sweep * (1.0 - duty) / duty
    tau = (phase - duty) / (1.0 - duty)
    return _quintic(sweep, 0.0, v, v, tau)


def tangential_offset(angle, hip_xy):
    """Small-angle counterpart of turn_cycle, useful for sanity checks: the
    arc a foot at `hip_xy` travels when the body yaws by `angle`.
    """
    x, y = float(hip_xy[0]), float(hip_xy[1])
    return float(angle) * np.array([y, -x])


# ------------------------------------------------- exact stance composition

def yaw_travel(t, yaw_rate):
    """G(t) = int_0^t R(yaw_rate * s) ds, the body's own path measured in its
    own turning frame.

    This is the piece almost every hand-rolled quadruped controller leaves out.
    A planted foot is fixed in the *world*, so its body-frame coordinate is

        R(-a) (r_td - G(t) v),      a = yaw_rate * t

    -- the frame rotates, and the body's displacement has to be measured in the
    frame it was accumulated in. Writing it as R(-a) r + dir*dx (a straight
    stride vector in the body frame) is only right at zero yaw rate; the error
    is wz*|v|*t^2/2, i.e. 2.4 mm per stance at wz = 0.4 rad/s, |v| = 0.3 m/s.
    That is not a cosmetic number: the foot is re-anchored every stance, so the
    mistake is re-injected at every step and shows up as diverging pose spikes.

    yaw_rate -> 0 gives t * I, i.e. exactly the naive answer, so this reduces
    to foot_cycle() rather than replacing it.
    """
    t = float(t)
    yaw_rate = float(yaw_rate)
    a = yaw_rate * t
    if abs(a) < 1e-7:
        # Series about a = 0: exact to machine precision there, and it is the
        # only branch that survives yaw_rate = 0 (where a/wz is 0/0).
        k = 1.0 - a * a / 6.0
        h = 0.5 * a * (1.0 - a * a / 12.0)
        return np.array([[t * k, -t * h], [t * h, t * k]])
    s, c = np.sin(a), np.cos(a)
    return np.array([[s, -(1.0 - c)], [1.0 - c, s]]) / yaw_rate


def foot_target(phase, duty, stance_time, swing_time, velocity, yaw_rate, lift,
                r_landed, r_next=None):
    """Body-frame foot target (x, y, dz) for one leg, exact under simultaneous
    translation and yaw.

    `r_landed` is the body-frame position this foot actually touched down at
    (nominal + the lead the swing ended with); `r_next` is where it should
    touch down next, which is where a Raibert correction enters. Both are
    2-vectors in the body frame -- the rotation below is about the body origin,
    so the hip offset must be inside them.

    Serves double duty: while planted it is the exact contact trajectory
    above; during the swing it is a C1 Hermite carrying the foot from where it
    left the ground to `r_next`. Both handovers match velocity, so this is
    continuous in value and in slope at touchdown and liftoff -- the same
    property foot_cycle has, generalised off the body x axis.

    With yaw_rate = 0 this reproduces foot_cycle() exactly.
    """
    r_landed = np.asarray(r_landed, dtype=np.float64).reshape(2)
    r_next = r_landed if r_next is None else \
        np.asarray(r_next, dtype=np.float64).reshape(2)
    velocity = np.asarray(velocity, dtype=np.float64).reshape(2)

    sweep = float(yaw_rate) * float(stance_time)
    angle = turn_cycle(phase, duty, sweep)

    if phase < duty:
        t = float(stance_time) * (phase / duty)
        u = r_landed - yaw_travel(t, yaw_rate) @ velocity
        dz = 0.0
    else:
        tau = (phase - duty) / (1.0 - duty)
        u_lo = r_landed - yaw_travel(stance_time, yaw_rate) @ velocity
        # Both ends must match the stance rate at that instant. The stance
        # moves the foot at -R(a) v in the body frame, so the swing has to
        # leave at -R(sweep) v and arrive at -v, each scaled by the swing
        # duration to land in tau units.
        v_lo = -(rotate_z(sweep)[:2, :2] @ velocity) * float(swing_time)
        v_td = -velocity * float(swing_time)
        u = _quintic(u_lo, r_next, v_lo, v_td, tau)
        dz = float(lift) * lift_profile(tau)

    xy = rotate_z(-angle)[:2, :2] @ u
    return float(xy[0]), float(xy[1]), float(dz)
