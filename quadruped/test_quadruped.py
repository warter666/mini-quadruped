"""Self-contained tests for the quadruped stack.

    python -m quadruped.test_quadruped
"""

import numpy as np

from .controller import KinematicSimulator, TrotController, solve_body_pose
from .gait import (BOUND_OFFSETS, CRAWL_OFFSETS, PACE_OFFSETS, TROT_OFFSETS,
                   Gait, foot_cycle, foot_target, lift_profile,
                   raibert_foothold, tangential_offset, turn_cycle,
                   yaw_travel)
from .kinematics import LEG_NAMES, QuadrupedModel, rotate_z
from .stabilizer import (chebyshev_center, convex_hull, distribute_gravity,
                         point_in_convex,
                         project_friction_cone, solve_forces, stability_margin,
                         support_polygon, zmp)

MODEL = QuadrupedModel(hip_x=0.19, hip_y=0.085, thigh=0.20, shank=0.20)


# ---------------------------------------------------------------- kinematics

def test_fk_ik_roundtrip():
    rng = np.random.default_rng(0)
    err_q = err_p = 0.0
    for _ in range(3000):
        q = np.array([rng.uniform(-0.4, 0.4), rng.uniform(0.2, 1.2),
                      rng.uniform(-2.2, -0.4)])
        p = MODEL.leg_fk(q)
        q2 = MODEL.leg_ik(p)
        err_q = max(err_q, float(np.abs(q - q2).max()))
        err_p = max(err_p, float(np.abs(MODEL.leg_fk(q2) - p).max()))
    assert err_q < 1e-9, err_q
    assert err_p < 1e-9, err_p
    print("  FK/IK roundtrip              angle err %.2e  position err %.2e"
          % (err_q, err_p))


def test_all_legs_nominal_pose():
    for name in LEG_NAMES:
        target = MODEL.nominal_foot(name)
        q = MODEL.foot_ik(name, target)
        back = MODEL.foot_body(name, q)
        assert np.allclose(back, target, atol=1e-12), (name, back, target)
    print("  nominal stance               all 4 legs invert exactly")


def test_jacobian_matches_numeric():
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(20):
        q = np.array([rng.uniform(-0.3, 0.3), rng.uniform(0.4, 1.1),
                      rng.uniform(-1.9, -0.6)])
        J = MODEL.leg_jacobian(q)
        num = np.zeros((3, 3))
        for i in range(3):
            dq = np.zeros(3)
            dq[i] = 1e-7
            num[:, i] = (MODEL.leg_fk(q + dq) - MODEL.leg_fk(q - dq)) / 2e-7
        worst = max(worst, float(np.abs(J - num).max()))
    assert worst < 1e-6, worst
    print("  leg Jacobian vs finite diff  max abs err %.2e" % worst)


def test_unreachable_target_raises():
    try:
        MODEL.leg_ik(np.array([0.0, 0.0, -0.95]))
    except ValueError:
        print("  reachability guard           over-extension rejected")
        return
    raise AssertionError("over-extended target was accepted")


# ---------------------------------------------------------------------- gait

def test_foot_cycle_conserves_stride():
    duty, stride, lift = 0.5, 0.12, 0.05
    # The cycle must be continuous at the phase wrap (1 -> 0).
    x_end, z_end = foot_cycle(0.9999, duty, stride, lift)
    x_start, z_start = foot_cycle(0.0001, duty, stride, lift)
    assert abs(z_end - z_start) < 1e-3
    assert abs(x_end - x_start) < 1e-3
    # Stance slides exactly one stride backwards, swing returns it forwards.
    x_td, _ = foot_cycle(0.0, duty, stride, lift)
    x_lo, _ = foot_cycle(duty - 1e-9, duty, stride, lift)
    assert abs((x_td - x_lo) - stride) < 1e-6, (x_td, x_lo)
    z_peak = max(foot_cycle(p / 200.0, duty, stride, lift)[1]
                 for p in range(200))
    assert abs(z_peak - lift) < 1e-3, z_peak
    print("  foot_cycle                   stride %.3f m conserved, lift %.3f m "
          "peak, continuous at wrap" % (stride, z_peak))


def test_foot_cycle_matched_velocity():
    """Swing must leave/enter stance with the same relative velocity."""
    duty, speed, period = 0.5, 0.5, 0.4
    stride = speed * duty * period
    eps = 1e-6

    def dx(phase):
        return foot_cycle(phase, duty, stride, 0.05, speed=speed)[0]

    v_stance = (dx(eps) - dx(0.0)) / eps
    v_swing_start = (dx(duty + eps) - dx(duty)) / eps
    rel_stance = v_stance / period          # convert d/dphase -> m/s
    rel_swing = v_swing_start / period
    assert abs(rel_stance - rel_swing) < 1e-3, (rel_stance, rel_swing)
    print("  swing velocity matching      stance %.4f m/s vs swing %.4f m/s"
          % (rel_stance, rel_swing))


def test_raibert_sign():
    nominal = 0.5 * 0.4 * 0.2
    fast = raibert_foothold(v_cmd=0.4, v_meas=0.6, stance_time=0.2, gain=0.1)
    slow = raibert_foothold(v_cmd=0.4, v_meas=0.2, stance_time=0.2, gain=0.1)
    # Overspeed must plant the foot further forward (which brakes), underspeed
    # further back (which accelerates) -- Raibert's stabilising sign.
    assert fast > nominal > slow, (fast, nominal, slow)
    assert abs(fast - (nominal + 0.02)) < 1e-12
    print("  Raibert foothold             overspeed %.4f m, nominal %.4f m, "
          "underspeed %.4f m" % (fast, nominal, slow))


def test_turning_is_tangential():
    """Three independent facts about yaw.

    First, an infinitesimal body yaw d(theta) moves a foot at r by d(theta) x r
    in the body frame -- i.e. -d(theta) z_hat x r, the tangential direction. A
    purely radial or constant offset would be wrong.

    Second, turn_cycle must be a *rate* profile, not an offset: it has to sweep
    a nonzero angle across the stance, otherwise a planted foot is silently
    dragged around and the body never actually turns.

    Third, it must be zero-based at touchdown. The body yaw a planted leg
    implies is yaw_touchdown + theta, and that has to come out as the same
    yaw(t) for every planted leg. Only theta(phi) = wz*phi*T does that when the
    planted legs sit at different phases.
    """
    r = np.array([0.19, 0.085])
    assert np.allclose(tangential_offset(0.0, r), 0.0)
    d = 1e-6
    exact = np.array([[np.cos(d), np.sin(d)], [-np.sin(d), np.cos(d)]]) @ r - r
    assert np.abs(exact - tangential_offset(d, r)).max() < 1e-11

    duty, sweep = 0.5, 0.1
    assert turn_cycle(0.0, duty, sweep) == 0.0
    assert abs(turn_cycle(duty - 1e-9, duty, sweep) - sweep) < 1e-6
    assert abs(turn_cycle(1.0 - 1e-9, duty, sweep)) < 1e-4
    # Linear in phase while planted => a constant counter-rotation rate of
    # sweep/stance_time = wz, for every leg, whatever its phase offset.
    for f in (0.05, 0.1, 0.25, 0.5, 0.9, 0.999):
        assert abs(turn_cycle(f * duty, duty, sweep) - f * sweep) < 1e-12
    assert turn_cycle(0.7, duty, 0.0) == 0.0
    print("  turning kinematics           d(theta) x r matched to 1e-11; "
          "turn_cycle linear 0 -> %.2f rad over the stance, back to 0 in swing"
          % sweep)


def test_turn_is_gait_agnostic():
    """The realised yaw rate must equal the command for every gait and duty.

    This is the regression guard for the zero-based turn_cycle. A symmetric
    sweep gives the right answer for trot/pace/bound -- their planted legs all
    share a phase -- and quietly lags on a crawl, where three planted legs sit
    at three different phases and disagree about the body yaw by
    (theta_i - wz*phi_i*T).
    """
    worst_rate, worst_gait = 0.0, ""
    for name, offs in (("trot", TROT_OFFSETS), ("pace", PACE_OFFSETS),
                       ("bound", BOUND_OFFSETS), ("crawl", CRAWL_OFFSETS)):
        for duty in (0.5, 0.75):
            gait = Gait(period=0.5, duty=duty, offsets=offs)
            controller = TrotController(MODEL, gait, height=0.26, lift=0.05)
            sim = KinematicSimulator(MODEL, controller, dt=0.002)
            sim.run({"wz": 0.4}, 6.0)
            ys = [(t, y) for t, _, y, _ in sim.log if 2.0 <= t <= 6.0]
            rate = (ys[-1][1] - ys[0][1]) / (ys[-1][0] - ys[0][0])
            err = abs(np.rad2deg(rate - 0.4))
            if err > worst_rate:
                worst_rate, worst_gait = err, "%s duty=%.2f" % (name, duty)
    assert worst_rate < 0.01, (worst_rate, worst_gait)
    print("  turn, all 4 gaits x 2 duties  worst rate error %.4f deg/s (%s)"
          % (worst_rate, worst_gait))


def test_planted_foot_is_stationary_in_the_world():
    """The stance trajectory must hold a planted foot perfectly still.

    Independent reference: with a constant body-frame velocity v and yaw rate
    w, the body origin follows p(t) = int_0^t R(w s) v ds and yaw(t) = w t, so
    a foot fixed at world P appears at R(-yaw)(P - p) in the body frame. The
    controller has to reproduce that from the command alone.

    Dropping the "measured in a turning frame" part -- writing the stride as a
    straight vector in the current body frame, R(-a) r + dir*dx -- misses it by
    wz*|v|*t^2/2. At wz = 0.5 and |v| = 0.33 that is ~2.7 mm by the end of a
    0.2 s stance, re-injected every step.
    """
    v = np.array([0.31, -0.12])
    w = 0.5
    duty, period, lift = 0.5, 0.4, 0.05
    stance_time, swing_time = duty * period, (1.0 - duty) * period
    r_nom = np.array([0.19, 0.085])
    r_td = r_nom + v * (stance_time / 2.0)       # landed half a stance ahead

    worst, worst_naive = 0.0, 0.0
    for k in range(0, 200):
        t = stance_time * k / 200.0
        a = w * t
        p = yaw_travel(t, w) @ v                 # body origin, world path
        exact = rotate_z(-a)[:2, :2] @ (r_td - p)

        phi = duty * (t / stance_time)
        x, y, _ = foot_target(phi, duty, stance_time, swing_time, v, w, lift,
                              r_td)
        # the old composition, for scale: rotate the nominal, then add the
        # stride as a straight vector in the current body frame
        naive = rotate_z(-a)[:2, :2] @ r_nom + v * (stance_time / 2.0 - t)
        worst = max(worst, float(np.abs(np.array([x, y]) - exact).max()))
        worst_naive = max(worst_naive, float(np.abs(naive - exact).max()))

    assert worst < 1e-12, worst
    # And confirm the fix is not cosmetic: the old composition was off by a
    # fraction of a millimetre here, ~1e13 times worse than the new one.
    assert worst_naive > 1e-4, worst_naive
    print("  planted foot stays put       exact to %.1e m over a stance "
          "(old composition was off by %.1e m)" % (worst, worst_naive))


def test_foot_target_reduces_to_foot_cycle():
    """foot_cycle() is how the gait is *specified* (a scalar profile along the
    body x axis); foot_target() is how it is *composed* (exact under yaw). They
    must agree identically at zero yaw rate, otherwise one of them has drifted
    and the specification no longer describes the implementation.
    """
    duty, period, speed, lift = 0.5, 0.4, 0.5, 0.05
    stance_time, swing_time = duty * period, (1.0 - duty) * period
    stride = speed * stance_time
    r_nom = np.array([0.19, 0.085])
    r_td = r_nom + np.array([stride / 2.0, 0.0])
    worst = 0.0
    for k in range(400):
        phi = k / 400.0
        x, y, dz = foot_target(phi, duty, stance_time, swing_time,
                               np.array([speed, 0.0]), 0.0, lift, r_td)
        dx, dz_1d = foot_cycle(phi, duty, stride, lift, speed=speed)
        worst = max(worst, abs(x - (r_nom[0] + dx)), abs(y - r_nom[1]),
                    abs(dz - dz_1d))
    assert worst < 1e-12, worst
    print("  foot_target == foot_cycle    agreement at wz=0 to %.1e over the "
          "whole cycle" % worst)


# --------------------------------------------------------------- stability

def test_convex_hull_and_containment():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
                       [0.5, 0.5]])
    hull = convex_hull(square)
    assert len(hull) == 4, hull
    assert point_in_convex(hull, (0.5, 0.5))
    assert not point_in_convex(hull, (1.5, 0.5))
    m_in = stability_margin(hull, (0.5, 0.5))
    m_out = stability_margin(hull, (1.5, 0.5))
    assert abs(m_in - 0.5) < 1e-9, m_in
    assert m_out < 0, m_out
    print("  hull / containment           margin inside %.3f, outside %.3f"
          % (m_in, m_out))


def test_trot_is_not_statically_stable():
    """Diagonal stance is a degenerate, zero-area support polygon: trot has to
    be stabilised dynamically, which is why it needs a controller at all."""
    diag = np.array([[0.19, 0.085], [-0.19, -0.085]])
    poly = support_polygon(diag)
    assert len(poly) == 2, poly
    assert stability_margin(poly, [0.0, 0.0]) == -float("inf")
    # Nudge the CoM projection sideways and it leaves the support line at once.
    assert not point_in_convex(poly, (0.0, 0.02))
    print("  trot diagonal stance         2-point (zero-area) polygon, no "
          "static margin -> dynamic stabilisation required")


def test_crawl_three_foot_support():
    """A 3-foot support triangle does exist for a crawl, but the nominal CoM
    (body origin) sits exactly *on* its diagonal -- zero margin. That is why a
    static crawl needs a body shift: the CoM has to be moved to the deepest
    interior point of the triangle before a leg is lifted.
    """
    gait = Gait(period=0.6, duty=0.75, offsets=CRAWL_OFFSETS)
    worst_nominal, worst_center = float("inf"), float("inf")
    n_down_min, n_samples = 4, 0
    for t in np.arange(0.0, gait.period, gait.period / 120.0):
        phases = gait.phases(t)
        down = [n for n in LEG_NAMES if phases[n] < gait.duty]
        n_down_min = min(n_down_min, len(down))
        if len(down) != 3:
            continue
        n_samples += 1
        poly = support_polygon([MODEL.nominal_foot(n)[:2] for n in down])
        worst_nominal = min(worst_nominal, stability_margin(poly, [0.0, 0.0]))
        _, best = chebyshev_center(poly)
        worst_center = min(worst_center, best)
    assert n_down_min >= 3, n_down_min
    assert n_samples > 0
    assert abs(worst_nominal) < 1e-9, worst_nominal
    assert worst_center > 0.015, worst_center
    print("  crawl 3-foot support         min %d down; margin at nominal CoM "
          "%.4f m (marginal) vs %.4f m after a body shift"
          % (n_down_min, worst_nominal, worst_center))


def test_zmp_shifts_against_acceleration():
    com = np.array([0.0, 0.0])
    acc = np.array([1.0, 0.0])
    z = zmp(com, acc, height=0.26)
    assert z[0] < 0, z
    assert abs(z[0] + 0.26 / 9.81) < 1e-9
    static = zmp(com, np.zeros(2), height=0.26)
    assert np.allclose(static, com)
    print("  ZMP                          a=+1 m/s^2 shifts it %.4f m backwards"
          % z[0])


# ------------------------------------------------------------------ forces

def test_force_allocation_satisfies_wrench():
    contacts = np.array([[0.19, 0.085, 0.0], [0.19, -0.085, 0.0],
                         [-0.19, 0.085, 0.0], [-0.19, -0.085, 0.0]])
    com = np.array([0.0, 0.0, 0.25])
    mass = 8.0
    force, torque = np.array([0.0, 0.0, mass * 9.81]), np.zeros(3)
    f = solve_forces(contacts, com, force, torque)
    assert np.allclose(f.sum(0), force, atol=1e-9), f.sum(0)
    r = contacts - com
    m = np.cross(r, f).sum(0)
    assert np.allclose(m, torque, atol=1e-9), m
    # A symmetric body must load all four feet equally.
    assert np.allclose(f[:, 2], mass * 9.81 / 4.0, atol=1e-9), f[:, 2]
    print("  force allocation             residual %.2e, fz per foot %.2f N"
          % (np.abs(np.cross(r, f).sum(0) - torque).max(), f[0, 2]))


def test_friction_cone_projection():
    f = np.array([[10.0, 0.0, 5.0], [3.0, 4.0, 10.0]])
    mu = 0.6
    g = project_friction_cone(f, mu)
    for row in g:
        assert np.hypot(row[0], row[1]) <= mu * row[2] + 1e-12, row
    assert abs(np.hypot(g[0, 0], g[0, 1]) - mu * 5.0) < 1e-12
    assert np.allclose(g[1], f[1])          # already inside the cone
    print("  friction cone                clipped to mu*fz, interior force "
          "untouched")


def test_gravity_distribution_and_friction_limit():
    """Split a pure gravity load, then a load that needs the cone.

    The symmetric case is the sanity check; the interesting one is a CoM offset
    forward, where the moment balance forces the front feet to carry more, and
    a lateral shove, where the minimum-norm split leaves the cone and has to be
    clipped. Both were uncovered code until this test existed.
    """
    contacts = np.array([[0.19, 0.085, 0.0], [0.19, -0.085, 0.0],
                         [-0.19, 0.085, 0.0], [-0.19, -0.085, 0.0]])
    mass = 8.0
    weight = mass * 9.81

    f = distribute_gravity(contacts, np.array([0.0, 0.0, 0.25]), mass)
    assert np.allclose(f[:, 2], weight / 4.0, atol=1e-9), f[:, 2]

    # CoM 10 cm forward: r_x is +0.09 for the front pair and -0.29 for the rear,
    # so the front pair must carry the larger share.
    f = distribute_gravity(contacts, np.array([0.10, 0.0, 0.25]), mass)
    assert np.allclose(f.sum(0), [0.0, 0.0, weight], atol=1e-9), f.sum(0)
    front, rear = f[0, 2] + f[1, 2], f[2, 2] + f[3, 2]
    assert front > rear, (front, rear)
    assert abs(front / rear - 0.29 / 0.09) < 1e-6, (front, rear)

    # A 50 N forward shove at CoM height 0.25 m. That is a 12.5 N.m pitching
    # moment, and only an uneven vertical split can cancel it -- the min-norm
    # solution moves the load to the *rear* pair (the feet push the ground
    # forward, so the reaction tips the body backwards). The front pair ends up
    # at 3.2 N, under a 1.9 N shear limit, while each foot wants 12.5 N.
    force = np.array([50.0, 0.0, weight])
    raw = solve_forces(contacts, np.array([0.0, 0.0, 0.25]), force, np.zeros(3))
    assert np.allclose(raw.sum(0), force, atol=1e-9), raw.sum(0)
    front_z, rear_z = raw[0, 2] + raw[1, 2], raw[2, 2] + raw[3, 2]
    assert rear_z > front_z, (front_z, rear_z)

    shear = np.hypot(raw[:, 0], raw[:, 1])
    limit = 0.6 * raw[:, 2]
    over = shear > limit
    # Only the unloaded front pair escapes the cone; the heavily loaded rear
    # pair has plenty of friction headroom and keeps its full 12.5 N.
    assert over.tolist() == [True, True, False, False], (shear, limit)
    clipped = project_friction_cone(raw, 0.6)
    for row in clipped:
        assert np.hypot(row[0], row[1]) <= 0.6 * row[2] + 1e-12, row
    assert np.allclose(clipped.sum(0)[2], weight, atol=1e-9), clipped.sum(0)
    # The projection leaves the vertical wrench intact but *not* the shear: it
    # clips each foot in place rather than re-solving with the cone as a
    # constraint. Asserting the loss keeps that honest instead of pretending
    # the allocation is still feasible.
    lost = float((shear - np.hypot(clipped[:, 0], clipped[:, 1])).sum())
    assert abs(lost - (2 * (12.5 - 0.6 * front_z / 2.0))) < 1e-9, lost
    print("  gravity distribution         symmetric %.2f N/foot; CoM 10 cm "
          "forward moves %.1f%% of the load to the front pair"
          % (weight / 4.0, 100.0 * (front - rear) / weight))
    print("  friction limit               50 N shove: rear pair carries "
          "%.1f N vs front %.1f N, front shear clipped 12.5 -> %.1f N "
          "(%.1f N of shear dropped, not re-solved)"
          % (rear_z, front_z, 0.6 * front_z / 2.0, lost))


# --------------------------------------------------------------- simulation

def _steady_velocity(sim, t_from, t_to):
    xs = [(t, p[0]) for t, p, _, _ in sim.log if t_from <= t <= t_to]
    return (xs[-1][1] - xs[0][1]) / (xs[-1][0] - xs[0][0])


def _arc_speed(sim, t_from, t_to):
    """Arc length per unit time.

    With a yaw command the world x axis is no longer the direction of travel --
    the body traces a circle -- so measuring dx/dt reports near zero even when
    the legs are doing their job. The translation speed is the arc speed.
    """
    rows = [r for r in sim.log if t_from <= r[0] <= t_to]
    p = np.array([r[1] for r in rows])
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
                 / (rows[-1][0] - rows[0][0]))


def test_contact_schedule_is_consistent():
    """No foot may be lifted while it is still acting as a ground constraint.

    The launch invariant is that the contact set is keyed on the commanded
    ground clearance (dz), not on the phase boolean phase < duty. The two
    disagree for exactly one tick at every liftoff, because the sampled phase
    lands a hair past duty on the far side of a floating-point boundary. The
    observable consequence is a one-way height ratchet, so this test watches
    the height on a run long enough for the tick clock to drift into the bad
    case (it first appeared at t ~ 2.2 s, not at t ~ 0.4 s).

    An earlier version leaked lift_profile(dt/swing_time) = 8.6e-5 m at every
    liftoff: 0.5 mm/s of sink, unbounded, invisible to a 4 s test.
    """
    controller = TrotController(MODEL, height=0.26)
    sim = KinematicSimulator(MODEL, controller, dt=0.002)
    sim.run({"vx": 0.35}, 20.0)
    zs = np.array([p[2] for _, p, _, _ in sim.log])
    worst = float(np.abs(zs - 0.26).max())
    assert worst < 1e-12, worst
    assert abs(np.rad2deg(sim.yaw)) < 1e-6, np.rad2deg(sim.yaw)
    print("  contact scheduling           20 s run, height deviation %.2e m "
          "(no ratchet), yaw drift %.2e deg"
          % (worst, np.rad2deg(sim.yaw)))


def test_trot_tracks_forward_speed():
    controller = TrotController(MODEL, Gait(period=0.4, duty=0.5),
                                height=0.26, lift=0.055)
    sim = KinematicSimulator(MODEL, controller, dt=0.002).run({"vx": 0.4}, 4.0)
    v = _steady_velocity(sim, 1.0, 4.0)
    zs = [p[2] for _, p, _, _ in sim.log]
    # Tolerance is 1e-4, not 1e-2: the surviving error here is trapezoidal
    # sampling noise, nothing more. A 1e-2 window silently hid a 1% deficit
    # caused by two frozen ticks per period at the stance/swing handover.
    assert abs(v - 0.4) < 1e-4, v
    assert abs(np.mean(zs) - 0.26) < 1e-12, np.mean(zs)
    assert abs(np.rad2deg(sim.yaw)) < 1e-6, np.rad2deg(sim.yaw)
    print("  trot speed tracking          v=%.6f m/s (cmd 0.4), height %.6f m, "
          "yaw drift %.2e deg" % (v, np.mean(zs), np.rad2deg(sim.yaw)))


def test_trot_backwards_and_lateral():
    controller = TrotController(MODEL, height=0.26)
    sim = KinematicSimulator(MODEL, controller, dt=0.002).run({"vx": -0.3}, 3.0)
    v = _steady_velocity(sim, 1.0, 3.0)
    assert abs(v + 0.3) < 1e-4, v
    # Pure lateral: the same gait, rotated 90 degrees in the body frame.
    sim2 = KinematicSimulator(MODEL, controller, dt=0.002).run({"vy": 0.25}, 3.0)
    ys = [(t, p[1]) for t, p, _, _ in sim2.log if 1.0 <= t <= 3.0]
    vy = (ys[-1][1] - ys[0][1]) / (ys[-1][0] - ys[0][0])
    xs = [p[0] for _, p, _, _ in sim2.log]
    assert abs(vy - 0.25) < 1e-4, vy
    assert abs(max(xs) - min(xs)) < 1e-9, max(xs) - min(xs)
    print("  reverse / lateral trot       vx=%.6f (cmd -0.3), vy=%.6f "
          "(cmd 0.25), crosstalk %.2e m" % (v, vy, max(xs) - min(xs)))


def test_turn_in_place():
    controller = TrotController(MODEL, height=0.26)
    sim = KinematicSimulator(MODEL, controller, dt=0.002)
    sim.run({"wz": 0.5}, 3.0)
    target = np.rad2deg(1.5)
    err = abs(np.rad2deg(sim.yaw) - target)
    # Half a degree, not fifteen: the only residual is the one-tick anchor
    # delay. A 15 deg window let a 3.4% turn-rate error through, caused by
    # reset() anchoring the feet at the neutral pose while the t = 0 plan had
    # already rotated them by turn_cycle(0) = -half.
    assert err < 0.5, (np.rad2deg(sim.yaw), target)
    assert np.linalg.norm(sim.pose[:2]) < 1e-9, sim.pose[:2]
    # Steady-state rate must be exact, not just close on average: the whole
    # point of turn_cycle is that 2*half/stance_time == wz identically.
    ys = [(t, y) for t, _, y, _ in sim.log if 1.0 <= t <= 3.0]
    rate = (ys[-1][1] - ys[0][1]) / (ys[-1][0] - ys[0][0])
    assert abs(rate - 0.5) < 1e-4, rate
    print("  turn in place                yaw %.2f deg over 3 s (cmd 0.5 rad/s "
          "-> %.2f deg), steady rate %.6f rad/s, position drift %.1e m"
          % (np.rad2deg(sim.yaw), target, rate, np.linalg.norm(sim.pose[:2])))


def test_drive_and_turn_together():
    """Forward speed and yaw rate must be independent of each other.

    They share one channel -- the foot target -- so a coupled implementation
    loses both at once. The failure this guards against was severe: adding a
    yaw command collapsed the forward speed from 0.30 to 0.019 m/s, because the
    tangential term and the stride term were not composed in the right frame.
    """
    controller = TrotController(MODEL, height=0.26)
    sim = KinematicSimulator(MODEL, controller, dt=0.002)
    sim.run({"vx": 0.3, "wz": 0.4}, 8.0)
    v = _arc_speed(sim, 2.0, 8.0)
    ys = [(t, y) for t, _, y, _ in sim.log if 2.0 <= t <= 8.0]
    rate = (ys[-1][1] - ys[0][1]) / (ys[-1][0] - ys[0][0])
    zs = np.array([p[2] for t, p, _, _ in sim.log if t >= 1.0])
    assert abs(v - 0.3) < 1e-3, v
    assert abs(rate - 0.4) < 1e-3, rate
    assert np.abs(zs - 0.26).max() < 1e-12, np.abs(zs - 0.26).max()
    print("  drive + turn together        arc speed %.6f (cmd 0.3), yaw rate "
          "%.6f (cmd 0.4), height deviation %.1e m"
          % (v, rate, np.abs(zs - 0.26).max()))


def test_odometry_solve_is_exact():
    """The pose solve must invert the forward transform exactly."""
    rng = np.random.default_rng(3)
    pts_body = rng.normal(0.0, 0.15, (4, 3)) + np.array([0.19, 0.0, -0.26])
    pos = np.array([1.2, -0.4, 0.31])
    yaw = 0.42
    from .kinematics import rotate_z
    pts_world = pts_body @ rotate_z(yaw).T + pos
    est_pos, est_yaw = solve_body_pose(pts_body, pts_world)
    assert np.allclose(est_pos, pos, atol=1e-12), (est_pos, pos)
    assert abs(est_yaw - yaw) < 1e-12, (est_yaw, yaw)
    print("  odometry solve               pose recovered to %.2e m / %.2e rad"
          % (np.abs(est_pos - pos).max(), abs(est_yaw - yaw)))


def main():
    print("quadruped tests")
    print("[kinematics]")
    test_fk_ik_roundtrip()
    test_all_legs_nominal_pose()
    test_jacobian_matches_numeric()
    test_unreachable_target_raises()
    print("[gait]")
    test_foot_cycle_conserves_stride()
    test_foot_cycle_matched_velocity()
    test_raibert_sign()
    test_turning_is_tangential()
    test_turn_is_gait_agnostic()
    test_planted_foot_is_stationary_in_the_world()
    test_foot_target_reduces_to_foot_cycle()
    print("[stability]")
    test_convex_hull_and_containment()
    test_trot_is_not_statically_stable()
    test_crawl_three_foot_support()
    test_zmp_shifts_against_acceleration()
    print("[force allocation]")
    test_force_allocation_satisfies_wrench()
    test_friction_cone_projection()
    test_gravity_distribution_and_friction_limit()
    print("[simulation]")
    test_odometry_solve_is_exact()
    test_contact_schedule_is_consistent()
    test_trot_tracks_forward_speed()
    test_trot_backwards_and_lateral()
    test_turn_in_place()
    test_drive_and_turn_together()
    print("all quadruped tests passed")


if __name__ == "__main__":
    main()
