"""Whole-body trot controller plus a contact-constrained kinematic simulator.

Why a kinematic simulator instead of a physics engine: the controller's job is
to decide *where the feet should be*, and the only thing that makes the body
move is the contact constraint "a planted foot does not slide". Solving that
constraint directly (2-D Procrustes over the stance feet) is a closed-form,
deterministic and exactly invertible test of the gait logic -- no contact
solver, no tuning, no stochasticity. It deliberately ignores roll/pitch and
ground reaction forces, which is stated as a limitation in the README.

Per step
    1. gait phase -> per-leg foot target in the body frame (stride, lift,
       Raibert foothold correction, turn offset)
    2. inverse kinematics -> joint angles
    3. forward kinematics -> the feet that were actually achieved
    4. the feet planted *during this tick* are pinned to their world
       positions, so the body pose is the rigid transform that best explains
       them; only then is the contact set refreshed
"""

import numpy as np

from .gait import Gait, foot_target, raibert_foothold
from .kinematics import LEG_NAMES, rotate_z

# Ground clearance below which a foot still counts as planted. The gap between
# the two meaningful values is enormous -- a foot whose phase sits 1e-16 past
# touchdown/liftoff commands dz ~1e-32 m, while one tick of swing commands
# dz ~9e-5 m -- so any threshold in between separates them cleanly.
PLANT_EPS = 1e-9


def solve_body_pose(contacts_body, contacts_world):
    """Rigid transform (position, yaw) best mapping body points to world points.

    Closed-form 2-D Procrustes on the xy projection; z follows from the mean
    height difference. Two contact points are enough (4 scalar constraints for
    4 unknowns), three or more make it least squares.
    """
    A = np.asarray(contacts_body, dtype=np.float64)
    B = np.asarray(contacts_world, dtype=np.float64)
    if len(A) < 2:
        raise ValueError("need at least two stance contacts")
    a_c, b_c = A[:, :2].mean(0), B[:, :2].mean(0)
    a, b = A[:, :2] - a_c, B[:, :2] - b_c
    num = float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))
    den = float(np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]))
    yaw = float(np.arctan2(num, den))
    R = rotate_z(yaw)
    pos = np.empty(3)
    pos[:2] = b_c - (R[:2, :2] @ a_c)
    pos[2] = float(np.mean(B[:, 2]) - np.mean((A @ R.T)[:, 2]))
    return pos, yaw


class TrotController:
    """Open-loop gait + Raibert foothold feedback + yaw rate feedforward."""

    def __init__(self, model, gait=None, height=0.26, lift=0.055,
                 raibert_gain=0.09):
        self.model = model
        self.gait = gait or Gait(period=0.40, duty=0.5)
        self.height = float(height)
        self.lift = float(lift)
        self.raibert_gain = float(raibert_gain)
        # Per-leg foothold bookkeeping. A stance trajectory must start from
        # where the foot really touched down; that is not the same as this
        # tick's Raibert-corrected target, which keeps moving while the body
        # is still chasing its commanded speed.
        self._landed_lead = {n: np.zeros(2) for n in LEG_NAMES}
        self._pending_lead = {n: np.zeros(2) for n in LEG_NAMES}
        self._was_stance = {n: False for n in LEG_NAMES}

    def target_height(self, cmd_height=None):
        return self.height if cmd_height is None else float(cmd_height)

    def plan(self, t, cmd, v_meas, state=None):
        """One control tick.

        cmd: dict with vx, vy, wz (body frame), optionally height.
        v_meas: measured (vx, vy) from the odometry solve.
        Returns {leg: dict(target, phase, stance, stride, dz)}.

        Stateful, deliberately: it has to remember the lead each foot landed
        with (see _landed_lead).
        """
        gait = self.gait
        phases = gait.phases(t)
        vx, vy = float(cmd.get("vx", 0.0)), float(cmd.get("vy", 0.0))
        wz = float(cmd.get("wz", 0.0))
        height = self.target_height(cmd.get("height"))
        velocity = np.array([vx, vy])
        speed = float(np.hypot(vx, vy))
        stride = gait.stride(speed) if speed > 1e-9 else 0.0
        direction = (velocity / speed) if speed > 1e-9 else np.zeros(2)
        v_meas = np.asarray(v_meas, dtype=np.float64).reshape(2)
        stance_time, swing_time = gait.stance_time(), gait.swing_time()

        plan = {}
        for name in LEG_NAMES:
            phase = phases[name]
            stance = phase < gait.duty
            nominal = self.model.nominal_foot(name).copy()
            nominal[2] = -height                       # keep the belly height

            if stance and not self._was_stance[name]:
                # Just touched down: freeze the lead this foot came down with.
                # The endpoint its swing was aiming at last tick is the honest
                # record of that, not whatever the Raibert term says now.
                self._landed_lead[name] = self._pending_lead[name]
            self._was_stance[name] = stance

            r_landed = nominal[:2] + self._landed_lead[name]
            if stance:
                r_next = r_landed
            else:
                # Raibert: land ahead by half a stance of travel, corrected by
                # the velocity error, so a body running hot plants its feet
                # further forward and brakes on landing.
                along = float(direction @ v_meas)
                lead = direction * raibert_foothold(speed, along, stance_time,
                                                    self.raibert_gain)
                r_next = nominal[:2] + lead
                self._pending_lead[name] = lead

            x, y, dz = foot_target(phase, gait.duty, stance_time, swing_time,
                                   velocity, wz, self.lift, r_landed, r_next)
            target = self._reachable(name, np.array([x, y, nominal[2] + dz]))
            plan[name] = {"target": target, "phase": float(phase),
                          "stance": bool(stance), "stride": stride,
                          "dz": float(dz)}
        return plan

    def _reachable(self, name, target):
        """Clamp the target inside the leg's reachable sphere."""
        offset = target - self.model.hip[name]
        return self.model.hip[name] + self.model.clamp_reachable(offset)

    def joints(self, plan):
        return {n: self.model.foot_ik(n, p["target"]) for n, p in plan.items()}


class KinematicSimulator:
    """Integrates body motion through the stance-contact constraint."""

    def __init__(self, model, controller, dt=0.002):
        self.model = model
        self.controller = controller
        self.dt = float(dt)
        self.reset()

    def reset(self, t0=0.0):
        self.t = float(t0)
        self.pose = np.array([0.0, 0.0, 0.0])
        self.yaw = 0.0
        self.contacts = {}
        self.have_prev = False
        self.vel = np.zeros(2)
        self.log = []
        # Start with all four feet planted at their nominal positions.
        joints = self.model.default_pose(self.controller.height)
        feet = self.model.feet_body(joints)
        self.pose[2] = -min(float(p[2]) for p in feet.values())
        for name in LEG_NAMES:
            self.contacts[name] = self._to_world(feet[name])

    # ------------------------------------------------------------- transforms
    def _to_world(self, p_body):
        return self.pose + rotate_z(self.yaw) @ np.asarray(p_body)

    # -------------------------------------------------------------------- tick
    def step(self, cmd):
        dt = self.dt
        plan = self.controller.plan(self.t, cmd, self.vel)
        joints = self.controller.joints(plan)
        feet_body = self.model.feet_body(joints)

        # "Planted" is a *height*, not the phase boolean. The phase comparison
        # phase < duty flips a hair before the foot actually leaves the ground:
        # accumulating t += dt drifts, so a leg can be admitted to the contact
        # set on its very last stance sample (phase 0.4999...), and one tick
        # later it is already in swing (dz = 8.6e-5 m) while still anchored.
        # The solve then sinks the body by that clearance and never gives it
        # back -- a one-way ratchet of exactly lift_profile(dt/swing_time) per
        # liftoff. Keying the contact set on the commanded clearance closes it:
        # the swing branch is continuous from dz = 0 at tau = 0, so the only
        # samples that pass PLANT_EPS are the genuinely grounded ones.
        planted = {n: plan[n]["dz"] <= PLANT_EPS for n in LEG_NAMES}

        # ---- 0. one-off initialisation. reset() can only anchor the feet at
        # the *neutral* pose, because it does not see the command yet. Any yaw
        # command rotates the feet at t = 0 (turn_cycle is nonzero there), so
        # those stale anchors disagree with the first plan and the very first
        # solve splits the difference -- measured as a one-off -0.049 rad yaw
        # jump that then costs 3.4% of the total turn over a 3 s run. Re-anchor
        # once, against the neutral pose, as soon as the first command is known.
        if not self.have_prev:
            for name in LEG_NAMES:
                if planted[name]:
                    self.contacts[name] = self._to_world(feet_body[name])

        # ---- 1. solve the pose from the feet that were planted DURING this
        # tick, i.e. the contact set inherited from the previous tick.
        #
        # Order matters, and getting it wrong costs exactly 1% of the commanded
        # speed. If the contact set is refreshed first, a foot that lands at
        # this instant is anchored at _to_world(feet_body) computed with the
        # *pre-solve* pose; the Procrustes problem then has the current pose as
        # an exact solution, so the body does not advance at all for that tick.
        # Measured: two such frozen ticks per gait period, 2*dt*v = 0.0016 m of
        # lost travel per 0.16 m stride. A foot lifting off now was on the
        # ground for the whole interval so it still takes part in the solve; a
        # foot landing now was airborne for the whole interval so it must not.
        active = [n for n in LEG_NAMES if n in self.contacts and planted[n]]
        if len(active) < 2:
            # Degenerate handover. A staggered gait (a crawl at low duty) can
            # take one foot off the ground a tick before its replacement lands,
            # leaving a single constraint -- and one contact point cannot
            # observe yaw at all, so the body would sit still for a whole tick
            # and make it up on the next one. Measured cost when it happens:
            # 0.0008284 rad per 0.5 s cycle, i.e. 0.41% of the commanded turn
            # rate, on every cycle. Note this straddle is a sampling accident:
            # a 2 ms tick against a 0.4 s period hits the phase boundary
            # exactly (and a trot never sees this), while against 0.5 s it
            # steps over it.
            #
            # Falling back to the full inherited set is safe: the foot that
            # just lifted is at most a fraction of a millimetre up, and the
            # following tick re-solves against the older anchor, so the height
            # transient is not retained.
            active = list(self.contacts)
        if len(active) >= 2:
            body_pts = np.array([feet_body[n] for n in active])
            world_pts = np.array([self.contacts[n] for n in active])
            prev = self.pose.copy()
            prev_yaw = self.yaw
            self.pose, yaw = solve_body_pose(body_pts, world_pts)
            # solve_body_pose returns a branch-folded angle in (-pi, pi]. Unwrap
            # it against the previous sample, otherwise the body yaw jumps by
            # 2*pi every half turn and any rate measured from it is garbage.
            self.yaw = float(yaw + 2.0 * np.pi
                             * np.round((prev_yaw - yaw) / (2.0 * np.pi)))
            if self.have_prev:
                self.vel = (self.pose[:2] - prev[:2]) / dt
            self.have_prev = True

        # ---- 2. retire the feet that have left the ground, and pin the feet
        # that are now down at the world position they went down with.
        for name in LEG_NAMES:
            if planted[name]:
                if name not in self.contacts:
                    self.contacts[name] = self._to_world(feet_body[name])
            elif name in self.contacts:
                del self.contacts[name]

        self.t += dt
        self.log.append((self.t, self.pose.copy(), self.yaw,
                         {n: joints[n].copy() for n in LEG_NAMES}))
        return joints

    def run(self, cmd, duration):
        n = int(round(duration / self.dt))
        for _ in range(n):
            self.step(cmd)
        return self
