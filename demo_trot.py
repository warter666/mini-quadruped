"""Manoeuvre sweep plus an SVG report for the mini quadruped stack.

    python demo_trot.py

Runs a set of commanded manoeuvres through the contact-constrained kinematic
simulator, compares what was asked for against what the legs actually produced,
and writes demo_trot.svg with four panels:

    1  top-down     body path and footfall pattern for a straight walk and a
                    circle, in world coordinates
    2  foot cycle   one leg's body-frame foot trajectory, i.e. the gait
    3  schedule     stance/swing bars per leg for trot (duty 0.5) and crawl
                    (duty 0.75); this is the whole difference between them
    4  crawl support  the three-foot triangle, its Chebyshev centre, and why a
                    static crawl cannot start from the nominal stance

Only numpy is used; the SVG is written by hand.
"""

import numpy as np

from pathlib import Path

from quadruped.controller import KinematicSimulator, TrotController
from quadruped.gait import CRAWL_OFFSETS, TROT_OFFSETS, Gait
from quadruped.kinematics import LEG_NAMES, QuadrupedModel
from quadruped.stabilizer import chebyshev_center, support_polygon

MODEL = QuadrupedModel(hip_x=0.19, hip_y=0.085, thigh=0.20, shank=0.20)
LEG_COLOUR = {"FL": "#d64541", "FR": "#2e86c1", "RL": "#28a745",
              "RR": "#b9751f"}
FONT = "Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif"


# ------------------------------------------------------------------- helpers

def run(cmd, duration, period=0.4, duty=0.5, offsets=None, dt=0.002):
    gait = Gait(period=period, duty=duty, offsets=offsets)
    controller = TrotController(MODEL, gait, height=0.26, lift=0.055)
    return KinematicSimulator(MODEL, controller, dt=dt).run(cmd, duration)


def measure(sim, t_from=1.0, t_to=None):
    """Arc speed (|v|, so always >= 0), yaw rate, height deviation."""
    rows = [r for r in sim.log if r[0] >= t_from and
            (t_to is None or r[0] <= t_to)]
    p = np.array([r[1] for r in rows])
    ys = np.array([r[2] for r in rows])
    span = rows[-1][0] - rows[0][0]
    arc = float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum() / span)
    return {"arc": arc,
            "yaw_rate": float((ys[-1] - ys[0]) / span),
            "height_dev": float(np.abs(p[:, 2] - 0.26).max())}


def footfalls(sim, tol=1e-9):
    """World position of every touchdown, per leg.

    A foot is on the ground exactly when its height above the terrain is zero,
    and the world height of a foot is pose_z + body_z (the yaw is about the
    body z axis, so it does not mix in). Count a touchdown at the first sample
    of each grounded run.
    """
    grounded = {n: [] for n in LEG_NAMES}
    for _, pose, _, joints in sim.log:
        feet = MODEL.feet_body(joints)
        for n in LEG_NAMES:
            grounded[n].append(bool(pose[2] + feet[n][2] <= tol))

    falls = {}
    for n in LEG_NAMES:
        pts = []
        for i in range(1, len(grounded[n])):
            if grounded[n][i] and not grounded[n][i - 1]:
                _, pose, yaw, joints = sim.log[i]
                c, s = np.cos(yaw), np.sin(yaw)
                R = np.array([[c, -s], [s, c]])
                pts.append(R @ MODEL.feet_body(joints)[n][:2] + pose[:2])
        falls[n] = np.array(pts) if pts else np.zeros((0, 2))
    return falls


# ---------------------------------------------------------------- manoeuvres

def manoeuvres():
    print("quadruped manoeuvre sweep   hip(0.19, 0.085)  thigh+shank 0.20+0.20")
    print("steering v_arc is a magnitude, so reverse shows its absolute value")
    print()
    print("%-26s %10s %10s %10s %10s %10s"
          % ("command", "v_arc", "v_cmd", "yaw_rate", "wz_cmd", "h_dev"))
    print("-" * 82)
    cases = [
        ("stand", {"vx": 0.0}),
        ("forward  vx=0.40", {"vx": 0.40}),
        ("reverse  vx=-0.30", {"vx": -0.30}),
        ("lateral  vy=0.25", {"vy": 0.25}),
        ("diagonal vx=0.2 vy=0.2", {"vx": 0.2, "vy": 0.2}),
        ("spin     wz=0.50", {"wz": 0.50}),
        ("circle   vx=0.3 wz=0.4", {"vx": 0.3, "wz": 0.40}),
        ("tight    vx=0.2 wz=-0.8", {"vx": 0.2, "wz": -0.80}),
    ]
    straight = None
    for label, cmd in cases:
        sim = run(cmd, 4.0)
        m = measure(sim, 1.0, 4.0)
        v_cmd = float(np.hypot(cmd.get("vx", 0.0), cmd.get("vy", 0.0)))
        print("%-26s %10.5f %10.5f %10.5f %10.5f %10.2e"
              % (label, m["arc"], v_cmd, m["yaw_rate"],
                 cmd.get("wz", 0.0), m["height_dev"]))
        if label.startswith("forward"):
            straight = sim

    print()
    crawl = run({"vx": 0.25}, 6.0, period=0.6, duty=0.75,
                offsets=CRAWL_OFFSETS)
    m = measure(crawl, 2.0, 6.0)
    print("crawl duty 0.75, vx=0.25, T=0.6")
    print("  arc speed %.5f (cmd 0.25000)   yaw drift %.2e deg   "
          "height dev %.2e m"
          % (m["arc"], np.rad2deg(crawl.yaw), m["height_dev"]))
    return straight, crawl


# ---------------------------------------------------------------------- svg

def _polyline(pts, sx, sy):
    if len(pts) == 0:
        return ""
    d = "M %.2f %.2f" % (sx(pts[0][0]), sy(pts[0][1]))
    for q in pts[1:]:
        d += " L %.2f %.2f" % (sx(q[0]), sy(q[1]))
    return d


def _mapper(box, x0, y0, w, h, pad=0.12):
    """Fit `box` (an (N, 2) array) into a w x h rect, y flipped."""
    lo, hi = np.asarray(box).min(0), np.asarray(box).max(0)
    size = np.maximum(hi - lo, 1e-9)
    lo, hi = lo - pad * size, hi + pad * size
    size = np.maximum(hi - lo, 1e-9)
    k = min(w / size[0], h / size[1])
    cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0

    def sx(x):
        return x0 + w / 2.0 + (x - cx) * k

    def sy(y):
        return y0 + h / 2.0 - (y - cy) * k
    return sx, sy, k


def _crawl_triangle():
    """The tightest three-foot crawl support triangle, its Chebyshev centre and
    the margin available there. Cached: the SVG asks for it several times."""
    if not hasattr(_crawl_triangle, "_cache"):
        gait = Gait(period=0.6, duty=0.75, offsets=CRAWL_OFFSETS)
        worst, rank = None, float("inf")
        for t in np.arange(0.0, gait.period, gait.period / 200.0):
            phases = gait.phases(t)
            down = [n for n in LEG_NAMES if phases[n] < gait.duty]
            if len(down) != 3:
                continue
            poly = support_polygon([MODEL.nominal_foot(n)[:2] for n in down])
            if chebyshev_center(poly)[1] < rank:
                rank = chebyshev_center(poly)[1]
                worst = poly
        centre, margin = chebyshev_center(worst)
        _crawl_triangle._cache = (worst, centre, margin)
    return _crawl_triangle._cache


def svg(straight, circle):
    W, H = 960, 1500
    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
           'width="%d" height="%d" font-family="%s">' % (W, H, W, H, FONT),
           '<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H)]

    def text(x, y, s, size=13, colour="#222222", weight="normal",
             anchor="start"):
        out.append('<text x="%.1f" y="%.1f" font-size="%d" fill="%s" '
                   'font-weight="%s" text-anchor="%s">%s</text>'
                   % (x, y, size, colour, weight, anchor, s))

    rects = []

    def panel(x, y, w, h, text_id):
        rects.append((x, y, w, h))
        out.append('<rect id="%s" x="%.1f" y="%.1f" width="%.1f" '
                   'height="%.1f" fill="#fbfbfd" stroke="#e3e3e8"/>'
                   % (text_id, x, y, w, h))

    # ---------------------------------------------------------- 1  top-down
    text(40, 38, "① 俯视图：机身轨迹与落足点", 17, weight="bold")
    text(40, 60, "灰 = 直行 vx=0.40；蓝 = 圆周 vx=0.30 / wz=0.40。"
                 "每个小点是一次触地瞬间的世界坐标，颜色按腿区分。", 12,
         "#555555")
    px, py, pw, ph = 60, 84, 840, 360
    panel(px, py, pw, ph, "map")
    box = np.vstack([np.array([r[1][:2] for r in s.log])
                     for s in (straight, circle)])
    sx, sy, k = _mapper(box, px, py, pw, ph, pad=0.08)
    for sim, colour, width in ((straight, "#9aa0a6", 2.0),
                               (circle, "#2e86c1", 2.8)):
        path = np.array([r[1][:2] for r in sim.log])
        out.append('<path d="%s" fill="none" stroke="%s" stroke-width="%.1f"/>'
                   % (_polyline(path, sx, sy), colour, width))
        for leg, pts in footfalls(sim).items():
            for q in pts:
                out.append('<circle cx="%.1f" cy="%.1f" r="2.6" fill="%s" '
                           'fill-opacity="0.85"/>'
                           % (sx(q[0]), sy(q[1]), LEG_COLOUR[leg]))
    start = straight.log[0][1][:2]
    out.append('<circle cx="%.1f" cy="%.1f" r="5.5" fill="none" '
               'stroke="#111111" stroke-width="2"/>' % (sx(start[0]),
                                                        sy(start[1])))
    text(sx(start[0]) + 10, sy(start[1]) - 8, "起点", 12, "#111111")
    bar = 160
    out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#555555" '
               'stroke-width="2"/>' % (px + 30, py + ph - 18,
                                       px + 30 + k, py + ph - 18))
    text(px + 30, py + ph - 26, "1 m", 12, "#555555")

    # ---------------------------------------------------------- 2  foot cycle
    y2 = py + ph + 54
    text(40, y2, "② 单腿足端在机体坐标系中的轨迹（trot，duty 0.5）", 17,
         weight="bold")
    text(40, y2 + 22, "○ = 触地那一刻。支撑相是直线：足端相对机身匀速后滑；"
                      "摆动相平滑前摆，两对角腿同相、另两腿反相。", 12, "#555555")
    px2, py2, pw2, ph2 = 60, y2 + 36, 840, 250
    panel(px2, py2, pw2, ph2, "foot")
    cycle = [r for r in straight.log if 3.0 <= r[0] <= 3.4]
    tracks = {n: np.array([MODEL.feet_body(r[3])[n][:2] for r in cycle])
              for n in LEG_NAMES}
    sx2, sy2, _ = _mapper(np.vstack(list(tracks.values())),
                          px2, py2, pw2, ph2, pad=0.10)
    for n in LEG_NAMES:
        out.append('<path d="%s" fill="none" stroke="%s" stroke-width="1.9"/>'
                   % (_polyline(tracks[n], sx2, sy2), LEG_COLOUR[n]))
        out.append('<circle cx="%.1f" cy="%.1f" r="3.6" fill="none" '
                   'stroke="%s" stroke-width="2"/>'
                   % (sx2(tracks[n][0][0]), sy2(tracks[n][0][1]),
                      LEG_COLOUR[n]))
    lx = px2 + pw2 - 90
    for i, n in enumerate(LEG_NAMES):
        out.append('<rect x="%d" y="%.1f" width="10" height="10" fill="%s"/>'
                   % (lx, py2 + 14 + i * 20, LEG_COLOUR[n]))
        text(lx + 16, py2 + 23 + i * 20, n, 12)

    # ---------------------------------------------------------- 3  schedule
    y3 = py2 + ph2 + 54
    text(40, y3, "③ 步态相位表：trot 与 crawl 的全部差别就在 duty", 17,
         weight="bold")
    text(40, y3 + 22, "实心 = 支撑相。trot duty=0.5 时对角两足同起同落，"
                      "支撑多边形退化成一条线（零面积）；crawl duty=0.75 "
                      "全程至少三足着地。", 12, "#555555")
    bx, bw = 132, 740
    block = 150
    for bi, (name, duty, offs, period) in enumerate(
            (("trot", 0.5, TROT_OFFSETS, 0.4),
             ("crawl", 0.75, CRAWL_OFFSETS, 0.6))):
        by = y3 + 42 + bi * block
        out.append('<text x="40" y="%.1f" font-size="14" font-weight="bold" '
                   'fill="#222222">%s   duty=%.2f   T=%.1f s</text>'
                   % (by + 14, name, duty, period))
        span = 2.0 * period
        for li, leg in enumerate(LEG_NAMES):
            ry = by + 26 + li * 25
            text(bx - 10, ry + 12, leg, 12, "#222222", anchor="end")
            panel(bx, ry, bw, 16, "bar_%s_%d" % (leg, bi))
            for k in (-1, 0, 1, 2):
                st = (k - offs[leg]) * period
                en = st + duty * period
                if en <= 0.0 or st >= span:
                    continue
                a, b = max(st, 0.0), min(en, span)
                out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="16" '
                           'fill="%s" fill-opacity="0.8"/>'
                           % (bx + a / span * bw, ry, (b - a) / span * bw,
                              LEG_COLOUR[leg]))
        for k in range(3):
            gx = bx + k * bw / 2.0
            out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" '
                       'stroke="#bbbbbb" stroke-dasharray="3 3"/>'
                       % (gx, by + 22, gx, by + 26 + 4 * 25))
            text(gx, by + 146, "%.1f s" % (k * period), 11, "#777777",
                 anchor="middle")

    # ---------------------------------------------------------- 4  crawl support
    y4 = y3 + 42 + 2 * block + 22
    poly, centre, margin = _crawl_triangle()
    text(40, y4, "④ 爬行步态的三足支撑三角形", 17, weight="bold")
    text(40, y4 + 22, "名义质心（红）恰好落在三角形对角线上，裕度 0——"
                      "静态爬行必须先横移机身；切比雪夫中心（绿）有 %.2f cm 余量。"
         % (100 * margin), 12, "#555555")
    bx4, by4, bw4, bh4 = 60, y4 + 38, 840, 210
    panel(bx4, by4, bw4, bh4, "triangle")
    sx4, sy4, _ = _mapper(poly, bx4, by4, bw4, bh4, pad=0.30)
    out.append('<path d="%s Z" fill="#2e86c1" fill-opacity="0.10" '
               'stroke="#2e86c1" stroke-width="2"/>'
               % _polyline(np.vstack([poly, poly[:1]]), sx4, sy4))
    for q in poly:
        out.append('<circle cx="%.1f" cy="%.1f" r="4.5" fill="#2e86c1"/>'
                   % (sx4(q[0]), sy4(q[1])))
    out.append('<circle cx="%.1f" cy="%.1f" r="5.5" fill="#d64541"/>'
               % (sx4(0.0), sy4(0.0)))
    text(sx4(0.0) + 11, sy4(0.0) - 8, "名义质心  裕度 0", 12, "#d64541")
    out.append('<circle cx="%.1f" cy="%.1f" r="5.5" fill="#28a745"/>'
               % (sx4(centre[0]), sy4(centre[1])))
    text(sx4(centre[0]) + 11, sy4(centre[1]) + 17,
         "切比雪夫中心  裕度 %.2f cm" % (100 * margin), 12, "#28a745")

    out.append('</svg>')
    payload = "\n".join(out) + "\n"
    over = [r for r in rects
            if r[0] < 0 or r[1] < 0 or r[0] + r[2] > W or r[1] + r[3] > H]
    return payload, {"size": (W, H), "rects": rects, "overflow": over}


def main():
    straight, _ = manoeuvres()
    circle = run({"vx": 0.3, "wz": 0.40}, 4.0)
    payload, layout = svg(straight, circle)
    if layout["overflow"]:
        raise SystemExit("panel overflows the canvas: %s (canvas %s)"
                         % (layout["overflow"], layout["size"]))
    lowest = max(y + h for _, y, _, h in layout["rects"])
    # output is anchored next to this script and containment-verified
    demo_dir = Path(__file__).resolve().parent
    out_path = (demo_dir / "demo_trot.svg").resolve()
    if out_path.parent != demo_dir:
        raise SystemExit("refusing to write outside the demo directory")
    out_path.write_text(payload, encoding="utf-8")
    print("wrote %s   canvas %dx%d, %d panels, lowest panel edge y=%.0f"
          % (out_path.name, layout["size"][0], layout["size"][1],
             len(layout["rects"]), lowest))


if __name__ == "__main__":
    main()
