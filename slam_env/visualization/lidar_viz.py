"""
lidar_viz.py
============
OpenCV lidar verification — run with regular python (NOT mjpython).
Reads from sim_runner.py via named pipe.

HOW TO RUN
----------
  Terminal 1:  mjpython visualization/sim_runner.py --map simple_room --noise
  Terminal 2:  python  visualization/lidar_viz.py

Controls (OpenCV window):
  Q / Esc  — quit
  C        — clear accumulated point cloud
  P        — toggle scan persistence (accumulate vs single scan)
  N        — toggle showing noiseless reference

What you see
------------
  • White lines   — true map walls
  • CYAN dots     — noiseless reference scan (where beams SHOULD land)
  • GREEN dots    — valid noisy hits (1px, scatter visible around cyan)
  • RED   dots    — short outliers (large ×, spurious near returns)
  • ORANGE dots   — long outliers / dropouts (large ○, missed detections)
  • PURPLE dots   — mixed-pixel returns (△, depth-edge blending)
  • Robot: yellow circle + heading arrow

RIGHT panel (polar, body frame):
  • Same colour coding, dots at bearing+range from sensor
"""

import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from slam_env.visualization._ipc import open_reader

# ── Canvas ────────────────────────────────────────────────────────────────────
CANVAS_W = 1400
CANVAS_H = 720
PANEL_W  = CANVAS_W // 2
BG       = (18, 18, 24)

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
WALL_C    = (180, 180, 180)
ROBOT_C   = (0,   210, 255)   # yellow
REF_C     = (200, 200,  40)   # cyan
VALID_C   = (50,  220,  50)   # green
SHORT_C   = (40,   40, 230)   # red
LONG_C    = (30,  130, 240)   # orange
MIXED_C   = (200,  60, 160)   # purple
GRID_C    = (35,   35,  45)

FLAG_C = {0: VALID_C, 1: SHORT_C, 2: LONG_C, 3: MIXED_C, 4: (180, 100, 40)}
FONT   = cv2.FONT_HERSHEY_SIMPLEX

# ── Dot sizes ─────────────────────────────────────────────────────────────────
VALID_R   = 1     # 1px — tiny so scatter is visible as halo around reference
REF_R     = 2     # cyan reference slightly larger
OUTLIER_R = 6     # large so outliers are immediately obvious
ROBOT_R   = 9


# ── Coordinate mappers ────────────────────────────────────────────────────────

class WorldView:
    """World metres → left-panel pixels."""
    M = 14  # margin px

    def __init__(self, bounds):
        xmin,xmax,ymin,ymax = bounds
        pad = 0.18
        self.xmin = xmin-pad; self.xmax = xmax+pad
        self.ymin = ymin-pad; self.ymax = ymax+pad
        self.w = PANEL_W - 2*self.M
        self.h = CANVAS_H - 2*self.M

    def to_px(self, wx, wy):
        sx = (wx - self.xmin) / (self.xmax - self.xmin)
        sy = 1.0 - (wy - self.ymin) / (self.ymax - self.ymin)  # flip y
        return (int(self.M + sx * self.w),
                int(self.M + sy * self.h))


class PolarView:
    """Body-frame (rel_angle, range) → right-panel pixels."""
    def __init__(self, max_range):
        self.cx  = PANEL_W + PANEL_W // 2
        self.cy  = CANVAS_H // 2
        self.sc  = min(PANEL_W, CANVAS_H) // 2 - 20
        self.mr  = max_range

    def to_px(self, angle, dist):
        # forward = up = image angle -π/2; CW positive
        ia  = -math.pi/2 + angle
        r   = dist / self.mr * self.sc
        return (int(self.cx + r*math.cos(ia)),
                int(self.cy + r*math.sin(ia)))


# ── Persistence buffer ────────────────────────────────────────────────────────

class ScanBuffer:
    """Accumulates world-frame hit points across multiple scans."""
    def __init__(self, maxscans=30):
        self.maxscans = maxscans
        self._scans   = []    # each entry: (hx, hy, flags, dists, max_range)

    def push(self, hx, hy, flags, dists, max_range):
        self._scans.append((hx.copy(), hy.copy(),
                            flags.copy(), dists.copy(), max_range))
        if len(self._scans) > self.maxscans:
            self._scans.pop(0)

    def clear(self):
        self._scans.clear()

    def all_points(self):
        """Yields (hx, hy, flag, dist, max_range) per scan."""
        for entry in self._scans:
            yield entry


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_world(canvas, wv, walls, buf, ref_hx, ref_hy,
               rx, ry, rth, show_ref, payload):
    """Draw the left (world-frame) panel onto canvas."""

    # Grid
    xmin,xmax,ymin,ymax = wv.xmin+0.18, wv.xmax-0.18, wv.ymin+0.18, wv.ymax-0.18
    for gx in np.arange(math.ceil(xmin), math.floor(xmax)+1):
        cv2.line(canvas, wv.to_px(gx, ymin), wv.to_px(gx, ymax), GRID_C, 1)
    for gy in np.arange(math.ceil(ymin), math.floor(ymax)+1):
        cv2.line(canvas, wv.to_px(xmin, gy), wv.to_px(xmax, gy), GRID_C, 1)

    # Walls
    for (x1,y1,x2,y2) in walls:
        cv2.line(canvas, wv.to_px(x1,y1), wv.to_px(x2,y2),
                 WALL_C, 2, cv2.LINE_AA)

    # Accumulated valid + outlier dots from buffer
    for (hx, hy, flags, dists, max_r) in buf.all_points():
        # Valid hits — 1px green (scatter visible around cyan reference)
        valid = (flags == 0) & (dists < max_r * 0.99)
        for wx,wy in zip(hx[valid], hy[valid]):
            cv2.circle(canvas, wv.to_px(wx,wy), VALID_R, VALID_C, -1)

    # Outliers drawn on top from latest scan only (always visible)
    hx  = payload["hx"];  hy  = payload["hy"]
    fl  = payload["flags"]; d  = payload["dists"]
    max_r = payload["max_range"]

    # Short outliers — large red X
    short = fl == 1
    for wx,wy in zip(hx[short], hy[short]):
        p = wv.to_px(wx, wy)
        cv2.drawMarker(canvas, p, SHORT_C, cv2.MARKER_CROSS,
                       OUTLIER_R*2, 2, cv2.LINE_AA)

    # Long/dropout — large orange circle
    long_ = fl == 2
    for wx,wy in zip(hx[long_], hy[long_]):
        p = wv.to_px(wx, wy)
        cv2.circle(canvas, p, OUTLIER_R, LONG_C, 2, cv2.LINE_AA)

    # Mixed pixel — purple diamond
    mixed = fl == 3
    for wx,wy in zip(hx[mixed], hy[mixed]):
        p = wv.to_px(wx, wy)
        cv2.drawMarker(canvas, p, MIXED_C, cv2.MARKER_DIAMOND,
                       OUTLIER_R*2, 2, cv2.LINE_AA)

    # Reference scan — cyan 2px dots on top
    if show_ref and ref_hx is not None:
        for wx,wy in zip(ref_hx, ref_hy):
            cv2.circle(canvas, wv.to_px(wx,wy), REF_R, REF_C, -1)

    # Robot
    rp = wv.to_px(rx, ry)
    cv2.circle(canvas, rp, ROBOT_R, ROBOT_C, -1, cv2.LINE_AA)
    cv2.circle(canvas, rp, ROBOT_R, (255,255,255), 1, cv2.LINE_AA)
    ep = wv.to_px(rx + 0.25*math.cos(rth), ry + 0.25*math.sin(rth))
    cv2.arrowedLine(canvas, rp, ep, ROBOT_C, 2, cv2.LINE_AA, tipLength=0.3)


def draw_polar(canvas, pv, payload, show_ref, ref_angles, ref_dists):
    """Draw right (polar) panel onto canvas."""
    max_r = payload["max_range"]
    ra    = payload["rel_angles"]
    dists = payload["dists"]
    flags = payload["flags"]

    # Range rings
    for r_m in np.arange(1.0, max_r+0.5, 1.0):
        r_px = int(r_m / max_r * pv.sc)
        cv2.circle(canvas, (pv.cx, pv.cy), r_px, (45,45,55), 1, cv2.LINE_AA)
        lp = (pv.cx+4, pv.cy-r_px-3)
        cv2.putText(canvas, f"{r_m:.0f}m", lp, FONT, 0.28, (70,70,80), 1)

    # Cardinal spokes
    for ang, lbl in [(0,"fwd"),(math.pi/2,"R"),(math.pi,"bwd"),(-math.pi/2,"L")]:
        ia = -math.pi/2+ang
        ex = int(pv.cx+pv.sc*math.cos(ia)); ey=int(pv.cy+pv.sc*math.sin(ia))
        cv2.line(canvas, (pv.cx,pv.cy),(ex,ey),(40,40,50),1,cv2.LINE_AA)

    # Reference
    if show_ref and ref_angles is not None:
        rd  = np.asarray(ref_dists)
        ra_ = np.asarray(ref_angles)
        for a,d in zip(ra_[rd<max_r*0.99], rd[rd<max_r*0.99]):
            p = pv.to_px(float(a), float(d))
            if PANEL_W <= p[0] < CANVAS_W and 0 <= p[1] < CANVAS_H:
                cv2.circle(canvas, p, 2, REF_C, -1)

    # Valid hits — 1px
    valid = (flags==0) & (dists<max_r*0.99)
    for a,d in zip(ra[valid], dists[valid]):
        p = pv.to_px(float(a), float(d))
        if PANEL_W <= p[0] < CANVAS_W and 0 <= p[1] < CANVAS_H:
            cv2.circle(canvas, p, 1, VALID_C, -1)

    # Outliers — large markers
    for fv, col, mtype, ms in [
        (1, SHORT_C, cv2.MARKER_CROSS,    12),
        (2, LONG_C,  cv2.MARKER_CIRCLE,   10),
        (3, MIXED_C, cv2.MARKER_DIAMOND,  10),
    ]:
        mask = flags == fv
        for a,d in zip(ra[mask], dists[mask]):
            p = pv.to_px(float(a), float(d))
            if PANEL_W <= p[0] < CANVAS_W and 0 <= p[1] < CANVAS_H:
                if mtype == cv2.MARKER_CIRCLE:
                    cv2.circle(canvas, p, 6, col, 2, cv2.LINE_AA)
                else:
                    cv2.drawMarker(canvas, p, col, mtype, ms, 2, cv2.LINE_AA)

    # Robot at centre
    cv2.circle(canvas, (pv.cx,pv.cy), ROBOT_R, ROBOT_C, -1, cv2.LINE_AA)


def draw_hud(canvas, payload, buf_len, show_ref, persist):
    """Draw overlay text / legend."""
    d   = payload["dists"]; f = payload["flags"]
    v   = payload["v_phys"]; step = payload["step"]
    rx  = payload["rx"];    ry  = payload["ry"]
    rth = payload["rth"]
    max_r = payload["max_range"]
    d = np.asarray(d); f = np.asarray(f)
    n_valid = int(((f==0)&(d<max_r*0.99)).sum())
    n_short = int((f==1).sum()); n_long=(f==2).sum(); n_mixed=(f==3).sum()

    # Left panel info
    lines = [
        f"step {step}  v={v:+.3f}m/s",
        f"x={rx:.3f}m  y={ry:.3f}m  t={math.degrees(rth)%360:.1f}deg",
        f"hits={n_valid}  short={n_short}  long={n_long}  mixed={n_mixed}",
        f"scans buffered={buf_len}  persist={'ON' if persist else 'OFF'}",
    ]
    for i,line in enumerate(lines):
        cv2.putText(canvas, line, (10, CANVAS_H-60+i*15),
                    FONT, 0.36, (180,180,180), 1, cv2.LINE_AA)

    # Legend — left panel top
    items = [
        (VALID_C,  "valid (1px - scatter=noise)"),
        (REF_C,    "noiseless ref (2px cyan)"),
        (SHORT_C,  "short outlier (red X)"),
        (LONG_C,   "long/dropout (orange O)"),
        (MIXED_C,  "mixed-pixel (purple <>)"),
    ]
    for i,(c,lbl) in enumerate(items):
        y = 30 + i*18
        cv2.circle(canvas,(12,y),4,c,-1,cv2.LINE_AA)
        cv2.putText(canvas,lbl,(20,y+4),FONT,0.33,(200,200,200),1,cv2.LINE_AA)

    # Right panel label
    cv2.putText(canvas,"Body-frame polar  (fwd=up)",
                (PANEL_W+10, 22),FONT,0.45,(160,160,160),1,cv2.LINE_AA)

    # Controls reminder
    cv2.putText(canvas,"C=clear  P=persist  N=ref  Q=quit",
                (PANEL_W+10, CANVAS_H-10),FONT,0.33,(100,100,100),1)

    # Panel divider
    cv2.line(canvas,(PANEL_W,0),(PANEL_W,CANVAS_H),(55,55,65),1)

    # Left panel label
    cv2.putText(canvas,"World frame — lidar point cloud",
                (10,22),FONT,0.45,(160,160,160),1,cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Lidar Viz (OpenCV)  — waiting for sim_runner.py")
    print("  Controls: C=clear  P=persist  N=ref  Q=quit")
    print("=" * 60)

    reader   = open_reader()
    print("  Connected.\n")

    win = "Lidar Verification"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, CANVAS_W, CANVAS_H)

    buf      = ScanBuffer(maxscans=40)
    persist  = True    # accumulate scans to show noise scatter
    show_ref = True
    wv = pv = None     # init on first frame

    while True:
        payload = reader.recv()
        if payload is None:
            break

        # Unpack
        rx    = payload["rx"];  ry  = payload["ry"];  rth = payload["rth"]
        ra    = np.asarray(payload["rel_angles"])
        dists = np.asarray(payload["dists"])
        flags = np.asarray(payload["flags"], dtype=np.uint8)
        max_r = payload["max_range"]
        walls = payload["walls"]
        bounds= payload["bounds"]

        # World-frame hit coordinates
        ga = rth + ra
        hx = rx + dists * np.cos(ga)
        hy = ry + dists * np.sin(ga)
        payload["hx"] = hx; payload["hy"] = hy

        # Reference scan
        ref_angles = np.asarray(payload["ref_angles"])
        ref_dists  = np.asarray(payload["ref_dists"])
        ref_mask   = ref_dists < max_r * 0.99
        ga_r  = rth + ref_angles[ref_mask]
        ref_hx = rx + ref_dists[ref_mask]*np.cos(ga_r)
        ref_hy = ry + ref_dists[ref_mask]*np.sin(ga_r)

        # Init coord mappers once
        if wv is None:
            wv = WorldView(bounds)
            pv = PolarView(max_r)

        # Buffer scan
        if persist:
            buf.push(hx, hy, flags, dists, max_r)
        else:
            buf.clear()
            buf.push(hx, hy, flags, dists, max_r)

        # Render
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, dtype=np.uint8)

        draw_world(canvas, wv, walls, buf,
                   ref_hx if show_ref else None,
                   ref_hy if show_ref else None,
                   rx, ry, rth, show_ref, payload)

        draw_polar(canvas, pv, payload, show_ref,
                   ref_angles, ref_dists)

        draw_hud(canvas, payload, len(buf._scans), show_ref, persist)

        cv2.imshow(win, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('c'), ord('C')):
            buf.clear(); print("  Cleared.", flush=True)
        elif key in (ord('p'), ord('P')):
            persist = not persist
            if not persist: buf.clear()
            print(f"  Persist: {persist}", flush=True)
        elif key in (ord('n'), ord('N')):
            show_ref = not show_ref
            print(f"  Ref scan: {show_ref}", flush=True)

    reader.close()
    cv2.destroyAllWindows()
    print("  Closed.\n")


if __name__ == "__main__":
    main()