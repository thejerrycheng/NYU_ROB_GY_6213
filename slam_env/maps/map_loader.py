"""
MapLoader
=========
Builds a MuJoCo XML string by:
  1. Reading  assets/env_base.xml     — static world shell
  2. Reading  assets/robot.xml        — car body template
  3. Loading  config/vehicle_config.yaml — all dimensions / colours
  4. Filling  {{PLACEHOLDER}} tokens with computed values
  5. Generating wall / obstacle geoms from the map dict

No hard-coded geometry numbers live here — everything comes from the YAML
or from the XML templates.
"""

import math, yaml
from pathlib import Path

_ROOT      = Path(__file__).parent.parent
_ASSET_DIR = _ROOT / "assets"
_CFG_DIR   = _ROOT / "config"


def _load_text(path: Path) -> str:
    with open(path) as f:
        return f.read()


def _rgba(lst) -> str:
    return " ".join(f"{v:.4f}" for v in lst)


class MapLoader:
    """
    Parameters
    ----------
    map_dict     : map definition dict  (walls, obstacles, robot_start, bounds)
    vehicle_cfg  : path to vehicle_config.yaml or None (uses default)
    """

    def __init__(self, map_dict: dict, vehicle_cfg=None):
        self.map_dict    = map_dict
        self.walls       = map_dict["walls"]
        self.obstacles   = map_dict.get("obstacles", [])
        self.robot_start = map_dict["robot_start"]
        self.bounds      = map_dict.get("bounds", (-1, 10, -1, 10))

        # Load vehicle config
        cfg_path = Path(vehicle_cfg) if vehicle_cfg else _CFG_DIR / "vehicle_config.yaml"
        with open(cfg_path) as f:
            self._vcfg = yaml.safe_load(f)

        self._precompute()

    # ── Pre-compute geometry from YAML ────────────────────────────────────────

    def _precompute(self):
        k  = self._vcfg["kinematics"]
        g  = self._vcfg["geometry"]
        ap = self._vcfg["appearance"]

        self.L         = float(k["wheelbase_m"])
        self.MAX_STEER = float(k["max_steer_deg"])

        self.BODY_LEN  = float(g["body_length_m"])
        self.BODY_WID  = float(g["body_width_m"])
        self.BODY_H    = float(g["body_height_m"])
        self.WHEEL_R   = float(g["wheel_radius_m"])
        self.WHEEL_HW  = float(g["wheel_halfwidth_m"])

        # Collision half-extents (exported for OBB clamping in teleop)
        self.COL_HALF_X = float(g["col_half_x_m"])
        self.COL_HALF_Y = float(g["col_half_y_m"])

        # Derived geometry
        self.WHEEL_Y    = self.BODY_WID / 2 + self.WHEEL_HW
        self.CHASSIS_CX = self.L / 2
        self.CHASSIS_Z  = self.WHEEL_R + self.BODY_H / 2
        self.WHEEL_Z    = self.WHEEL_R

        # Z offset: freejoint origin sits just above floor
        self.ROBOT_JOINT_Z = 0.001
        cy_ch = self.CHASSIS_Z - self.ROBOT_JOINT_Z
        cy_wh = self.WHEEL_Z   - self.ROBOT_JOINT_Z

        self._cy_ch = cy_ch
        self._cy_wh = cy_wh

        # Lidar z positions (above chassis top)
        chassis_top = cy_ch + self.BODY_H / 2
        self._lid_mt_z   = chassis_top + 0.015
        self._lid_dr_z   = chassis_top + 0.036
        self._lid_site_z = chassis_top + 0.055

        # Hub cap y position
        self._hub_y = self.WHEEL_Y + self.WHEEL_HW + 0.001

        # Bumper positions
        self._bumper_fx = self.CHASSIS_CX + self.COL_HALF_X
        self._bumper_rx = self.CHASSIS_CX - self.COL_HALF_X
        self._bumper_h  = self.BODY_H / 2 * 0.6

        # Appearance
        self._ap = ap

    # ── Public ────────────────────────────────────────────────────────────────

    def get_xml(self) -> str:
        env_tmpl   = _load_text(_ASSET_DIR / "env_base.xml")
        robot_tmpl = _load_text(_ASSET_DIR / "robot.xml")

        robot_xml  = self._fill_robot(robot_tmpl)
        wall_xml   = "\n".join(self._wall_geoms())
        obs_xml    = "\n".join(self._obstacle_geoms())

        xmin, xmax, ymin, ymax = self.bounds
        cx  = (xmin + xmax) / 2
        cy  = (ymin + ymax) / 2
        cd  = max(xmax - xmin, ymax - ymin) * 0.9
        fw  = (xmax - xmin) / 2 + 1.0
        fh  = (ymax - ymin) / 2 + 1.0

        xml = env_tmpl
        xml = xml.replace("{{CX}}",      f"{cx:.4f}")
        xml = xml.replace("{{CY}}",      f"{cy:.4f}")
        xml = xml.replace("{{FW}}",      f"{fw:.4f}")
        xml = xml.replace("{{FH}}",      f"{fh:.4f}")
        xml = xml.replace("{{CD}}",      f"{cd:.4f}")
        xml = xml.replace("{{CD_HALF}}", f"{cd*0.4:.4f}")
        xml = xml.replace("{{WALLS}}",     wall_xml)
        xml = xml.replace("{{OBSTACLES}}", obs_xml)
        xml = xml.replace("{{ROBOT}}",     robot_xml)
        return xml

    # ── Robot XML ─────────────────────────────────────────────────────────────

    def _fill_robot(self, tmpl: str) -> str:
        x, y, theta = self.robot_start
        ap = self._ap

        subs = {
            "{{RX}}":           f"{x:.4f}",
            "{{RY}}":           f"{y:.4f}",
            "{{RZ_J}}":         f"{self.ROBOT_JOINT_Z:.4f}",
            "{{THETA_DEG}}":    f"{math.degrees(theta):.4f}",
            "{{CX_CH}}":        f"{self.CHASSIS_CX:.4f}",
            "{{CY_CH}}":        f"{self._cy_ch:.4f}",
            "{{CY_WH}}":        f"{self._cy_wh:.4f}",
            "{{CY_LID_MT}}":    f"{self._lid_mt_z:.4f}",
            "{{CY_LID_DR}}":    f"{self._lid_dr_z:.4f}",
            "{{CY_LID_SITE}}":  f"{self._lid_site_z:.4f}",
            "{{COL_HX}}":       f"{self.COL_HALF_X:.4f}",
            "{{COL_HY}}":       f"{self.COL_HALF_Y:.4f}",
            "{{VIS_HX}}":       f"{self.BODY_LEN/2:.4f}",
            "{{VIS_HY}}":       f"{self.BODY_WID/2:.4f}",
            "{{BODY_H2}}":      f"{self.BODY_H/2:.4f}",
            "{{ROOF_HX}}":      f"{self.BODY_LEN/2*0.65:.4f}",
            "{{ROOF_HY}}":      f"{self.BODY_WID/2*0.80:.4f}",
            "{{ROOF_Z}}":       f"{self._cy_ch + self.BODY_H/2 + 0.018:.4f}",
            "{{BUMPER_FX}}":    f"{self._bumper_fx:.4f}",
            "{{BUMPER_RX}}":    f"{self._bumper_rx:.4f}",
            "{{BUMPER_H}}":     f"{self._bumper_h:.4f}",
            "{{WR}}":           f"{self.WHEEL_R:.4f}",
            "{{WHW}}":          f"{self.WHEEL_HW:.4f}",
            "{{WY}}":           f"{self.WHEEL_Y:.4f}",
            "{{HUB_Y}}":        f"{self._hub_y:.4f}",
            "{{L_WB}}":         f"{self.L:.4f}",
            "{{MAX_STEER_DEG}}":f"{self.MAX_STEER:.2f}",
            "{{RGBA_CHASSIS}}": _rgba(ap["chassis_rgba"]),
            "{{RGBA_ROOF}}":    _rgba(ap["roof_rgba"]),
            "{{RGBA_WHEEL}}":   _rgba(ap["wheel_rgba"]),
            "{{RGBA_HUB}}":     _rgba(ap["hub_rgba"]),
            "{{RGBA_AXLE}}":    _rgba(ap["axle_rgba"]),
            "{{RGBA_BF}}":      _rgba(ap["bumper_front_rgba"]),
            "{{RGBA_BR}}":      _rgba(ap["bumper_rear_rgba"]),
            "{{RGBA_LID_MT}}":  _rgba(ap["lidar_mount_rgba"]),
            "{{RGBA_LID_DR}}":  _rgba(ap["lidar_drum_rgba"]),
        }
        for token, value in subs.items():
            tmpl = tmpl.replace(token, value)
        return tmpl

    # ── Wall geoms ────────────────────────────────────────────────────────────

    def _wall_geoms(self):
        WALL_H = 0.40
        WALL_T = 0.05
        geoms  = []
        for i, (x1, y1, x2, y2) in enumerate(self.walls):
            cx  = (x1 + x2) / 2
            cy  = (y1 + y2) / 2
            cz  = WALL_H / 2
            length = math.hypot(x2-x1, y2-y1)
            if length < 1e-6:
                continue
            # *** euler is in DEGREES ***
            angle_deg = math.degrees(math.atan2(y2-y1, x2-x1))
            geoms.append(
                f'    <geom name="wall_{i}" type="box" '
                f'pos="{cx:.4f} {cy:.4f} {cz:.4f}" '
                f'size="{length/2:.4f} {WALL_T/2:.4f} {WALL_H/2:.4f}" '
                f'euler="0 0 {angle_deg:.6f}" '
                f'rgba="0.22 0.22 0.26 1" '
                f'contype="1" conaffinity="1" friction="0.8 0.05 0.01"/>'
            )
        return geoms

    def _obstacle_geoms(self):
        geoms = []
        for i, (cx, cy, hw, hh, height) in enumerate(self.obstacles):
            geoms.append(
                f'    <geom name="obs_{i}" type="box" '
                f'pos="{cx:.4f} {cy:.4f} {height/2:.4f}" '
                f'size="{hw:.4f} {hh:.4f} {height/2:.4f}" '
                f'rgba="0.55 0.18 0.18 1" '
                f'contype="1" conaffinity="1"/>'
            )
        return geoms


# ── Module-level exports (imported by keyboard_teleop) ────────────────────────
def _get_defaults():
    loader = MapLoader.__new__(MapLoader)
    loader.map_dict    = {"walls":[],"obstacles":[],"robot_start":[0,0,0],"bounds":(-1,5,-1,5)}
    loader.walls       = []
    loader.obstacles   = []
    loader.robot_start = [0,0,0]
    loader.bounds      = (-1,5,-1,5)
    with open(_CFG_DIR / "vehicle_config.yaml") as f:
        loader._vcfg = yaml.safe_load(f)
    loader._precompute()
    return loader

_defaults = _get_defaults()
COL_HALF_X   = _defaults.COL_HALF_X
COL_HALF_Y   = _defaults.COL_HALF_Y
ROBOT_JOINT_Z = _defaults.ROBOT_JOINT_Z
CHASSIS_CX    = _defaults.CHASSIS_CX