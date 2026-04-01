"""
MapLoader
=========
Converts a map dict → MuJoCo XML string.

Robot model
-----------
4-wheel Ackermann car matching the calibrated bicycle model (L=0.145 m).

  Local frame origin = rear-axle centre at floor level (z=0).
  Forward = +X,  Left = +Y,  Up = +Z.

  Geometry (all metres):
    Wheelbase          L = 0.145   (rear→front axle)
    Body length          = 0.232   (1.6 × L), half = 0.116
    Body width           = 0.145   (1.0 × L), half = 0.0725
    Body height          = 0.040,  half = 0.020
    Wheel radius         = 0.030
    Wheel half-width     = 0.009
    Wheel Y-offset       = ±0.0815 (body_half_w + wheel_half_w)
    Chassis centre       = (0.0725, 0, 0.052)  in body frame

Collision
---------
  Python-side axis-aligned box (OBB in body frame → AABB check in world frame).
  Dimensions: half_x=0.116  half_y=0.0905  (includes wheel width).
  Walls are 0.05 m thick boxes, axis-aligned, fixed to worldbody.

Wall euler angles
-----------------
  MuJoCo euler attribute is in DEGREES.  We convert atan2 output from
  radians → degrees before writing the XML.
"""

import math

# ── Calibrated wheelbase (must match motion_model.py) ────────────────────────
L = 0.145

# ── Derived robot dimensions ──────────────────────────────────────────────────
BODY_LEN     = L * 1.6          # 0.232 m
BODY_WID     = L * 1.0          # 0.145 m
BODY_H       = 0.040
WHEEL_R      = 0.030
WHEEL_HW     = 0.009            # wheel half-width
WHEEL_Y      = BODY_WID/2 + WHEEL_HW   # 0.0815 m
CHASSIS_CX   = L / 2            # 0.0725 m — chassis centre from rear axle
CHASSIS_Z    = WHEEL_R + BODY_H/2      # 0.052 m
WHEEL_Z      = WHEEL_R                  # 0.030 m
LIDAR_Z      = WHEEL_R + BODY_H + 0.025  # sits on top of chassis

# ── Collision half-extents in BODY frame (used by Python clamping) ────────────
# Exported so keyboard_teleop can import them
COL_HALF_X   = BODY_LEN / 2     # 0.1160 m
COL_HALF_Y   = WHEEL_Y + WHEEL_HW  # 0.0905 m  (includes wheels)

# ── Wall geometry ─────────────────────────────────────────────────────────────
WALL_HEIGHT    = 0.40
WALL_THICKNESS = 0.05

# ── Robot z for freejoint (rear-axle centre at floor = z=0, but freejoint
#    origin should sit so wheels just touch floor) ────────────────────────────
ROBOT_JOINT_Z = 0.001           # freejoint origin just above floor


class MapLoader:
    def __init__(self, map_dict: dict):
        self.map         = map_dict
        self.walls       = map_dict["walls"]
        self.obstacles   = map_dict.get("obstacles", [])
        self.robot_start = map_dict["robot_start"]
        self.bounds      = map_dict.get("bounds", (-1, 10, -1, 10))

    def get_xml(self) -> str:
        wall_geoms     = "\n".join(self._wall_geoms())
        obstacle_geoms = "\n".join(self._obstacle_geoms())
        robot_xml      = self._robot_xml()
        return self._xml_template(wall_geoms, obstacle_geoms, robot_xml)

    # ── Walls ─────────────────────────────────────────────────────────────────

    def _wall_geoms(self):
        geoms = []
        for i, (x1, y1, x2, y2) in enumerate(self.walls):
            cx  = (x1 + x2) / 2
            cy  = (y1 + y2) / 2
            cz  = WALL_HEIGHT / 2
            dx, dy  = x2 - x1, y2 - y1
            length  = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            # *** DEGREES — MuJoCo euler is in degrees, not radians ***
            angle_deg = math.degrees(math.atan2(dy, dx))
            geoms.append(
                f'<geom name="wall_{i}" type="box" '
                f'pos="{cx:.4f} {cy:.4f} {cz:.4f}" '
                f'size="{length/2:.4f} {WALL_THICKNESS/2:.4f} {WALL_HEIGHT/2:.4f}" '
                f'euler="0 0 {angle_deg:.6f}" '
                f'rgba="0.22 0.22 0.26 1" '
                f'contype="1" conaffinity="1" '
                f'friction="0.8 0.05 0.01"/>'
            )
        return geoms

    def _obstacle_geoms(self):
        geoms = []
        for i, (cx, cy, hw, hh, height) in enumerate(self.obstacles):
            geoms.append(
                f'<geom name="obs_{i}" type="box" '
                f'pos="{cx:.4f} {cy:.4f} {height/2:.4f}" '
                f'size="{hw:.4f} {hh:.4f} {height/2:.4f}" '
                f'rgba="0.55 0.18 0.18 1" '
                f'contype="1" conaffinity="1"/>'
            )
        return geoms

    # ── Robot ──────────────────────────────────────────────────────────────────

    def _robot_xml(self):
        x, y, theta = self.robot_start
        # euler in body tag is also DEGREES in MuJoCo
        theta_deg = math.degrees(theta)

        # ── Sub-body offsets ──────────────────────────────────────────────────
        # All positions are in the robot body frame:
        #   origin = rear-axle centre, z=ROBOT_JOINT_Z above floor
        # We shift all z values down by ROBOT_JOINT_Z so parts sit correctly.
        cy_chassis = CHASSIS_Z - ROBOT_JOINT_Z
        cy_wheel   = WHEEL_Z   - ROBOT_JOINT_Z

        # Front wheel hinge bodies (visual only — no physics joint,
        # steering angle is purely visual, driven from alpha_cmd display)
        # We use a fixed euler to tilt front wheels; the teleop will update
        # them each frame via set_front_steer() below.

        return f"""
        <!-- ═══════════════════════════════════════════════════════
             Robot body  — freejoint origin = rear-axle, z≈0
             Forward = +X   Left = +Y   Up = +Z
             ═══════════════════════════════════════════════════════ -->
        <body name="robot" pos="{x:.4f} {y:.4f} {ROBOT_JOINT_Z:.4f}"
              euler="0 0 {theta_deg:.4f}">
          <freejoint name="robot_joint"/>

          <!-- ── Chassis ─────────────────────────────────────── -->
          <!-- Collision geom: box exactly covering the car footprint.
               Positioned so it spans from rear to front axle plus overhang. -->
          <geom name="chassis_col" type="box"
                pos="{CHASSIS_CX:.4f} 0 {cy_chassis:.4f}"
                size="{COL_HALF_X:.4f} {COL_HALF_Y:.4f} {BODY_H/2:.4f}"
                rgba="0 0 0 0"
                contype="1" conaffinity="1"
                mass="0.001"
                friction="0.0 0.0 0.0"/>

          <!-- Visual chassis body (slightly narrower than collision, looks cleaner) -->
          <geom name="chassis_vis" type="box"
                pos="{CHASSIS_CX:.4f} 0 {cy_chassis:.4f}"
                size="{COL_HALF_X:.4f} {BODY_WID/2:.4f} {BODY_H/2:.4f}"
                rgba="0.12 0.60 0.12 1"
                contype="0" conaffinity="0"
                mass="1.0"/>

          <!-- ── Chassis detail: roof panel ────────────────── -->
          <geom name="roof" type="box"
                pos="{CHASSIS_CX:.4f} 0 {cy_chassis + BODY_H/2 + 0.018:.4f}"
                size="{COL_HALF_X*0.65:.4f} {BODY_WID/2*0.80:.4f} 0.014"
                rgba="0.08 0.45 0.08 1"
                contype="0" conaffinity="0" mass="0"/>

          <!-- ── Front bumper ───────────────────────────────── -->
          <geom name="bumper_front" type="box"
                pos="{CHASSIS_CX + COL_HALF_X:.4f} 0 {cy_chassis:.4f}"
                size="0.008 {BODY_WID/2:.4f} {BODY_H/2*0.6:.4f}"
                rgba="0.7 0.7 0.1 1"
                contype="0" conaffinity="0" mass="0"/>

          <!-- ── Rear bumper ────────────────────────────────── -->
          <geom name="bumper_rear" type="box"
                pos="{CHASSIS_CX - COL_HALF_X:.4f} 0 {cy_chassis:.4f}"
                size="0.008 {BODY_WID/2:.4f} {BODY_H/2*0.6:.4f}"
                rgba="0.7 0.1 0.1 1"
                contype="0" conaffinity="0" mass="0"/>

          <!-- ── Rear wheels (fixed, no steering) ──────────── -->
          <geom name="wheel_rl" type="cylinder"
                pos="0 {WHEEL_Y:.4f} {cy_wheel:.4f}"
                size="{WHEEL_R:.4f} {WHEEL_HW:.4f}"
                euler="90 0 0"
                rgba="0.08 0.08 0.08 1"
                contype="0" conaffinity="0" mass="0"/>
          <geom name="wheel_rr" type="cylinder"
                pos="0 {-WHEEL_Y:.4f} {cy_wheel:.4f}"
                size="{WHEEL_R:.4f} {WHEEL_HW:.4f}"
                euler="90 0 0"
                rgba="0.08 0.08 0.08 1"
                contype="0" conaffinity="0" mass="0"/>

          <!-- Rear wheel hub caps -->
          <geom name="hub_rl" type="cylinder"
                pos="0 {WHEEL_Y + WHEEL_HW + 0.001:.4f} {cy_wheel:.4f}"
                size="0.013 0.003" euler="90 0 0"
                rgba="0.6 0.6 0.6 1" contype="0" conaffinity="0" mass="0"/>
          <geom name="hub_rr" type="cylinder"
                pos="0 {-(WHEEL_Y + WHEEL_HW + 0.001):.4f} {cy_wheel:.4f}"
                size="0.013 0.003" euler="90 0 0"
                rgba="0.6 0.6 0.6 1" contype="0" conaffinity="0" mass="0"/>

          <!-- ── Front steering sub-body ────────────────────── -->
          <!-- Front axle body — rotates for steering visualisation.
               Euler Z = steering angle (degrees), updated each frame by teleop. -->
          <body name="front_axle" pos="{L:.4f} 0 {cy_wheel:.4f}"
                euler="0 0 0">
            <joint name="steer_joint" type="hinge" axis="0 0 1"
                   limited="true" range="-45 45" damping="100"/>

            <!-- Left front wheel -->
            <geom name="wheel_fl" type="cylinder"
                  pos="0 {WHEEL_Y:.4f} 0"
                  size="{WHEEL_R:.4f} {WHEEL_HW:.4f}"
                  euler="90 0 0"
                  rgba="0.08 0.08 0.08 1"
                  contype="0" conaffinity="0" mass="0"/>
            <!-- Right front wheel -->
            <geom name="wheel_fr" type="cylinder"
                  pos="0 {-WHEEL_Y:.4f} 0"
                  size="{WHEEL_R:.4f} {WHEEL_HW:.4f}"
                  euler="90 0 0"
                  rgba="0.08 0.08 0.08 1"
                  contype="0" conaffinity="0" mass="0"/>
            <!-- Front hub caps -->
            <geom name="hub_fl" type="cylinder"
                  pos="0 {WHEEL_Y + WHEEL_HW + 0.001:.4f} 0"
                  size="0.013 0.003" euler="90 0 0"
                  rgba="0.6 0.6 0.6 1" contype="0" conaffinity="0" mass="0"/>
            <geom name="hub_fr" type="cylinder"
                  pos="0 {-(WHEEL_Y + WHEEL_HW + 0.001):.4f} 0"
                  size="0.013 0.003" euler="90 0 0"
                  rgba="0.6 0.6 0.6 1" contype="0" conaffinity="0" mass="0"/>
            <!-- Front axle bar -->
            <geom name="axle_front" type="cylinder"
                  pos="0 0 0"
                  size="0.007 {WHEEL_Y:.4f}"
                  euler="90 0 0"
                  rgba="0.35 0.35 0.35 1"
                  contype="0" conaffinity="0" mass="0.001"/>
          </body>

          <!-- ── Rear axle bar (visual) ─────────────────────── -->
          <geom name="axle_rear" type="cylinder"
                pos="0 0 {cy_wheel:.4f}"
                size="0.007 {WHEEL_Y:.4f}"
                euler="90 0 0"
                rgba="0.35 0.35 0.35 1"
                contype="0" conaffinity="0" mass="0"/>

          <!-- ── Lidar ──────────────────────────────────────── -->
          <!-- Mount lidar slightly forward of chassis centre -->
          <geom name="lidar_mount" type="cylinder"
                pos="{CHASSIS_CX:.4f} 0 {cy_chassis + BODY_H/2 + 0.015:.4f}"
                size="0.010 0.012"
                rgba="0.25 0.25 0.25 1"
                contype="0" conaffinity="0" mass="0"/>
          <geom name="lidar_drum" type="cylinder"
                pos="{CHASSIS_CX:.4f} 0 {cy_chassis + BODY_H/2 + 0.036:.4f}"
                size="0.032 0.016"
                rgba="0.08 0.08 0.75 1"
                contype="0" conaffinity="0" mass="0.05"/>
          <site name="lidar_site"
                pos="{CHASSIS_CX:.4f} 0 {cy_chassis + BODY_H/2 + 0.055:.4f}"
                size="0.003"/>

        </body>
        """

    # ── Full XML ──────────────────────────────────────────────────────────────

    def _xml_template(self, wall_geoms, obstacle_geoms, robot_xml):
        xmin, xmax, ymin, ymax = self.bounds
        cx  = (xmin + xmax) / 2
        cy  = (ymin + ymax) / 2
        cd  = max(xmax - xmin, ymax - ymin) * 0.9

        return f"""
<mujoco model="slam_env">
  <option timestep="0.02" integrator="Euler"
          gravity="0 0 -9.81"/>

  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.5 0.5 0.5" specular="0 0 0"/>
    <rgba haze="0.08 0.18 0.28 1"/>
    <global offwidth="1280" offheight="960"/>
    <quality shadowsize="2048"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient"
             rgb1="0.72 0.82 1.0" rgb2="0.38 0.52 0.70"
             width="512" height="512"/>
    <texture name="floor_tex" type="2d" builtin="checker"
             rgb1="0.80 0.80 0.80" rgb2="0.90 0.90 0.90"
             width="512" height="512" mark="none"/>
    <material name="floor_mat" texture="floor_tex"
              texrepeat="6 6" reflectance="0.04" shininess="0.02"/>
  </asset>

  <worldbody>
    <light name="sun" directional="true"
           pos="{cx:.2f} {cy:.2f} 5.0" dir="-0.2 -0.2 -1"
           diffuse="0.88 0.88 0.85" specular="0.15 0.15 0.12"
           ambient="0.28 0.28 0.30" castshadow="true"/>
    <light name="fill" directional="true"
           pos="{cx:.2f} {cy:.2f} 3.0" dir="0.5 0.3 -1"
           diffuse="0.30 0.30 0.32" specular="0 0 0"
           ambient="0.08 0.08 0.10"/>

    <geom name="floor" type="plane"
          pos="{cx:.2f} {cy:.2f} 0"
          size="{(xmax-xmin)/2+1:.2f} {(ymax-ymin)/2+1:.2f} 0.01"
          material="floor_mat"
          contype="1" conaffinity="1"
          friction="0.5 0.05 0.01"/>

{wall_geoms}

{obstacle_geoms}

{robot_xml}

    <camera name="top_down"
            pos="{cx:.2f} {cy:.2f} {cd:.2f}"
            xyaxes="1 0 0 0 1 0"/>
    <camera name="follow_cam"
            pos="{cx:.2f} {cy:.2f} {cd*0.4:.2f}"
            xyaxes="1 0 0 0 0.5 1"/>
  </worldbody>

  <!-- Steer joint actuator — position-controlled, driven by teleop -->
  <actuator>
    <position name="steer_act" joint="steer_joint"
              kp="500" kv="50" forcerange="-50 50"/>
  </actuator>

</mujoco>
"""