"""
keyboard_client.py  —  Keyboard controller (regular python, NOT mjpython)
=========================================================================
Sends keyboard commands to env_server.py over the "cmd" topic.
Run in a separate terminal while env_server is running.

  Terminal 1:  mjpython env_server.py --map maze --noise
  Terminal 2:  python   keyboard_client.py

Controls:
  ↑ / ↓   v_cmd  ±5 per tap   (+ forward, - reverse)
  ← / →   alpha_cmd ±5 per tap
  X        STOP
  R        RESET
  Q        quit

Uses pynput for keyboard capture (pip install pynput).
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from slam_env.env_ipc import EnvPublisher
from slam_env.utils.motion_model import V_M, V_C, DELTA_COEFFS, MAX_STEER

try:
    from pynput import keyboard as pk
except ImportError:
    print("  pip install pynput")
    sys.exit(1)

V_STEP=5.; ALPHA_STEP=5.
V_MAX=100.; V_MIN=-100.; ALPHA_MAX=100.; ALPHA_MIN=-100.

v_cmd    = 0.0
alpha_cmd = 0.0


def _print_cmd():
    vc = v_cmd
    vp = max(0.,V_M*vc+V_C) if vc>0 else (min(0.,-(V_M*abs(vc)+V_C)) if vc<0 else 0.)
    d  = math.degrees(float(np.clip(
        DELTA_COEFFS[0]*alpha_cmd**2+DELTA_COEFFS[1]*alpha_cmd+DELTA_COEFFS[2],
        -MAX_STEER, MAX_STEER)))
    print(f"\r  v_cmd={v_cmd:+6.1f}  α={alpha_cmd:+6.1f}"
          f"  v_phys={vp:+.4f}m/s  δ={d:+.1f}°     ", end="", flush=True)


def main():
    global v_cmd, alpha_cmd

    pub = EnvPublisher(topics=["cmd"])

    print("  Keyboard client ready.")
    print("  ↑/↓  v_cmd   ←/→  alpha   X stop   R reset   Q quit\n")

    def on_press(key):
        global v_cmd, alpha_cmd
        msg = None
        if   key == pk.Key.up:    v_cmd     = float(np.clip(v_cmd+V_STEP,       V_MIN,V_MAX));    msg={"v_cmd":v_cmd,"alpha_cmd":alpha_cmd}
        elif key == pk.Key.down:  v_cmd     = float(np.clip(v_cmd-V_STEP,       V_MIN,V_MAX));    msg={"v_cmd":v_cmd,"alpha_cmd":alpha_cmd}
        elif key == pk.Key.left:  alpha_cmd = float(np.clip(alpha_cmd-ALPHA_STEP,ALPHA_MIN,ALPHA_MAX)); msg={"v_cmd":v_cmd,"alpha_cmd":alpha_cmd}
        elif key == pk.Key.right: alpha_cmd = float(np.clip(alpha_cmd+ALPHA_STEP,ALPHA_MIN,ALPHA_MAX)); msg={"v_cmd":v_cmd,"alpha_cmd":alpha_cmd}
        elif hasattr(key,"char"):
            c = key.char
            if c in ('x','X'):   v_cmd=0.; alpha_cmd=0.; msg={"stop":True};  print("\n  [X] STOP", flush=True)
            elif c in ('r','R'): v_cmd=0.; alpha_cmd=0.; msg={"reset":True}; print("\n  [R] RESET", flush=True)
            elif c in ('q','Q'): msg={"quit":True}; print("\n  [Q] Quit", flush=True); return False
        if msg:
            pub.publish("cmd", msg)
            _print_cmd()

    with pk.Listener(on_press=on_press) as listener:
        listener.join()

    pub.close()


if __name__ == "__main__":
    main()
