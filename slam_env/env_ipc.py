"""
env_ipc.py  —  Multi-topic IPC for the SLAM env
=================================================
Lightweight publish/subscribe over named FIFOs.
Each topic gets its own pipe — subscribers only open what they need.

Topics (by convention):
  "state"  — robot true state (published by env_server every step)
  "lidar"  — scan data        (published by env_server every step)
  "map"    — static map info  (published once at startup)
  "cmd"    — control command  (published by keyboard_client / controller)

Usage
-----
  # env_server (publisher side):
  pub = EnvPublisher()
  pub.publish("state", {...})
  pub.publish("lidar", {...})
  pub.close()

  # Any subscriber (regular python process):
  sub = EnvSubscriber("lidar", "state")   # subscribe to any topics you want
  while True:
      msg = sub.recv("lidar")             # blocks until next lidar frame
      # or:
      msgs = sub.recv_available()         # non-blocking, returns {topic: msg}

  # keyboard_client or controller (sends commands):
  pub2 = EnvPublisher(topics=["cmd"])
  pub2.publish("cmd", {"v_cmd": 40.0, "alpha_cmd": 10.0})
"""

import os, pickle, struct, threading, time
from pathlib import Path
import tempfile

_BASE = Path(tempfile.gettempdir()) / "slam_env_ipc"

# Topic → pipe filename
def _pipe_path(topic: str) -> Path:
    return _BASE / f"{topic}.fifo"


def _ensure_dir():
    _BASE.mkdir(exist_ok=True)


def _write_msg(fd, obj):
    """Length-prefix pickle frame. Returns False on broken pipe."""
    try:
        blob = pickle.dumps(obj, protocol=4)
        hdr  = struct.pack(">I", len(blob))
        fd.write(hdr + blob)
        fd.flush()
        return True
    except (BrokenPipeError, OSError):
        return False


def _read_exact(fd, n):
    buf = b""
    while len(buf) < n:
        chunk = fd.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _read_msg(fd):
    """Read one length-prefix frame. Returns None on EOF."""
    hdr = _read_exact(fd, 4)
    if hdr is None:
        return None
    n = struct.unpack(">I", hdr)[0]
    if n == 0:
        return None   # sentinel
    blob = _read_exact(fd, n)
    if blob is None:
        return None
    return pickle.loads(blob)


# ─────────────────────────────────────────────────────────────────────────────
class EnvPublisher:
    """
    Opens write-end of one or more topic pipes.
    Silently drops frames if no subscriber has connected yet.

    Parameters
    ----------
    topics : list of topic names to publish on.
             Default: ["state", "lidar", "map", "cmd"]
    """

    _DEFAULT_TOPICS = ["state", "lidar", "map", "cmd"]

    def __init__(self, topics=None):
        _ensure_dir()
        self._topics = topics or self._DEFAULT_TOPICS
        self._fds    = {}   # topic → file handle

        for t in self._topics:
            p = _pipe_path(t)
            # Remove stale pipe
            try: p.unlink()
            except FileNotFoundError: pass
            os.mkfifo(str(p))

        print(f"  [EnvPublisher] pipes ready in {_BASE}")
        print(f"  [EnvPublisher] topics: {self._topics}")
        print(f"  [EnvPublisher] opening pipes (will block until subscribers connect)...")
        print(f"  [EnvPublisher] start subscribers now, then press Enter to continue.")
        input()   # give user time to start subscribers

        # Open write-ends (blocks until each reader opens the other side)
        for t in self._topics:
            p  = _pipe_path(t)
            self._fds[t] = open(str(p), "wb", buffering=0)
        print(f"  [EnvPublisher] all subscribers connected.\n")

    def publish(self, topic: str, msg: dict):
        """Send msg on topic. Silently drops if topic not registered or pipe broken."""
        fd = self._fds.get(topic)
        if fd is None:
            return
        _write_msg(fd, msg)

    def close(self):
        for t, fd in self._fds.items():
            try:
                fd.write(struct.pack(">I", 0))   # sentinel
                fd.flush()
            except Exception:
                pass
            try: fd.close()
            except Exception: pass
        self._fds.clear()


# ─────────────────────────────────────────────────────────────────────────────
class EnvSubscriber:
    """
    Opens read-end of one or more topic pipes.
    Provides blocking recv() per topic and non-blocking recv_all().

    Parameters
    ----------
    *topics : topic names to subscribe to, e.g. EnvSubscriber("lidar", "state")
    """

    def __init__(self, *topics):
        if not topics:
            raise ValueError("Specify at least one topic")
        _ensure_dir()
        self._topics = list(topics)
        self._fds    = {}

        # Wait for pipes to exist (env_server creates them on startup)
        for t in self._topics:
            p     = _pipe_path(t)
            waited = 0
            while not p.exists():
                if waited == 0:
                    print(f"  [EnvSubscriber] waiting for '{t}' pipe from env_server...",
                          flush=True)
                time.sleep(0.25)
                waited += 0.25
                if waited > 60:
                    raise TimeoutError(f"Pipe '{t}' not created within 60s")
            self._fds[t] = open(str(p), "rb", buffering=0)
            print(f"  [EnvSubscriber] connected to '{t}'", flush=True)

    def recv(self, topic: str):
        """
        Block until next message on `topic`.
        Returns None on EOF / pipe closed.
        """
        fd = self._fds.get(topic)
        if fd is None:
            return None
        return _read_msg(fd)

    def recv_nowait(self, topic: str):
        """
        Non-blocking read attempt. Returns message dict or None.
        Uses O_NONBLOCK — does not block if no data is available.
        """
        fd = self._fds.get(topic)
        if fd is None:
            return None
        try:
            import fcntl, os as _os
            flags = fcntl.fcntl(fd.fileno(), fcntl.F_GETFL)
            fcntl.fcntl(fd.fileno(), fcntl.F_SETFL, flags | _os.O_NONBLOCK)
            msg = _read_msg(fd)
            fcntl.fcntl(fd.fileno(), fcntl.F_SETFL, flags)   # restore blocking
            return msg
        except (BlockingIOError, OSError):
            return None

    def close(self):
        for fd in self._fds.values():
            try: fd.close()
            except Exception: pass
        # Clean up pipes
        for t in self._topics:
            try: _pipe_path(t).unlink()
            except Exception: pass
        self._fds.clear()
