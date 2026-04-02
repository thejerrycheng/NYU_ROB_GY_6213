"""
map_registry.py
===============
Loads map dicts from YAML files in maps/configs/.
Provides a registry for use by teleop, slam, and visualization scripts.

Usage
-----
    from slam_env.maps.map_registry import load_map, MAP_NAMES
    map_dict = load_map("simple_room")
"""

import math, yaml
from pathlib import Path

_CONFIG_DIR = Path(__file__).parent / "configs"

MAP_NAMES = ["simple_room", "l_shaped", "maze"]


def load_map(name: str) -> dict:
    """
    Load a map by name from maps/configs/<name>.yaml.
    Returns a map dict compatible with MapLoader.
    """
    path = _CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        available = [p.stem for p in _CONFIG_DIR.glob("*.yaml")]
        raise FileNotFoundError(
            f"Map '{name}' not found. Available: {available}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return _parse(raw)


def _parse(raw: dict) -> dict:
    """Convert YAML dict → map dict format expected by MapLoader."""
    rs = raw["robot_start"]
    b  = raw["bounds"]
    return {
        "name":        raw["name"],
        "description": raw.get("description", ""),
        "robot_start": [float(rs["x"]), float(rs["y"]), float(rs["theta"])],
        "bounds": (float(b["x_min"]), float(b["x_max"]),
                   float(b["y_min"]), float(b["y_max"])),
        "walls":     [tuple(w) for w in raw.get("walls", [])],
        "obstacles": [tuple(o) for o in raw.get("obstacles", [])],
    }


def list_maps() -> list[str]:
    return [p.stem for p in sorted(_CONFIG_DIR.glob("*.yaml"))]
