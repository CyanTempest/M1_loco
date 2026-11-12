"""Reuse the pinned upstream training/play scripts without editing their source."""
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROBOT_LAB_COMMIT = "09f6a9dfdf48f32f38bb851dfa3a7d44db32b270"
ISAAC_LAB_COMMIT = "37ddf626871758333d6ed89cf64ad702aef127d0"


def checked_checkout():
    value = os.environ.get("ROBOT_LAB_ROOT")
    if not value:
        raise RuntimeError("Set ROBOT_LAB_ROOT to the robot_lab v2.3.2 checkout")
    root = Path(value).resolve()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != ROBOT_LAB_COMMIT:
        raise RuntimeError(f"Expected robot_lab {ROBOT_LAB_COMMIT}; found {commit}")
    return root


def upstream(mode):
    root = checked_checkout()
    # This module only registers string entrypoints and does not import Isaac/Omniverse.
    import m1_locomotion.tasks  # noqa: F401
    folder = root / "scripts/reinforcement_learning/rsl_rl"
    sys.path.insert(0, str(root / "source/robot_lab"))
    sys.path.insert(0, str(folder))
    runpy.run_path(str(folder / f"{mode}.py"), run_name="__main__")
