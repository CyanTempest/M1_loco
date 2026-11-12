"""A2 leg kp/kd, authorized fallback for missing A2W data; user-defined wheel gains."""
import json
import math
from .contract import GAIN_PROFILES, JOINTS, LEG_JOINTS, ROOT, WHEEL_JOINTS


def load_gains(profile="user_wheels"):
    if profile not in GAIN_PROFILES:
        raise ValueError(f"Unknown gain profile: {profile}")
    legs = json.loads((ROOT / "config/a2_leg_gains.json").read_text())
    path = ROOT / "config" / ("wheel_gains.json" if profile == "user_wheels" else "preview_wheel_gains.json")
    wheels = json.loads(path.read_text())
    if set(legs["motors"]) != set(LEG_JOINTS) or set(wheels["motors"]) != set(WHEEL_JOINTS):
        raise ValueError("A2 leg and user wheel gain files must contain exactly their respective joint groups")
    motors = {**legs["motors"], **wheels["motors"]}
    if set(motors) != set(JOINTS):
        raise ValueError(f"Gain file must specify exactly all 16 M1 joint names: {path}")
    for name, motor in motors.items():
        for key in ("kp", "kd"):
            value = motor[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Missing/invalid {profile} {name}.{key}. Fill your wheel gains in {path}; "
                                 "Preview tasks use separate test-only wheel gains.")
        if motor["kd"] <= 0.0:
            raise ValueError(f"{name}: kd must be positive")
    if not legs.get("source") or not wheels.get("source"):
        raise ValueError("Record the gain source")
    return motors
