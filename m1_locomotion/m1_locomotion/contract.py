"""Simulator-independent policy contract. All coordinates here are URDF coordinates."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGS = ("FAR", "FBL", "RAR", "RBL")  # FR, FL, RR, RL (SDK leg slots 0, 1, 2, 3)
LEG_JOINTS = [f"{leg}_{kind}_JOINT" for leg in LEGS for kind in ("ABAD", "HIP", "KNEE")]
WHEEL_JOINTS = [f"{leg}_FOOT_JOINT" for leg in LEGS]
JOINTS = LEG_JOINTS + WHEEL_JOINTS
WHEEL_LINKS = [f"{leg}_FOOT_LINK" for leg in LEGS]
DEFAULT_POSITION = {
    # Symmetric front/rear stance in M1 URDF coordinates, verified in CPU physics.
    # These are NOT SDK motor coordinates or a copied A2 default pose.
    name: (0.0 if "ABAD" in name or "FOOT" in name else
           (-1.0 if name.startswith("R") else 1.0) * (0.65 if "HIP" in name else -1.30))
    for name in JOINTS
}
POSITION_SCALE = {name: 0.15 if "ABAD" in name else 0.35 for name in LEG_JOINTS}
WHEEL_VELOCITY_SCALE = 5.0  # rad/s per action; robot_lab B2W reference
ACTION_CLIP = 3.0
POLICY_DT = 0.02
WHEEL_RADIUS = 0.0975
GAIN_PROFILES = ("user_wheels", "preview")

# No wheel angle: continuous wheel rotations must not change the observation.
ACTOR_TERMS = (("base_ang_vel", 3), ("projected_gravity", 3), ("velocity_commands", 3),
               ("leg_pos_relative", 12), ("all_joint_velocity", 16), ("previous_action", 16))
ACTOR_DIM = sum(n for _, n in ACTOR_TERMS)


def action_targets(action):
    """Reference transform for tests/Sim2Sim, never writes hardware."""
    import math
    if len(action) != 16 or not all(math.isfinite(float(x)) for x in action):
        raise ValueError("Expected 16 finite actions")
    a = [max(-ACTION_CLIP, min(ACTION_CLIP, float(x))) for x in action]
    q = [DEFAULT_POSITION[n] + POSITION_SCALE[n] * a[i] for i, n in enumerate(LEG_JOINTS)]
    dq = [WHEEL_VELOCITY_SCALE * x for x in a[12:]]
    return q, dq
