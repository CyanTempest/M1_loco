"""M1 physical model. Only kp/kd are selected from a reference gain profile."""
import json

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg

from .contract import DEFAULT_POSITION, LEG_JOINTS, ROOT, WHEEL_JOINTS
from .gains import load_gains


def robot_cfg(gain_profile: str = "user_wheels") -> ArticulationCfg:
    asset_path = ROOT / "assets/m1/m1_cad.urdf"
    if not asset_path.is_file():
        raise FileNotFoundError(f"Run python scripts/prepare_assets.py first: {asset_path}")
    prior = json.loads((ROOT / "config/parameter_sources.json").read_text())
    nominal = prior["m1_cad"]
    gains = load_gains(gain_profile)
    common = dict(min_delay=0, max_delay=20, friction=0.0, armature=0.0)
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(asset_path), fix_base=False, merge_fixed_joints=True,
            replace_cylinders_with_capsules=False, activate_contact_sensors=True,
            self_collision=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, retain_accelerations=False,
                linear_damping=0.0, angular_damping=0.0,
                max_linear_velocity=20.0, max_angular_velocity=100.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.54), joint_pos=DEFAULT_POSITION, joint_vel={".*": 0.0}
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "legs": DelayedPDActuatorCfg(
                joint_names_expr=LEG_JOINTS,
                effort_limit=nominal["leg_effort_limit_Nm"],
                effort_limit_sim=nominal["leg_effort_limit_Nm"],
                velocity_limit=nominal["leg_velocity_limit_rad_s"],
                velocity_limit_sim=nominal["leg_velocity_limit_rad_s"],
                stiffness={n: gains[n]["kp"] for n in LEG_JOINTS},
                damping={n: gains[n]["kd"] for n in LEG_JOINTS}, **common,
            ),
            "wheels": DelayedPDActuatorCfg(
                joint_names_expr=WHEEL_JOINTS,
                effort_limit=nominal["wheel_effort_limit_Nm"],
                effort_limit_sim=nominal["wheel_effort_limit_Nm"],
                velocity_limit=nominal["wheel_velocity_limit_rad_s"],
                velocity_limit_sim=nominal["wheel_velocity_limit_rad_s"],
                stiffness={n: gains[n]["kp"] for n in WHEEL_JOINTS},
                damping={n: gains[n]["kd"] for n in WHEEL_JOINTS}, **common,
            ),
        },
    )
