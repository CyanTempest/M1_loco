"""Port of robot_lab's B2W task, pinned to v2.3.2 (see docs/research.md)."""
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_b2w.rough_env_cfg import (
    UnitreeB2WRoughEnvCfg,
)

from .assets import robot_cfg
from .contract import JOINTS, LEG_JOINTS, POSITION_SCALE, WHEEL_JOINTS, WHEEL_VELOCITY_SCALE
from .terrains import terrain_cfg


@configclass
class M1RoughEnvCfg(UnitreeB2WRoughEnvCfg):
    gain_profile: str = "user_wheels"
    base_link_name = "BASE_LINK"
    foot_link_name = ".*_FOOT_LINK"
    leg_joint_names = LEG_JOINTS
    wheel_joint_names = WHEEL_JOINTS
    joint_names = JOINTS

    def __post_init__(self):
        super().__post_init__()
        # 1 kHz explicit PD updates with 50 Hz policy; keep physical delay range 0-20 ms.
        self.sim.dt = 0.001
        self.decimation = 20
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.robot = robot_cfg(self.gain_profile)
        self.scene.num_envs = 2048
        self.scene.terrain.terrain_generator = terrain_cfg()
        self.scene.terrain.max_init_terrain_level = 1
        self.scene.terrain.physics_material.restitution = 0.0
        # Avoid remote decorative asset downloads for headless training.
        self.scene.terrain.visual_material = None
        self.scene.sky_light.spawn.texture_file = None

        # Select actual leg joints directly. Upstream wheel masking assumes global index order.
        for group in (self.observations.policy, self.observations.critic):
            group.joint_pos.func = mdp.joint_pos_rel
            group.joint_pos.params = {"asset_cfg": SceneEntityCfg(
                "robot", joint_names=LEG_JOINTS, preserve_order=True)}
            group.joint_vel.params = {"asset_cfg": SceneEntityCfg(
                "robot", joint_names=JOINTS, preserve_order=True)}
            group.base_ang_vel.scale = 0.25
            group.joint_pos.scale = 1.0
            group.joint_vel.scale = 0.05
        self.observations.critic.base_lin_vel.scale = 2.0
        # Actor 53D (proprioception only); critic 243D includes velocity and 187 height samples.
        # Keep critic height scan in flat training too, so checkpoints resume without dimension changes.
        self.actions.joint_pos.scale = POSITION_SCALE
        self.actions.joint_pos.clip = None  # wrapper clips raw action to +/-3 before scaling
        self.actions.joint_vel = mdp.JointVelocityActionCfg(
            asset_name="robot", joint_names=WHEEL_JOINTS,
            scale=WHEEL_VELOCITY_SCALE, use_default_offset=False,
            clip=None, preserve_order=True,
        )

        self.events.randomize_reset_base.params = {
            "pose_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25),
                           "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0),
                               "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
        }
        self.events.randomize_reset_joints.func = mdp.reset_joints_by_offset
        self.events.randomize_reset_joints.params = {
            "position_range": (-0.03, 0.03), "velocity_range": (-0.1, 0.1)}
        self.events.randomize_rigid_body_mass_base.params.update(
            mass_distribution_params=(0.85, 1.15), operation="scale")
        self.events.randomize_rigid_body_mass_others.params.update(mass_distribution_params=(0.85, 1.15))
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)}
        self.events.randomize_rigid_body_material.params.update(
            static_friction_range=(0.4, 1.2), dynamic_friction_range=(0.4, 1.0),
            restitution_range=(0.0, 0.05), make_consistent=True)
        self.events.randomize_actuator_gains.params.update(
            stiffness_distribution_params=(0.8, 1.2), damping_distribution_params=(0.8, 1.2))
        self.events.randomize_apply_external_force_torque.params.update(
            force_range=(-10.0, 10.0), torque_range=(-3.0, 3.0))
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}

        # B2W's literal joint-mirror names and forced gait reward are not valid for M1.
        self.rewards.joint_mirror.weight = 0.0
        self.rewards.action_mirror.weight = 0.0
        self.rewards.action_sync.weight = 0.0
        self.rewards.feet_gait.weight = 0.0
        # Wheel center velocity is not slip: retain upstream feet_slide weight=0.
        self.rewards.joint_pos_penalty.weight = -0.25
        self.rewards.contact_forces.params["threshold"] = 200.0
        self.rewards.joint_torques_wheel_l2.weight = -1e-4
        self.rewards.upward.weight = 1.0
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["BASE_LINK"]), "threshold": 10.0})
        self.terminations.bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.10
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.15, 0.15)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        # Parent does not prune for subclasses; prune once after all M1 overrides.
        self.disable_zero_weight_rewards()


@configclass
class M1FlatEnvCfg(M1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
        self.terminations.terrain_out_of_bounds = None
        self.events.randomize_push_robot = None


@configclass
class M1PreviewRoughEnvCfg(M1RoughEnvCfg):
    gain_profile: str = "preview"


@configclass
class M1PreviewFlatEnvCfg(M1FlatEnvCfg):
    gain_profile: str = "preview"
