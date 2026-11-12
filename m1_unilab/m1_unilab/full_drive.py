"""M1 torque-PD drive with physical-step delay, sensors and persistent wrenches.

All state and force access uses public UniLab/UniSim interfaces. This owner is
used only by b2w-full; the original implicit-servo baseline stays available.
"""

from dataclasses import dataclass

import numpy as np
import torch
from m1_locomotion.contract import DEFAULT_POSITION, JOINTS, LEGS
from m1_locomotion.gains import load_gains
from unilab.managers.action_manager import ActionTerm, ActionTermCfg
from unisim.utils.math import quat_apply, quat_apply_inverse

from .full_rewards import reward_values

BODIES = ["BASE_LINK"] + [
    f"{leg}_{kind}_LINK" for leg in LEGS for kind in ("ABAD", "HIP", "KNEE", "FOOT")
]


@dataclass(kw_only=True)
class FullDriveCfg(ActionTermCfg):
    delay_min_steps: int = 0
    delay_max_steps: int = 20
    gain_scale_range: tuple[float, float] = (0.5, 2.0)
    force_range: tuple[float, float] = (-30.0, 30.0)
    torque_range: tuple[float, float] = (-10.0, 10.0)
    leg_abduction_scale: float = 0.125
    leg_flexion_scale: float = 0.25
    wheel_scale: float = 5.0
    target_clip: float = 100.0

    def build(self, env):
        return FullDrive(self, env)


class FullDrive(ActionTerm):
    requires_substep_state_feedback = True

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if str(env.device) != "cpu":
            raise ValueError("M1 native MuJoCo full drive requires CPU simulation")
        if not 0 <= cfg.delay_min_steps <= cfg.delay_max_steps:
            raise ValueError("Invalid physical-step delay range")
        self.entity = env.scene[cfg.entity_name]
        self.joint_ids, names = self.entity.find_joints(JOINTS, preserve_order=True)
        if names != JOINTS:
            raise ValueError("M1 joint order mismatch")
        self.q_indices = list(env.backend.get_joint_state_qpos_indices(JOINTS))
        self.v_indices = list(env.backend.get_joint_state_qvel_indices(JOINTS))
        self.root_indices = list(
            env.backend.get_root_state_layout("BASE_LINK").qpos_indices
        )
        self.root_v_indices = list(
            env.backend.get_root_state_layout("BASE_LINK").qvel_indices
        )
        self.base_backend_id = int(env.backend.get_body_ids(["BASE_LINK"])[0])
        self.com_offset = None
        self.default_q = torch.tensor(
            [DEFAULT_POSITION[n] for n in JOINTS], dtype=torch.float32
        )
        self.scale = torch.tensor(
            [
                cfg.leg_abduction_scale if "ABAD" in n else cfg.leg_flexion_scale
                for n in JOINTS[:12]
            ]
            + [cfg.wheel_scale] * 4
        )
        gains = load_gains("user_wheels")
        self.nominal_kp = torch.tensor(
            [gains[n]["kp"] for n in JOINTS], dtype=torch.float32
        )
        self.nominal_kd = torch.tensor(
            [gains[n]["kd"] for n in JOINTS], dtype=torch.float32
        )
        self.kp = self.nominal_kp.expand(env.num_envs, -1).clone()
        self.kd = self.nominal_kd.expand(env.num_envs, -1).clone()
        self.torque_limit = torch.tensor([180.0] * 12 + [28.68] * 4)
        self.delay_steps = torch.zeros((env.num_envs, 16), dtype=torch.long)
        self.force_local = torch.zeros((env.num_envs, 1, 3))
        self.torque_local = torch.zeros_like(self.force_local)
        base_local, _ = self.entity.find_bodies("BASE_LINK")
        _, self.wrench_body_ids = self.entity.bind_body_wrench(
            base_local, term_name="M1 persistent wrench", torque=True
        )
        limits = torch.tensor(
            env.backend.get_joint_range(names=JOINTS), dtype=torch.float32
        )[:12]
        mid = limits.mean(-1)
        half = (limits[:, 1] - limits[:, 0]) * 0.45
        self.soft_limits = torch.stack((mid - half, mid + half), dim=-1)
        self.symmetry_signs = torch.tensor([1.0] * 6 + [-1.0] * 6)
        names = (
            ("gyro", "local_linvel")
            + tuple(f"tau_{n}" for n in JOINTS)
            + tuple(f"contact_{n}" for n in BODIES)
        )
        self.sensors = env.backend.bind_sensor_data(names)
        assert tuple(self.sensors.dimensions) == (3, 3) + (1,) * 16 + (3,) * 17
        n = env.num_envs
        self._raw = torch.zeros((n, 16))
        self.targets = self.default_q.expand(n, -1).clone()
        self.delayed_targets = self.targets.clone()
        self.commanded_torque = torch.zeros_like(self._raw)
        self.delay_history = self.targets.expand(cfg.delay_max_steps + 1, n, 16).clone()
        self.delay_head = 0
        self.rows = torch.arange(n)[:, None]
        self.columns = torch.arange(16)[None, :]
        self.force_history = torch.zeros((n, 3, 17, 3))
        self.contact_age = torch.zeros((n, 17))
        self.previous_velocity = torch.zeros((n, 16))
        self.reset_pending = torch.ones(n, dtype=torch.bool)
        self.physics_tick = 0
        self.sampled_tick = -1
        self.wrench_applications = 0
        self.snapshot_data = None
        self.reward_cache = None

    @property
    def action_dim(self):
        return 16

    @property
    def raw_action(self):
        return self._raw

    def process_actions(self, actions):
        self._raw.copy_(actions)
        self._action_finite_valid.copy_(torch.isfinite(actions).all())
        self.publish_validation_verdict()
        self.targets.copy_(
            (actions * self.scale + self.default_q).clamp(
                -self.cfg.target_clip, self.cfg.target_clip
            )
        )
        self.reward_cache = None

    def randomize(self, env_ids, rng):
        """Called exclusively by the reset EventManager term."""
        ids = torch.arange(self.num_envs) if env_ids is None else env_ids
        shape = (ids.numel(), 16)
        self.kp[ids] = self.nominal_kp * rng.uniform(*self.cfg.gain_scale_range, shape)
        self.kd[ids] = self.nominal_kd * rng.uniform(*self.cfg.gain_scale_range, shape)
        self.delay_steps[ids] = rng.integers(
            self.cfg.delay_min_steps, self.cfg.delay_max_steps + 1, shape
        )
        self.force_local[ids] = rng.uniform(*self.cfg.force_range, (ids.numel(), 1, 3))
        self.torque_local[ids] = rng.uniform(
            *self.cfg.torque_range, (ids.numel(), 1, 3)
        )

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs) if env_ids is None else env_ids
        self._raw[ids] = 0
        self.targets[ids] = self.default_q
        self.delayed_targets[ids] = self.default_q
        self.delay_history[:, ids] = self.default_q
        self.commanded_torque[ids] = 0
        self.force_history[ids] = 0
        self.contact_age[ids] = 0
        self.reset_pending[ids] = True
        self.reward_cache = None

    def capture_physics(self):
        """Capture each completed physical step once, including the final substep."""
        advance = self.sampled_tick != self.physics_tick
        if not advance and not bool(self.reset_pending.any()):
            return self.snapshot_data
        views = self._env.backend.get_state_views(("qpos", "qvel"), device="cpu")
        q = views["qpos"][:, self.q_indices].float()
        qd = views["qvel"][:, self.v_indices].float()
        quat = views["qpos"][:, self.root_indices[3:7]].float()
        if self.com_offset is None:
            self.com_offset = torch.tensor(
                self._env.backend.get_body_ipos(env_ids=np.arange(self.num_envs))[
                    :, self.base_backend_id
                ],
                dtype=torch.float32,
            )
        omega = views["qvel"][:, self.root_v_indices[3:6]].float()
        velocity = quat_apply_inverse(
            quat, views["qvel"][:, self.root_v_indices[:3]].float()
        )
        velocity = velocity + torch.linalg.cross(omega, self.com_offset)
        values = torch.as_tensor(
            np.array(self.sensors.read(), copy=True), dtype=torch.float32
        )
        forces = values[:, 22:].reshape(self.num_envs, 17, 3)
        qdd = (qd - self.previous_velocity) / self._env.cfg.sim_dt
        if advance:
            self.force_history[:, 2].copy_(self.force_history[:, 1])
            self.force_history[:, 1].copy_(self.force_history[:, 0])
            self.force_history[:, 0].copy_(forces)
            contact = torch.linalg.vector_norm(forces, dim=-1) > 1.0
            self.contact_age.copy_(
                torch.where(contact, self.contact_age + self._env.cfg.sim_dt, 0.0)
            )
        qdd[self.reset_pending] = 0
        self.force_history[self.reset_pending] = 0
        self.contact_age[self.reset_pending] = 0
        self.previous_velocity.copy_(qd)
        self.reset_pending.zero_()
        gravity = quat_apply_inverse(
            quat, torch.tensor([0.0, 0.0, -1.0]).expand(self.num_envs, 3)
        )
        self.snapshot_data = {
            "q": q,
            "qd": qd,
            "qdd": qdd,
            "tau": values[:, 6:22],
            "gravity": gravity,
            "lin_vel": velocity,
            "ang_vel": omega,
            "contact_forces": self.force_history,
            "root_quat": quat,
            "first_contact": (self.contact_age > 0)
            & (self.contact_age < self._env.cfg.ctrl_dt + 1e-8),
        }
        self.sampled_tick = self.physics_tick
        return self.snapshot_data

    def apply_actions(self):
        state = self.capture_physics()
        self.delay_history[self.delay_head].copy_(self.targets)
        indices = (self.delay_head - self.delay_steps) % self.delay_history.shape[0]
        self.delayed_targets.copy_(self.delay_history[indices, self.rows, self.columns])
        self.delay_head = (self.delay_head + 1) % self.delay_history.shape[0]
        position_error = self.delayed_targets - state["q"]
        velocity_target = torch.zeros_like(state["qd"])
        velocity_target[:, 12:] = self.delayed_targets[:, 12:]
        torque = self.kp * position_error + self.kd * (velocity_target - state["qd"])
        self.commanded_torque.copy_(torque.clamp(-self.torque_limit, self.torque_limit))
        self.entity.set_joint_effort_target(
            self.commanded_torque, joint_ids=self.joint_ids
        )
        quat = state["root_quat"][:, None, :]
        self.entity.apply_body_wrench_to_sim(
            quat_apply(quat, self.force_local),
            quat_apply(quat, self.torque_local),
            self.wrench_body_ids,
            term_name="M1 persistent wrench",
        )
        self.wrench_applications += 1
        self.physics_tick += 1
        self.reward_cache = None

    def reward_values(self):
        if self.reward_cache is None:
            state = dict(self.capture_physics())
            state["command"] = self._env.command_manager.get_command("twist")
            state["action"] = self._env.action_manager.action
            state["last_action"] = self._env.action_manager.prev_action
            self.reward_cache = reward_values(
                state, self.default_q, self.soft_limits, self.symmetry_signs
            )
        return self.reward_cache
