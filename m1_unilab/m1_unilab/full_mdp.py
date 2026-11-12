"""B2W observation, command and terrain semantics on public UniLab contracts."""

import math
from dataclasses import dataclass

import torch
from m1_locomotion.contract import JOINTS, WHEEL_JOINTS
from unilab.base.reset_selectors import resolve_reset_env_ids
from unilab.envs.mdp import joint_pos_rel
from unilab.envs.mdp.commands.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from unilab.managers.manager_base import ManagerTermBase
from unilab.managers.rng import ManagerRng
from unilab.managers.scene_entity_config import SceneEntityCfg


class RootKinematics(ManagerTermBase):
    """Post-step root kinematics shared with the explicit drive/reward snapshot."""

    def __init__(self, cfg, env):
        super().__init__(env)
        if cfg.params.get("kind") not in ("ang_vel", "lin_vel", "gravity"):
            raise ValueError("Unknown root kinematic observation")

    def __call__(self, env, kind):
        if env.scene.read_plan is None:
            return torch.zeros(
                (env.num_envs, 3), dtype=torch.float32, device=env.device
            )
        return env.action_manager.get_term("drive").capture_physics()[kind].clone()

    def reset(self, env_ids=None):
        pass


def joint_pos_without_wheels(env):
    """Keep all 16 JOINTS columns; clear wheel positions before observation noise."""
    names = tuple(env.scene["robot"].joint_names)
    indices = [names.index(name) for name in JOINTS]
    asset_cfg = SceneEntityCfg("robot", joint_ids=indices)
    result = joint_pos_rel(env, asset_cfg=asset_cfg).clone()
    result[:, [JOINTS.index(name) for name in WHEEL_JOINTS]] = 0
    return result


class ThresholdVelocityCommand(UniformVelocityCommand):
    """Apply B2W's planar deadband, retaining the parent's command lifecycle."""

    tensor_body_names = ("BASE_LINK",)

    def _metric_velocities(self):
        state = self._env.action_manager.get_term("drive").capture_physics()
        return state["lin_vel"], state["ang_vel"]

    def _sample_full_width(self):
        samples = super()._sample_full_width()
        moving = torch.linalg.vector_norm(samples[:, :2], dim=-1) > 0.2
        samples[:, :2] *= moving[:, None]
        return samples


@dataclass(kw_only=True)
class ThresholdVelocityCommandCfg(UniformVelocityCommandCfg):
    def build(self, env):
        return ThresholdVelocityCommand(self, env)


class TerrainBounds(ManagerTermBase):
    """Whole generated-map boundary; configure the term with time_out=True.

    UniSim centers the generated grid at world zero. In curriculum mode its
    actual columns equal terrain types, not necessarily generator.num_cols.
    The materialized origin tensor provides both real grid dimensions.
    """

    def __init__(self, cfg, env):
        super().__init__(env)
        terrain_cfg = env.cfg.scene.terrain
        generator = None if terrain_cfg is None else terrain_cfg.generator
        self.half_extents = None
        self.xy_indices = None
        if generator is not None:
            rows, columns = env.scene.terrain.terrain_origins.shape[:2]
            self.half_extents = torch.tensor(
                [
                    rows * generator.size[0] / 2 + generator.border_width,
                    columns * generator.size[1] / 2 + generator.border_width,
                ],
                dtype=torch.float32,
                device=env.device,
            )
            layout = env.backend.get_root_state_layout("BASE_LINK")
            self.xy_indices = list(layout.qpos_indices[:2])
        self._validate_buffer(cfg.params.get("distance_buffer", 3.0))

    def _validate_buffer(self, distance_buffer):
        if not math.isfinite(distance_buffer) or distance_buffer < 0:
            raise ValueError(
                "TerrainBounds distance_buffer must be finite and non-negative"
            )
        if self.half_extents is not None and bool(
            (self.half_extents <= distance_buffer).any()
        ):
            raise ValueError(
                "TerrainBounds distance_buffer leaves no interior map area"
            )

    def __call__(self, env, distance_buffer=3.0):
        if self.half_extents is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._validate_buffer(distance_buffer)
        xy = env.backend.get_state_views(("qpos",), device=env.device)["qpos"][
            :, self.xy_indices
        ]
        return (xy.abs() > self.half_extents - distance_buffer).any(dim=-1)

    def reset(self, env_ids=None):
        """No episodic state; satisfy the manager's reset boundary."""


class TerrainLevelsDistanceFull(ManagerTermBase):
    """Distance curriculum with level-zero cold reset and random overflow rows."""

    def __init__(self, cfg, env):
        super().__init__(env)
        terrain = env.scene.terrain
        if not terrain.is_curriculum:
            raise ValueError(
                "TerrainLevelsDistanceFull requires generated curriculum terrain"
            )
        if not isinstance(env.rng, ManagerRng):
            raise TypeError("TerrainLevelsDistanceFull requires the manager-owned RNG")
        layout = env.backend.get_root_state_layout("BASE_LINK")
        self.xy_indices = list(layout.qpos_indices[:2])

    def reset_scoped(self, env, env_ids):
        env_ids = resolve_reset_env_ids(
            env_ids,
            num_envs=env.num_envs,
            device=env.device,
            owner="TerrainLevelsDistanceFull",
        )
        terrain = env.scene.terrain
        if not env_ids.numel():
            return {"mean_level": terrain.terrain_levels.float().mean()}
        # Reset is outside the packed observation-read phase.
        xy = env.backend.get_state_views(("qpos",), device=env.device)["qpos"][
            :, self.xy_indices
        ]
        origins = env.scene.env_origins(torch.device(env.device))
        distance = torch.linalg.vector_norm(xy[env_ids] - origins[env_ids, :2], dim=-1)
        commands = env.command_manager.get_command("twist")[env_ids, :2]
        expected = (
            torch.linalg.vector_norm(commands, dim=-1) * env.cfg.max_episode_seconds
        )
        move_up = distance > 4.0
        move_down = (distance < expected * 0.5) & ~move_up
        old = terrain.terrain_levels[env_ids]
        finished = env.episode_length_buf[env_ids] > 0
        levels = torch.where(
            finished, old + move_up.long() - move_down.long(), torch.zeros_like(old)
        ).clamp_min(0)
        rows = terrain.terrain_origins.shape[0]
        overflow = levels >= rows
        if bool(overflow.any()):
            random_levels = env.rng.integers(0, rows, (env_ids.numel(),))
            levels = torch.where(overflow, random_levels.to(levels.device), levels)
        terrain.select_levels(env_ids, levels)
        return {"mean_level": terrain.terrain_levels.float().mean()}

    def __call__(self, env, env_ids):
        return self.reset_scoped(env, env_ids)

    def reset(self, env_ids=None):
        """No extra episodic state; level changes happen only in reset_scoped."""
