"""M1 task terms using only public UniLab state/observation contracts."""

import numpy as np
import torch
from unilab.envs.mdp import joint_pos_rel
from unilab.managers.manager_base import ManagerTermBase


def leg_posture_l1(env, asset_cfg):
    """Select leg columns, preserving the leading environment dimension."""
    return joint_pos_rel(env, asset_cfg=asset_cfg).abs().sum(dim=-1)


class HeightScan(ManagerTermBase):
    """187 base-relative terrain heights for the privileged critic only.

    Native CPU hfield sampling is an explicit public backend capability, not
    a ray GPU dependency. The actor receives no terrain observation.
    """

    def __init__(self, cfg, env):
        super().__init__(env)
        self.scanner = None
        if env.cfg.scene.terrain is not None:
            xx, yy = np.meshgrid(
                np.linspace(-0.8, 0.8, 17), np.linspace(-0.5, 0.5, 11), indexing="ij"
            )
            offsets = np.column_stack((xx.ravel(), yy.ravel()))
            self.scanner = env.backend.bind_height_scanner(
                hfield_geom_id=env.backend.get_geom_id(
                    env.cfg.scene.terrain.geom_name or "floor"
                ),
                frame_body_id=int(env.backend.get_body_ids(["BASE_LINK"])[0]),
                offsets=offsets,
                alignment="yaw",
                output="height",
            )

    def __call__(self, env):
        # ObservationManager probes dimensions before compiling its read plan.
        if env.scene.read_plan is None:
            return torch.zeros(
                (env.num_envs, 187), dtype=torch.float32, device=env.device
            )
        base_z = env.scene.read_plan.body_tensor_view("robot", ("BASE_LINK",)).pos_w[
            :, 0, 2:3
        ]
        ground = (
            0.0
            if self.scanner is None
            else torch.as_tensor(
                self.scanner.scan(), dtype=torch.float32, device=env.device
            )
        )
        return (base_z.expand(-1, 187) - ground - 0.5).clamp(-1, 1)


class TerrainLevelsDistance(ManagerTermBase):
    """B2W-style per-episode progress curriculum, gated on completed episodes."""

    def __init__(self, cfg, env):
        super().__init__(env)
        layout = env.backend.get_root_state_layout("BASE_LINK")
        self.xy_indices = list(layout.qpos_indices[:2])

    def reset_scoped(self, env, env_ids):
        terrain = env.scene.terrain
        # Reset occurs outside the scene observation read phase. Consume the
        # public backend state-view contract rather than a closed read plan.
        base = env.backend.get_state_views(("qpos",), device=env.device)["qpos"][
            :, self.xy_indices
        ]
        origins = env.scene.env_origins(torch.device(env.device))
        distance = torch.linalg.vector_norm(
            base[env_ids] - origins[env_ids, :2], dim=-1
        )
        commands = env.command_manager.get_command("twist")[env_ids, :2]
        expected = (
            torch.linalg.vector_norm(commands, dim=-1) * env.cfg.max_episode_seconds
        )
        move_up = distance > 4.0
        move_down = (distance < 0.5 * expected) & ~move_up
        old = terrain.terrain_levels[env_ids]
        finished = env.episode_length_buf[env_ids] > 0
        levels = torch.where(
            finished, old + move_up.long() - move_down.long(), torch.zeros_like(old)
        )
        levels = levels.clamp(0, terrain.terrain_origins.shape[0] - 1)
        terrain.select_levels(env_ids, levels)
        return {"mean_level": levels.float().mean()}

    def __call__(self, env, env_ids):
        return self.reset_scoped(env, env_ids)
