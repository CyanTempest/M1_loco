"""B2W deadband, whole-map bounds and reset curriculum behavior."""

from types import SimpleNamespace

import pytest
import torch
from m1_locomotion.contract import JOINTS
from m1_unilab.full_mdp import (
    TerrainBounds,
    TerrainLevelsDistanceFull,
    ThresholdVelocityCommand,
    ThresholdVelocityCommandCfg,
    joint_pos_without_wheels,
)
from unilab.managers.rng import ManagerRng


class Scene(dict):
    read_plan = None


def test_wheel_positions_zeroed_in_contract_order_without_mutating_public_state():
    names = tuple(reversed(JOINTS))
    state = torch.arange(23 * 16, dtype=torch.float32).reshape(23, 16)
    before = state.clone()
    robot = SimpleNamespace(
        joint_names=names,
        data=SimpleNamespace(default_joint_pos=lambda device: torch.ones_like(state)),
    )
    scene = Scene(robot=robot)
    scene.read_plan = SimpleNamespace(
        joint_tensor_view=lambda asset: SimpleNamespace(joint_pos=state)
    )
    env = SimpleNamespace(scene=scene, num_envs=23, device="cpu")
    result = joint_pos_without_wheels(env)
    assert result.shape == (23, 16)
    expected = state[:, [names.index(name) for name in JOINTS]] - 1
    expected[:, 12:] = 0
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(state, before)


def command(vx, vy=0, *, standing=0, world=0):
    cfg = ThresholdVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(10.0, 10.0),
        heading_command=True,
        heading_control_stiffness=0.5,
        rel_heading_envs=1.0,
        rel_standing_envs=standing,
        rel_world_envs=world,
        ranges=ThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(vx, vx),
            lin_vel_y=(vy, vy),
            ang_vel_z=(-1.0, 1.0),
            heading=(1.0, 1.0),
        ),
    )
    env = SimpleNamespace(
        num_envs=7,
        device="cpu",
        rng=ManagerRng.seeded(3),
        scene=Scene(robot=SimpleNamespace(root_body_name="BASE_LINK")),
    )
    term = cfg.build(env)
    assert isinstance(term, ThresholdVelocityCommand)
    term.reset(torch.arange(env.num_envs))
    term._update_command()
    return term


@pytest.mark.parametrize(
    "vx,vy,zero",
    [
        (0.0, 0.0, True),
        (0.2, 0.0, True),
        (0.12, 0.16, True),
        (0.2001, 0.0, False),
        (0.16, 0.16, False),
    ],
)
def test_planar_threshold_keeps_heading_control(vx, vy, zero):
    term = command(vx, vy)
    expected = torch.zeros(7, 2) if zero else torch.tensor([[vx, vy]] * 7)
    torch.testing.assert_close(term.command[:, :2], expected)
    torch.testing.assert_close(term.command[:, 2], torch.full((7,), 0.5))


def test_standing_and_world_frame_parent_logic_remain_active():
    assert not command(0.8, standing=1).command.any()
    term = command(0.1, world=1)
    term._heading = lambda: torch.full((7,), 0.7)
    term._update_command()
    assert not term.command[:, :2].any()  # cached world command was thresholded too


class Terrain:
    is_curriculum = True

    def __init__(self, count):
        self.terrain_origins = torch.zeros(5, 6, 3)
        self.terrain_origins[:, :, 0] = torch.arange(5)[:, None] * 8 - 16
        self.terrain_origins[:, :, 1] = torch.arange(6)[None, :] * 8 - 20
        self.terrain_types = torch.arange(count) % 6
        self.terrain_levels = torch.zeros(count, dtype=torch.int64)
        self.env_origins = self.terrain_origins[0, self.terrain_types].clone()

    def select_levels(self, env_ids, levels):
        self.terrain_levels[env_ids] = levels
        self.env_origins[env_ids] = self.terrain_origins[
            levels, self.terrain_types[env_ids]
        ]


def terrain_env(count=6):
    terrain = Terrain(count)
    scene = Scene()
    scene.terrain = terrain
    scene.env_origins = lambda device: terrain.env_origins
    state = torch.zeros(count, 9)
    backend = SimpleNamespace(
        get_root_state_layout=lambda name: SimpleNamespace(qpos_indices=(2, 4, 5)),
        get_state_views=lambda fields, device: {"qpos": state},
    )
    commands = torch.zeros(count, 3)
    env = SimpleNamespace(
        num_envs=count,
        device="cpu",
        scene=scene,
        backend=backend,
        rng=ManagerRng.seeded(17),
        episode_length_buf=torch.ones(count),
        cfg=SimpleNamespace(
            max_episode_seconds=20.0,
            scene=SimpleNamespace(
                terrain=SimpleNamespace(
                    generator=SimpleNamespace(
                        size=(8.0, 8.0),
                        border_width=4.0,
                        num_rows=5,
                        num_cols=999,
                    )
                )
            ),
        ),
        command_manager=SimpleNamespace(get_command=lambda name: commands),
    )
    return env, state, commands


def test_whole_map_bounds_use_real_columns_and_include_border():
    env, state, _ = terrain_env()
    state[:, [2, 4]] = torch.tensor(
        [
            [0.0, 0.0],
            [4.1, 4.1],
            [21.0, 25.0],
            [21.01, 0.0],
            [0.0, -25.01],
            [-20.9, -24.9],
        ]
    )
    term = TerrainBounds(SimpleNamespace(params={}), env)
    assert term(env).tolist() == [False, False, False, True, True, False]
    term.reset(torch.tensor([0, 3]))


def test_plane_has_no_boundary_and_invalid_buffer_is_rejected():
    env, state, _ = terrain_env()
    with pytest.raises(ValueError, match="interior"):
        TerrainBounds(SimpleNamespace(params={"distance_buffer": 24.0}), env)
    env.cfg.scene.terrain = None
    state[:] = 1e9
    assert not TerrainBounds(SimpleNamespace(params={}), env)(env).any()


def test_distance_curriculum_cold_reset_thresholds_and_subset_updates():
    env, state, commands = terrain_env()
    terrain = env.scene.terrain
    terrain.select_levels(torch.arange(6), torch.tensor([2, 2, 2, 2, 2, 2]))
    state[:, [2, 4]] = terrain.env_origins[:, :2]
    state[:, 2] += torch.tensor([5.0, 5.0, 4.0, 1.0, 0.0, 5.0])
    commands[:, 0] = torch.tensor([1.0, 1.0, 0.3, 0.3, 0.0, 1.0])
    env.episode_length_buf[0] = 0
    term = TerrainLevelsDistanceFull(SimpleNamespace(params={}), env)
    term.reset_scoped(env, torch.tensor([0, 1, 2, 3, 4]))
    # Cold->0; success up even if below expected distance; strict d>4;
    # failure down; zero-command stationary stays; unselected row untouched.
    assert terrain.terrain_levels.tolist() == [0, 3, 2, 1, 2, 2]
    torch.testing.assert_close(
        terrain.env_origins,
        terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types],
    )
    term.reset(torch.arange(6))


def test_overflow_randomizes_valid_rows_with_manager_rng_and_replays_seed():
    def run():
        env, state, _ = terrain_env(256)
        terrain = env.scene.terrain
        ids = torch.arange(256)
        terrain.select_levels(ids, torch.full((256,), 4))
        state[:, [2, 4]] = terrain.env_origins[:, :2]
        state[:, 2] += 5
        term = TerrainLevelsDistanceFull(SimpleNamespace(params={}), env)
        term(env, ids)
        return terrain.terrain_levels

    levels = run()
    assert set(levels.tolist()) == set(range(5))
    torch.testing.assert_close(levels, run())


def test_initial_full_reset_is_level_zero_and_underflow_clamps():
    env, state, commands = terrain_env()
    env.episode_length_buf.zero_()
    state[:, 2] = 1000
    term = TerrainLevelsDistanceFull(SimpleNamespace(params={}), env)
    term(env, None)
    assert not env.scene.terrain.terrain_levels.any()
    env.episode_length_buf.fill_(1)
    state[:, [2, 4]] = env.scene.terrain.env_origins[:, :2]
    commands[:, 0] = 1
    term(env, None)
    assert not env.scene.terrain.terrain_levels.any()
    term(env, torch.empty(0, dtype=torch.int64))
