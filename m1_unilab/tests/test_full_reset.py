"""Real reset transactions checked by independent native MuJoCo collisions."""

from contextlib import contextmanager

import mujoco
import numpy as np
import pytest
import torch
from m1_unilab.assets import PACKAGE_ROOT
from m1_unilab.config import configuration
from unilab.base.config_adapter import EnvConfigAdapter, create_env

CAPTURED_POSE = np.array(
    [
        -15.61335563659668,
        12.30860424041748,
        0.06552022695541382,
        0.8636407256126404,
        0.3275908827781677,
        0.2742544114589691,
        -0.2675693929195404,
    ]
)
VELOCITY = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


def fixture_root(env, env_ids, pose_range, velocity_range):
    """Deterministic replacement for only the sampled root event in tests."""
    ids = env_ids.detach().cpu().numpy()
    poses = getattr(
        env, "clearance_test_poses", np.tile(CAPTURED_POSE, (env.num_envs, 1))
    )
    robot = env.scene["robot"]
    robot.write_root_link_pose_to_sim(torch.as_tensor(poses[ids]), env_ids)
    robot.write_root_link_velocity_to_sim(
        torch.as_tensor(np.tile(VELOCITY, (len(ids), 1))), env_ids
    )


@contextmanager
def real_env(terrain, *, corrected, count=1, fixed=True, poses=None):
    cfg = configuration(terrain, profile="b2w-full", num_envs=count, device="cpu")
    override = EnvConfigAdapter(
        cfg, root_dir=PACKAGE_ROOT, algo_name="ppo"
    ).build_task_env_cfg_override()
    # Explicitly remove the event if the production task has since enabled it:
    # baseline must continue to exercise the pre-fix reset.
    events = {
        name: term
        for name, term in override["events"].items()
        if term.get("func") != "m1_unilab.full_reset.TerrainClearance"
    }
    if fixed:
        events["root"]["func"] = f"{__name__}.fixture_root"
    if corrected:
        ordered = {}
        for name, term in events.items():
            ordered[name] = term
            if name == "root":
                ordered["terrain_clearance"] = {
                    "func": "m1_unilab.full_reset.TerrainClearance",
                    "mode": "reset",
                    "params": {},
                }
        events = ordered
    override["events"] = events
    env = create_env(cfg, num_envs=count, env_cfg_override=override)
    if poses is not None:
        env.clearance_test_poses = np.asarray(poses, dtype=np.float64)
    try:
        yield env
    finally:
        env.close()


def native_collisions(env):
    """Independent collision oracle on the final committed public state."""
    model = env.backend.get_playback_model()
    data = mujoco.MjData(model)
    states = env.backend.get_physics_state().copy()
    result = []
    for state in states:
        mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(model, data)
        result.append(
            (data.ncon, min((contact.dist for contact in data.contact), default=1.0))
        )
    return states, np.asarray(result)


def test_captured_old_reset_penetrates_then_corrected_reset_clears_actual_hfield():
    with real_env("mixed", corrected=False) as old:
        old.reset(seed=41)
        before, contacts_before = native_collisions(old)
        old_rng = old.rng.get_state().clone()
    # Penetration estimates differ between native EPA and libccd MPR.
    # The regression is deep initial overlap becoming collision-free, not
    # equality to the old algorithm's particular contact distance.
    assert contacts_before[0, 0] > 0
    assert contacts_before[0, 1] < -0.08
    with real_env("mixed", corrected=True) as new:
        new.reset(seed=41)
        after, contacts_after = native_collisions(new)
        torch.testing.assert_close(new.rng.get_state(), old_rng)
    assert contacts_after[:, 1].min() >= 0
    # Full physics state = time, qpos (root starts at 1), qvel. Only root Z changes.
    np.testing.assert_array_equal(
        after[:, np.arange(after.shape[1]) != 3],
        before[:, np.arange(before.shape[1]) != 3],
    )
    assert after[0, 3] - before[0, 3] == pytest.approx(0.17042791076583985, abs=1e-7)


def test_flat_nominal_upright_pose_is_not_lifted():
    poses = [[0, 0, 0.54, 1, 0, 0, 0]]
    with real_env("flat", corrected=True, poses=poses) as env:
        env.reset()
        state, contacts = native_collisions(env)
    np.testing.assert_array_equal(
        state[0, 1:8], np.asarray(poses[0], dtype=state.dtype)
    )
    assert contacts[:, 1].min() >= 0


@pytest.mark.parametrize("terrain", ["flat", "mixed"])
def test_selected_reset_preserves_other_rows_pose_orientation_and_velocity(terrain):
    poses = np.tile([0, 0, 0.8, 1, 0, 0, 0], (4, 1)).astype(float)
    with real_env(terrain, corrected=True, count=4, poses=poses) as env:
        env.reset(seed=12)
        before, _ = native_collisions(env)
        # The fixture creates penetrating orientations in noncontiguous rows.
        poses[[1, 3], 2] = 0.05
        poses[[1, 3], 3:] = CAPTURED_POSE[3:]
        env.clearance_test_poses = poses
        env.reset(env_ids=torch.tensor([3, 1]))
        after, contacts = native_collisions(env)
        np.testing.assert_array_equal(after[[0, 2]], before[[0, 2]])
        np.testing.assert_array_equal(
            after[[1, 3], 1:3], poses[[1, 3], :2].astype(after.dtype)
        )
        np.testing.assert_array_equal(
            after[[1, 3], 4:8], poses[[1, 3], 3:].astype(after.dtype)
        )
        # MuJoCo stores free-joint angular velocity locally; the Entity API
        # receives world velocity. Convert back independently for the check.
        for row in (1, 3):
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, after[row, 4:8].astype(float))
            actual_world_velocity = np.r_[
                after[row, 24:27], rotation.reshape(3, 3) @ after[row, 27:30]
            ]
            np.testing.assert_allclose(
                actual_world_velocity, VELOCITY, rtol=0, atol=1e-7
            )
        assert (after[[1, 3], 3] > poses[[1, 3], 2]).all()
        assert contacts[:, 1].min() >= 0
        snapshot = after.copy()
        env.reset(env_ids=torch.tensor([], dtype=torch.int64))
        np.testing.assert_array_equal(env.backend.get_physics_state(), snapshot)


def test_randomized_root_event_keeps_samples_and_rng_stream():
    with real_env("mixed", corrected=False, count=24, fixed=False) as old:
        old.reset(seed=312)
        before, _ = native_collisions(old)
        old_rng = old.rng.get_state().clone()
    with real_env("mixed", corrected=True, count=24, fixed=False) as new:
        new.reset(seed=312)
        after, contacts = native_collisions(new)
        torch.testing.assert_close(new.rng.get_state(), old_rng)
    mask = np.arange(after.shape[1]) != 3
    np.testing.assert_array_equal(after[:, mask], before[:, mask])
    assert (after[:, 3] >= before[:, 3]).all()
    assert contacts[:, 1].min() >= 0


def test_stair_edge_footprints_clear_high_neighboring_steps():
    # Level 4 stairs: the body's center can be below a neighboring tread while
    # its limbs overlap that tread. Check native contacts, not sampled heights.
    with real_env("mixed", corrected=False, count=16) as old:
        origins = old.scene.terrain.terrain_origins.detach().cpu().numpy()
        poses = []
        for column in (4, 5):
            for offset in (-1.25, -1.05, -0.95, -0.75, 0.75, 0.95, 1.05, 1.25):
                origin = origins[4, column]
                poses.append(
                    [origin[0] + offset, origin[1], origin[2] + 0.30, 1, 0, 0, 0]
                )
        old.clearance_test_poses = np.asarray(poses)
        old.reset(seed=23)
        before, old_contacts = native_collisions(old)
    assert old_contacts[:, 1].min() < -0.01
    with real_env("mixed", corrected=True, count=16, poses=poses) as new:
        new.reset(seed=23)
        after, contacts = native_collisions(new)
    assert contacts[:, 1].min() >= 0
    assert (after[:, 3] > before[:, 3]).any()
    np.testing.assert_array_equal(
        after[:, [1, 2, 4, 5, 6, 7]], before[:, [1, 2, 4, 5, 6, 7]]
    )
