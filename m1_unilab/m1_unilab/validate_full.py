"""Validate the native full profile without constructing a learner.

Committed per-environment model values are read through the public backend
``get_reset_term_default`` API. Playback models contain static asset properties,
not runtime DR. A temporary, call-through probe at two public backend boundaries
checks that every physical step receives the drive's wrench and torque control;
it never reads backend-private fields or substitutes simulated results.
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from m1_locomotion.contract import JOINTS
from unilab.base.config_adapter import EnvConfigAdapter, create_env
from unisim.utils.math import quat_apply

from .assets import PACKAGE_ROOT
from .config import configuration
from .full_drive import BODIES
from .full_events import MATERIAL_MAPPING, restitution_to_damping_ratio
from .full_rewards import REWARD_NAMES

MODEL_FIELDS = (
    "geom_friction",
    "geom_solref",
    "body_mass",
    "body_inertia",
    "body_ipos",
)
DRIVE_FIELDS = ("kp", "kd", "delay_steps", "force_local", "torque_local")


def committed_model_fields(backend, num_envs):
    """Read committed mjbatch fields, rejecting an unmaterialized model table."""
    values = {}
    for name in MODEL_FIELDS:
        value = np.array(backend.get_reset_term_default(name), copy=True)
        assert value.shape[0] == num_envs, (name, value.shape, num_envs)
        assert np.isfinite(value).all(), name
        values[name] = value
    return values


def check_observations(obs, num_envs):
    assert obs["obs"].shape == (num_envs, 57), {
        k: tuple(v.shape) for k, v in obs.items()
    }
    assert obs["critic"].shape == (num_envs, 247)
    assert all(bool(torch.isfinite(value).all()) for value in obs.values())


def check_drive_ranges(drive):
    assert bool((drive.kp[:, 12:] == 0).all()), (
        "Wheels must have zero position stiffness"
    )
    assert bool(
        (
            (drive.kp[:, :12] / drive.nominal_kp[:12] >= 0.5)
            & (drive.kp[:, :12] / drive.nominal_kp[:12] <= 2.0)
        ).all()
    )
    assert bool(
        (
            (drive.kd / drive.nominal_kd >= 0.5) & (drive.kd / drive.nominal_kd <= 2.0)
        ).all()
    )
    assert drive.delay_steps.dtype == torch.int64
    assert bool(
        (
            (drive.delay_steps >= drive.cfg.delay_min_steps)
            & (drive.delay_steps <= drive.cfg.delay_max_steps)
        ).all()
    )
    assert bool((drive.force_local.abs() <= 30).all())
    assert bool((drive.torque_local.abs() <= 10).all())
    assert all(
        bool(torch.isfinite(getattr(drive, name)).all()) for name in DRIVE_FIELDS
    )


def check_startup_dr(env, model):
    """Compare actual backend model parameters with both source and draws."""
    backend = env.backend
    values = committed_model_fields(backend, env.num_envs)
    robot = env.scene["robot"]
    geom_ids = np.array([backend.get_geom_id(name) for name in robot.geom_names])
    body_ids = np.array(backend.get_body_ids(BODIES))
    assert len(body_ids) == 17
    material = env.event_manager.get_term_cfg("materials").func
    mass = env.event_manager.get_term_cfg("mass_inertia").func
    friction = values["geom_friction"][:, geom_ids]
    solref = values["geom_solref"][:, geom_ids]
    body_mass = values["body_mass"][:, body_ids]
    inertia = values["body_inertia"][:, body_ids]
    np.testing.assert_allclose(friction, material.friction_values.numpy(), rtol=1e-7)
    np.testing.assert_allclose(solref, material.solref_values.numpy(), rtol=1e-7)
    assert material.bucket_values.shape == (64, 2)
    selected = material.bucket_values[material.bucket_ids]
    np.testing.assert_allclose(friction[..., 0], selected[..., 0].numpy())
    np.testing.assert_allclose(
        solref[..., 1], restitution_to_damping_ratio(selected[..., 1]).numpy()
    )
    assert np.all((friction[..., 0] >= 0.3) & (friction[..., 0] <= 1.0))
    np.testing.assert_allclose(friction[..., 1], 0.005)
    np.testing.assert_allclose(friction[..., 2], 0.0001)
    np.testing.assert_allclose(solref[..., 0] * solref[..., 1], 0.02, rtol=1e-6)
    assert np.all(model.geom_priority[geom_ids] == 1)
    np.testing.assert_allclose(body_mass, mass.mass_values.numpy(), rtol=1e-7)
    np.testing.assert_allclose(inertia, mass.inertia_values.numpy(), rtol=1e-7)
    original_mass = model.body_mass[body_ids]
    ratio = body_mass / original_mass
    base_offset = body_mass[:, 0] - original_mass[0]
    assert np.all((base_offset >= -1.0) & (base_offset <= 3.0))
    assert np.all((ratio[:, 1:] >= 0.7) & (ratio[:, 1:] <= 1.3))
    # Entity bindings and packed reset rows use float32; comparing against the
    # original XML's float64 mass/inertia needs its corresponding tolerance.
    np.testing.assert_allclose(
        inertia, model.body_inertia[body_ids][None] * ratio[..., None], rtol=1e-6
    )
    com_delta = values["body_ipos"][:, body_ids[0]] - model.body_ipos[body_ids[0]]
    assert np.all(np.abs(com_delta) <= 0.050001)
    assert np.any(np.abs(com_delta) > 1e-7), "COM randomization did not reach backend"
    assert np.any(np.abs(ratio - 1) > 1e-7), "Mass randomization did not reach backend"
    if env.num_envs > 1:
        for name in MODEL_FIELDS:
            assert np.any(values[name][0] != values[name][1]), (
                f"{name} is not per-environment"
            )
    return values, {
        "readback_api": "SimBackend.get_reset_term_default (live committed mjbatch fields)",
        "robot_bodies": len(body_ids),
        "robot_geoms": len(geom_ids),
        "material_buckets": 64,
        "sliding_friction_min_max": [
            float(friction[..., 0].min()),
            float(friction[..., 0].max()),
        ],
        "base_mass_offset_min_max_kg": [
            float(base_offset.min()),
            float(base_offset.max()),
        ],
        "other_body_mass_scale_min_max": [
            float(ratio[:, 1:].min()),
            float(ratio[:, 1:].max()),
        ],
        "base_com_offset_max_abs_m": float(np.abs(com_delta).max()),
        "mass_inertia_ratio_verified": True,
        "material_mapping": MATERIAL_MAPPING,
    }


def check_reset_lifecycle(env, initial_fields):
    drive = env.action_manager.get_term("drive")
    selected = torch.arange(0, env.num_envs, 2, device=env.device)
    unselected = torch.arange(1, env.num_envs, 2, device=env.device)
    for ids in (selected, None):
        before = {name: getattr(drive, name).clone() for name in DRIVE_FIELDS}
        obs, _ = env.reset(ids)
        check_observations(obs, env.num_envs)
        check_drive_ranges(drive)
        rows = torch.arange(env.num_envs) if ids is None else ids
        for name in DRIVE_FIELDS:
            current = getattr(drive, name)
            assert not torch.equal(current[rows], before[name][rows]), (
                f"reset did not resample {name}"
            )
            if ids is not None:
                assert torch.equal(current[unselected], before[name][unselected]), name
        for name, actual in committed_model_fields(env.backend, env.num_envs).items():
            np.testing.assert_array_equal(
                actual, initial_fields[name], err_msg=f"reset changed startup {name}"
            )
        assert bool((drive.raw_action[rows] == 0).all())
        assert bool((drive.delay_history[:, rows] == drive.default_q).all())


def check_reset_clearance(env, model):
    """Check the committed reset poses with independent native collision queries."""
    data = mujoco.MjData(model)
    states = env.backend.get_physics_state()
    contacts = 0
    for state in states:
        data.qpos[:] = state[1 : 1 + model.nq]
        mujoco.mj_forward(model, data)
        contacts += data.ncon
        assert all(contact.dist >= -1e-6 for contact in data.contact), (
            "A reset pose still penetrates the ground"
        )
    clearance = env.event_manager.get_term_cfg("terrain_clearance").func
    return {
        "collision_free_reset_verified": True,
        "native_contacts_after_reset": contacts,
        "maximum_reset_lift_m": float(clearance.last_lift.max()),
        "reset_clearance_m": clearance.clearance,
    }


class PublicStepProbe:
    """Observe calls into native physics without replacing its computation.

    Wrench verification covers submission to the public native backend boundary.
    The backend has no public live xfrc_applied accessor; this check does not claim
    to measure a solver-level force buffer or calibrated contact restitution.
    """

    def __init__(self, env, drive, actuator_ids):
        self.backend = env.backend
        self.drive = drive
        self.actuator_ids = actuator_ids
        self.steps = 0
        self.wrenches = 0
        self.pending = False

    def __enter__(self):
        self.original_interval = self.backend.apply_interval_randomization
        self.original_step = self.backend.step_tensor
        self.backend.apply_interval_randomization = self.interval
        self.backend.step_tensor = self.step
        return self

    def __exit__(self, *exc):
        self.backend.apply_interval_randomization = self.original_interval
        self.backend.step_tensor = self.original_step

    def interval(self, plan):
        ops = {op.term: op for op in plan.iter_ops()}
        if "body_force" in ops or "body_torque" in ops:
            assert {"body_force", "body_torque"}.issubset(ops)
            assert not self.pending, (
                "Two wrench submissions occurred before one physics step"
            )
            views = self.backend.get_state_views(("qpos",), device="cpu")
            quat = views["qpos"][:, self.drive.root_indices[3:7]].float()[:, None]
            for term, local in (
                ("body_force", self.drive.force_local),
                ("body_torque", self.drive.torque_local),
            ):
                op = ops[term]
                np.testing.assert_array_equal(
                    op.body_ids.cpu().numpy(), self.drive.wrench_body_ids
                )
                torch.testing.assert_close(op.payload, quat_apply(quat, local))
            self.wrenches += 1
            self.pending = True
        elif not plan.is_empty():
            assert not self.pending, (
                "Another interval plan would overwrite a pending wrench"
            )
        return self.original_interval(plan)

    def step(self, ctrl, nsteps=1):
        assert nsteps == 1, "Delayed PD must run once per physical step"
        assert self.pending, "A physical step is missing the persistent wrench"
        assert bool(torch.isfinite(ctrl).all())
        torch.testing.assert_close(
            ctrl[:, self.actuator_ids].float(), self.drive.commanded_torque
        )
        assert bool(
            (self.drive.commanded_torque.abs() <= self.drive.torque_limit + 1e-5).all()
        )
        self.pending = False
        result = self.original_step(ctrl, nsteps)
        actual = self.backend.get_state_views(("ctrl",), device="cpu")["ctrl"]
        torch.testing.assert_close(
            actual[:, self.actuator_ids].float(), self.drive.commanded_torque
        )
        self.steps += 1
        return result


def validate(terrain="mixed", num_envs=8, steps=100):
    if num_envs < 1 or steps < 1:
        raise ValueError("num_envs and steps must be positive")
    started = time.perf_counter()
    cfg = configuration(terrain, num_envs=num_envs, device="cpu", profile="b2w-full")
    override = EnvConfigAdapter(
        cfg, root_dir=PACKAGE_ROOT, algo_name="ppo"
    ).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=num_envs, env_cfg_override=override)
    try:
        obs, _ = env.reset()
        check_observations(obs, num_envs)
        assert env.action_manager.total_action_dim == 16
        assert env.scene["robot"].joint_names == tuple(JOINTS)
        assert tuple(env.reward_manager.active_terms) == REWARD_NAMES
        model = env.backend.get_playback_model()
        assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICIT
        assert model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_AUTORESET
        assert model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD
        assert np.isclose(model.opt.timestep, cfg.env.sim_dt)
        actuator_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_act")
            for name in JOINTS
        ]
        assert min(actuator_ids) >= 0
        np.testing.assert_allclose(model.actuator_gainprm[actuator_ids, 0], 1.0)
        np.testing.assert_allclose(model.actuator_biasprm[actuator_ids], 0.0)
        np.testing.assert_allclose(model.actuator_gear[actuator_ids, 0], 1.0)
        np.testing.assert_allclose(
            model.actuator_forcerange[actuator_ids[12:]],
            np.tile([-28.68, 28.68], (4, 1)),
        )
        fields, dr_result = check_startup_dr(env, model)
        drive = env.action_manager.get_term("drive")
        check_drive_ranges(drive)
        check_reset_lifecycle(env, fields)
        clearance_result = check_reset_clearance(env, model)
        ticks_before = drive.physics_tick
        applications_before = drive.wrench_applications
        substeps = round(float(cfg.env.ctrl_dt) / float(cfg.env.sim_dt))
        generator = torch.Generator().manual_seed(916)
        terminations = 0
        truncations = 0
        reward_sum = 0.0
        maxima = {name: 0.0 for name in REWARD_NAMES}
        with PublicStepProbe(env, drive, actuator_ids) as probe:
            for index in range(steps):
                actions = (
                    torch.zeros((num_envs, 16))
                    if index < 10
                    else torch.randn((num_envs, 16), generator=generator).clamp(-2, 2)
                    * 0.25
                )
                state = env.step(actions)
                check_observations(state.obs, num_envs)
                assert bool(torch.isfinite(state.reward).all())
                rewards = drive.reward_values()
                assert tuple(rewards) == REWARD_NAMES
                for name, value in rewards.items():
                    assert value.shape == (num_envs,), (name, value.shape)
                    assert bool(torch.isfinite(value).all()), name
                    maxima[name] = max(maxima[name], float(value.abs().max()))
                assert drive.force_history.shape == (num_envs, 3, 17, 3)
                assert drive.sampled_tick == drive.physics_tick
                check_drive_ranges(drive)
                terminations += int(state.terminated.sum())
                truncations += int(state.truncated.sum())
                reward_sum += float(state.reward.mean())
        assert probe.steps == probe.wrenches == steps * substeps
        assert drive.physics_tick - ticks_before == probe.steps
        assert drive.wrench_applications - applications_before == probe.steps
        for name, actual in committed_model_fields(env.backend, num_envs).items():
            np.testing.assert_array_equal(
                actual, fields[name], err_msg=f"rollout changed startup {name}"
            )
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {
            "profile": "b2w-full",
            "backend": "native MuJoCo/mjbatch",
            "terrain": terrain,
            "num_envs": num_envs,
            "policy_steps": steps,
            "physical_steps_per_environment": probe.steps,
            "policy_dim": 57,
            "critic_dim": 247,
            "action_dim": 16,
            "reward_terms": 17,
            "startup_dr": dr_result,
            **clearance_result,
            "startup_unchanged_after_selected_and_full_reset": True,
            "drive_resampled_on_reset_selected_rows_only": True,
            "wheel_kp": 0,
            "drive_delay_steps_min_max": [
                int(drive.delay_steps.min()),
                int(drive.delay_steps.max()),
            ],
            "wrench_submissions_at_public_backend_boundary": probe.wrenches,
            "wrench_verification_scope": "world-frame force+torque submitted before every native physical step; no private solver-buffer access",
            "native_ctrl_matches_explicit_pd": True,
            "contact_history_physical_steps": 3,
            "sim_dt": float(cfg.env.sim_dt),
            "ctrl_dt": float(cfg.env.ctrl_dt),
            "integrator": "implicit",
            "convex_collision_algorithm": "libccd",
            "terminations": terminations,
            "truncations": truncations,
            "reward_mean": reward_sum / steps,
            "raw_reward_abs_max": maxima,
            "elapsed_seconds": time.perf_counter() - started,
            "process_peak_rss_mib": float(
                rss / (1024**2 if sys.platform == "darwin" else 1024)
            ),
            "training_started": False,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", choices=("flat", "mixed"), default="mixed")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.terrain, args.num_envs, args.steps)
    output = (
        args.output
        or PACKAGE_ROOT / f"validation_full_{args.terrain}_{args.num_envs}.json"
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
