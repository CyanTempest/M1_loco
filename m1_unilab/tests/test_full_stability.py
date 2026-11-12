"""Regressions from the actual 1 ms high-contact reset failure.

The fixture records reset batch 11, environment 312: initial public reset
state, all committed material/inertial DR, and 97 native-step controls/wrenches.
Its recorded trajectory contains the original nonfinite result. Replay inputs
remain unchanged; the current production configuration supplies the model and
solver. No learner, policy sampling, or hard-coded integrator substitution is
used here.
"""

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from m1_unilab import assets, config
from m1_unilab.full_events import MaterialBuckets
from unilab.base.config_adapter import EnvConfigAdapter, create_env
from unilab.managers.event_manager import EventTermCfg
from unilab.managers.rng import ManagerRng
from unilab.managers.scene_entity_config import SceneEntityCfg

FIXTURE = Path(__file__).parent / "fixtures/high_contact_reset.npz"
MODEL_FIELDS = (
    "body_mass",
    "body_inertia",
    "body_ipos",
    "geom_friction",
    "geom_solref",
)
BAD_WARNINGS = (
    mujoco.mjtWarning.mjWARN_BADQACC,
    mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_BADQPOS,
)


def test_recorded_high_contact_reset_stays_finite_with_current_full_config(
    tmp_path, monkeypatch
):
    # Use the real asset builder/config, with generated XML confined to tmp_path.
    package_root = tmp_path / "m1_package"
    package_root.mkdir()
    monkeypatch.setattr(assets, "PACKAGE_ROOT", package_root)
    monkeypatch.setattr(config, "PACKAGE_ROOT", package_root)
    cfg = config.configuration("mixed", profile="b2w-full", num_envs=1, device="cpu")
    override = EnvConfigAdapter(
        cfg, root_dir=package_root, algo_name="ppo"
    ).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=1, env_cfg_override=override)
    native_path = tmp_path / "current_full_physics.mjb"
    try:
        model = env.backend.get_playback_model(env_index=0)
        assert model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD, (
            "Full terrain training must bypass the native CCD crash path via libccd"
        )
        layout = env.backend.get_physics_state_layout()
        assert (model.nq, model.nv) == (layout.nq, layout.nv)
        names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
            for index in range(model.ngeom)
        )
        assert names == tuple(env.backend.get_geom_names())
        mujoco.mj_saveModel(model, str(native_path))
    finally:
        env.close()

    with np.load(FIXTURE, allow_pickle=False) as recording:
        episode = {name: recording[name].copy() for name in recording.files}
    model = mujoco.MjModel.from_binary_path(str(native_path))
    assert model.opt.timestep == 0.001
    assert episode["controls"].shape == (97, model.nu)
    assert episode["forces"].shape == (97, model.nbody, 6)
    assert episode["initial_state"].shape == (1 + model.nq + model.nv,)
    assert all(
        getattr(model, name) == 0
        for name in ("na", "neq", "nmocap", "nplugin", "nuserdata")
    )

    # This is a full reset reconstruction, matching replay_utils.restore_episode:
    # restore actual DR -> mj_setConst -> reset -> initial qpos/qvel -> forward.
    for name in MODEL_FIELDS:
        target = getattr(model, name)
        assert target.shape == episode[name].shape
        assert np.isfinite(episode[name]).all()
        target[:] = episode[name]
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    mujoco.mj_resetData(model, data)
    initial = episode["initial_state"]
    data.time = initial[0]
    data.qpos[:] = initial[1 : 1 + model.nq]
    data.qvel[:] = initial[1 + model.nq :]
    data.ctrl[:] = 0
    data.qacc_warmstart[:] = 0
    data.qfrc_applied[:] = 0
    data.xfrc_applied[:] = 0
    mujoco.mj_forward(model, data)

    warnings_seen = np.array(
        [data.warning[int(kind)].number for kind in BAD_WARNINGS], dtype=np.int64
    )
    max_velocity = float(np.max(np.abs(data.qvel)))
    nonfinite_steps = []
    for step, (control, force) in enumerate(
        zip(episode["controls"], episode["forces"], strict=True)
    ):
        data.ctrl[:] = control
        data.xfrc_applied[:] = force
        mujoco.mj_step(model, data)  # No extra forward: preserve native warm-start.
        warnings_seen = np.maximum(
            warnings_seen, [data.warning[int(kind)].number for kind in BAD_WARNINGS]
        )
        if not (
            np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
            and np.isfinite(data.qacc).all()
        ):
            nonfinite_steps.append(step)
        max_velocity = max(max_velocity, float(np.max(np.abs(data.qvel))))

    evidence = {
        "integrator_from_configuration": int(model.opt.integrator),
        "bad_warning_counts": warnings_seen.tolist(),
        "nonfinite_steps": nonfinite_steps,
        "max_abs_qvel": max_velocity,
    }
    assert not warnings_seen.any(), evidence
    assert not nonfinite_steps, evidence
    assert max_velocity < 1e4, evidence


class MaterialWriteSink:
    """Public Entity surface receiving the real startup term's model writes."""

    def __init__(self):
        self.solref = None

    def bind_geom_friction_write(self, ids, *, term_name):
        return torch.tensor(ids), torch.tensor([[0.8, 0.005, 0.0001]] * len(ids))

    def bind_geom_solref_write(self, ids, *, term_name):
        return torch.tensor(ids), torch.tensor([[0.02, 1.0]] * len(ids))

    def write_geom_friction_to_sim(self, values, columns, env_ids, *, term_name):
        pass

    def write_geom_solref_to_sim(self, values, columns, env_ids, *, term_name):
        assert env_ids.tolist() == [0]
        assert torch.as_tensor(columns).tolist() == list(range(64))
        self.solref = values.clone()


class EveryBucketRng:
    """Real sampled material values, with one deterministic assignment per bucket."""

    def __init__(self):
        self.source = ManagerRng.seeded(17, device="cpu")
        self.device = self.source.device

    def uniform(self, low, high, size, *, dtype=torch.float32):
        return self.source.uniform(low, high, size, dtype=dtype)

    def integers(self, low, high, size):
        assert (low, high, size) == (0, 64, (1, 64))
        return torch.arange(64).reshape(1, 64)


def test_material_startup_preserves_contact_stiffness_across_all_64_buckets():
    entity = MaterialWriteSink()
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        rng=EveryBucketRng(),
        scene={"robot": entity},
    )
    asset_cfg = SceneEntityCfg("robot", geom_ids=list(range(64)))
    cfg = EventTermCfg(
        func=MaterialBuckets,
        mode="startup",
        params={"asset_cfg": asset_cfg, "num_buckets": 64, "time_constant": 0.02},
    )
    term = MaterialBuckets(cfg, env)
    term(env, torch.tensor([0]), **cfg.params)
    assert torch.unique(term.bucket_ids).numel() == 64
    assert entity.solref.shape == (1, 64, 2)
    tau, zeta = entity.solref[..., 0], entity.solref[..., 1]
    assert zeta.min() < 0.4  # Exercise genuinely underdamped material samples.
    # In positive-format MuJoCo solref, stiffness is proportional to
    # 1/(tau^2*zeta^2); changing restitution must not silently stiffen contact.
    torch.testing.assert_close(
        tau * zeta,
        torch.full((1, 64), 0.02, dtype=tau.dtype),
        rtol=1e-6,
        atol=1e-9,
    )
