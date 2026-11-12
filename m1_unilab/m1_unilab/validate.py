"""Check native actuator semantics, batch dimensions and mixed terrain resets."""

import argparse
import json
import time

import mujoco
import numpy as np
import torch
from m1_locomotion.contract import JOINTS, LEG_JOINTS, WHEEL_JOINTS
from unilab.base.config_adapter import EnvConfigAdapter, create_env

from .assets import PACKAGE_ROOT
from .config import configuration


def validate(terrain="flat", num_envs=8, steps=250):
    cfg = configuration(terrain, num_envs=num_envs, device="cpu")
    model = mujoco.MjModel.from_xml_path(cfg.env.scene.model_file)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    wheel_act = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_act")
        for n in WHEEL_JOINTS
    ]
    # Native wheel servo must be velocity feedback, not a position target.
    assert np.allclose(model.actuator_gainprm[wheel_act, 0], 1)
    assert np.allclose(model.actuator_biasprm[wheel_act, 1], 0)
    assert np.allclose(model.actuator_biasprm[wheel_act, 2], -1)
    assert np.allclose(model.actuator_forcerange[wheel_act], [-28.68, 28.68])
    override = EnvConfigAdapter(
        cfg, root_dir=PACKAGE_ROOT, algo_name="ppo"
    ).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=num_envs, env_cfg_override=override)
    start = time.perf_counter()
    try:
        obs, _ = env.reset()
        playback_model = env.backend.get_playback_model()
        assert playback_model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        assert obs["obs"].shape == (num_envs, 53), {k: v.shape for k, v in obs.items()}
        assert obs["critic"].shape == (num_envs, 243)
        assert env.action_manager.total_action_dim == 16
        assert env.scene["robot"].joint_names == tuple(JOINTS)
        assert env.action_manager.get_term("legs").target_names == list(LEG_JOINTS)
        assert env.action_manager.get_term("wheels").target_names == list(WHEEL_JOINTS)
        actions = torch.zeros((num_envs, 16), device=env.device)
        resets = 0
        rewards = []
        for _ in range(steps):
            state = env.step(actions)
            assert torch.isfinite(state.reward).all()
            assert all(torch.isfinite(x).all() for x in state.obs.values())
            rewards.append(float(state.reward.mean()))
            resets += int(state.terminated.sum())
        result = {
            "terrain": terrain,
            "num_envs": num_envs,
            "steps": steps,
            "policy_dim": 53,
            "critic_dim": 243,
            "action_dim": 16,
            "wheel_kp": 0,
            "wheel_kd": 1,
            "mass_kg": float(model.body_mass.sum()),
            "integrator": "implicitfast",
            "sim_dt": float(cfg.env.sim_dt),
            "ctrl_dt": float(cfg.env.ctrl_dt),
            "zero_action_terminations": resets,
            "reward_mean": float(np.mean(rewards)),
            "elapsed_seconds": time.perf_counter() - start,
        }
        if terrain == "mixed":
            result["terrain_types"] = list(cfg.env.scene.terrain.generator.sub_terrains)
            result["terrain_rows"] = int(env.scene.terrain.terrain_origins.shape[0])
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain", choices=("flat", "mixed"), default="flat")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=250)
    args = parser.parse_args()
    result = validate(args.terrain, args.num_envs, args.steps)
    output = PACKAGE_ROOT / f"validation_{args.terrain}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
