"""Evaluate saved PPO policies at fixed commands, separately for every terrain."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from unilab.base.config_adapter import EnvConfigAdapter, create_env
from unilab.rl import RslRlVecEnvAdapter
from unilab.training import algo_config_dict

from .assets import PACKAGE_ROOT
from .config import configuration


def evaluation_configuration(
    terrain, *, steps=1000, count=8, speed=0.5, level=0, profile="basic"
):
    """Shared test conditions for numeric evaluation and recorded rollouts."""
    cfg = configuration(
        "flat" if terrain == "flat" else "mixed",
        num_envs=count,
        device="cpu",
        profile=profile,
    )
    cfg.env.seed = 123
    if profile == "b2w-full":
        for name in ("materials", "mass_inertia", "com", "push"):
            del cfg.env.events[name]
        for axis in ("roll", "pitch", "yaw", "z"):
            cfg.env.events.root.params.pose_range[axis] = [0, 0]
        for axis in ("x", "y"):
            cfg.env.events.root.params.pose_range[axis] = [-0.1, 0.1]
        for axis in cfg.env.events.root.params.velocity_range:
            cfg.env.events.root.params.velocity_range[axis] = [0, 0]
        drive = cfg.env.actions.drive
        drive.gain_scale_range = [1, 1]
        drive.delay_min_steps = drive.delay_max_steps = 0
        drive.force_range = drive.torque_range = [0, 0]
        cfg.env.observations.policy.enable_corruption = False
        cfg.env.commands.twist.heading_command = False
        cfg.env.commands.twist.rel_heading_envs = 0.0
        cfg.env.terminations.fallen = {
            "func": "unilab.envs.mdp.bad_orientation",
            "params": {"limit_angle": 0.8},
        }
    cfg.env.max_episode_seconds = (steps + 1) * 0.02
    cfg.env.events.root.params.pose_range.yaw = [0, 0]
    cfg.env.commands.twist.rel_standing_envs = 0.0
    cfg.env.commands.twist.ranges = {
        "lin_vel_x": [speed, speed],
        "lin_vel_y": [0, 0],
        "ang_vel_z": [0, 0],
    }
    cfg.env.commands.twist.resampling_time_range = [100.0, 100.0]
    if terrain != "flat":
        # Keep the same terrain definitions, evaluating exactly one named type.
        generator = cfg.env.scene.terrain.generator
        selected = generator.sub_terrains[terrain]
        selected.proportion = 1.0
        generator.sub_terrains = {terrain: selected}
        generator.curriculum = False
        generator.num_rows = 1
        generator.num_cols = 1
        difficulty = (level + 0.5) / 5  # midpoint of the training row's interval
        generator.difficulty_range = [difficulty, difficulty]
        cfg.env.curriculum = {}
    return cfg


def evaluate(
    checkpoint, terrain, *, steps=1000, count=8, speed=0.5, level=0, profile="basic"
):
    cfg = evaluation_configuration(
        terrain, steps=steps, count=count, speed=speed, level=level, profile=profile
    )
    override = EnvConfigAdapter(
        cfg, root_dir=PACKAGE_ROOT, algo_name="ppo"
    ).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=count, env_cfg_override=override)
    try:
        wrapped = RslRlVecEnvAdapter(env, device="cpu")
        runner = OnPolicyRunner(
            wrapped, algo_config_dict(cfg), log_dir=None, device="cpu"
        )
        runner.load(str(checkpoint), map_location="cpu")
        policy = runner.get_inference_policy(device="cpu")
        obs = wrapped.get_observations()
        layout = env.backend.get_root_state_layout("BASE_LINK")
        qids = list(layout.qpos_indices)
        qpos = env.backend.get_state_views(("qpos",), device="cpu")["qpos"]
        start_x = qpos[:, qids[0]].clone()
        failures = torch.zeros(count, dtype=torch.bool)
        reached = torch.zeros(count, dtype=torch.bool)
        progress = torch.zeros(count)
        goal = min(4.0, speed * steps * 0.02 * 0.7)
        error = []
        measured = []
        clearances = []
        for _ in range(steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, done, _ = wrapped.step(actions)
                failures |= done.cpu() & ~reached
                # Survivor position is measured before any future reset.
                xpos = env.backend.get_state_views(("qpos",), device="cpu")["qpos"][
                    :, qids[0]
                ]
                progress = torch.where(failures | reached, progress, xpos - start_x)
                velocity = (
                    obs["critic"][:, 0]
                    if profile == "b2w-full"
                    else torch.as_tensor(
                        env.backend.get_sensor_view("local_linvel", device="cpu")
                    )[:, 0]
                )
                active = ~(failures | reached)
                if active.any():
                    error.append(float((velocity[active] - speed).abs().mean()))
                    measured.append(float(velocity[active].mean()))
                    # 53 proprioception + 3 velocity + 187 scan; central scan index=93.
                    clearances.append(
                        float((obs["critic"][active, 60 + 93] + 0.5).min())
                        if profile == "b2w-full"
                        else float((obs["critic"][active, 56 + 93] / 5 + 0.5).min())
                    )
                reached |= (progress >= goal) & ~failures
                if (reached | failures).all():
                    break
        return {
            "profile": profile,
            "terrain": terrain,
            "level": level,
            "difficulty": (level + 0.5) / 5 if terrain != "flat" else None,
            "episodes": count,
            "duration_s": steps * 0.02,
            "command_vx": speed,
            "survival_until_goal_or_horizon_rate": float((~failures).float().mean()),
            "forward_progress_mean_m": float(progress.mean()),
            "vx_mean_m_s": float(np.mean(measured)) if measured else None,
            "vx_mae_m_s": float(np.mean(error)) if error else None,
            "minimum_center_clearance_m": min(clearances) if clearances else None,
            "progress_success_rate": float(reached.float().mean()),
            "success_definition": "reach min(4m,70% commanded distance) without a fall or other termination; stop scoring each row at the goal",
            "scope": "nominal simulation, one fixed command/seed; not hardware validation",
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--terrain",
        choices=(
            "all",
            "flat",
            "uphill",
            "downhill",
            "rough",
            "stairs_up",
            "stairs_down",
        ),
        default="all",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--level", type=int, choices=range(5), default=0)
    parser.add_argument("--profile", choices=("basic", "b2w-full"), default="basic")
    args = parser.parse_args()
    torch.set_num_threads(2)
    names = (
        ("flat", "uphill", "downhill", "rough", "stairs_up", "stairs_down")
        if args.terrain == "all"
        else (args.terrain,)
    )
    result = {"checkpoint": str(args.checkpoint.resolve()), "results": []}
    for name in names:
        row = evaluate(
            args.checkpoint,
            name,
            steps=args.steps,
            count=args.num_envs,
            level=args.level,
            profile=args.profile,
        )
        result["results"].append(row)
        print(json.dumps(row), flush=True)
    path = (
        args.checkpoint.parent
        / f"evaluation_{args.checkpoint.stem}_{args.terrain}_level{args.level}.json"
    )
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
