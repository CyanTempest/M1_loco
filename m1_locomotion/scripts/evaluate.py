#!/usr/bin/env python3
"""Evaluate one terrain family and seed; writes actual measured metrics, never assumed success."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="M1-Velocity-Rough-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--terrain", choices=("flat", "rough", "slope_up", "slope_down", "stairs_up", "stairs_down"), required=True)
parser.add_argument("--difficulty", type=float, default=0.5)
parser.add_argument("--num_envs", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--duration", type=float, default=8.0)
parser.add_argument("--velocity", type=float, default=0.5)
parser.add_argument("--distance", type=float, default=3.0)
parser.add_argument("--randomize", action="store_true")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 0 <= args.difficulty <= 1:
    parser.error("difficulty must be between 0 and 1")
if args.duration <= 0 or args.num_envs < 1 or args.velocity <= 0 or args.distance <= 0:
    parser.error("duration, num_envs, velocity and distance must be positive")
app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import m1_locomotion.tasks
from m1_locomotion.terrains import terrain_cfg
from m1_locomotion.contract import JOINTS

try:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    # Horizon beyond rollout: automatic timeouts cannot masquerade as success.
    cfg.episode_length_s = args.duration + 1.0
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = terrain_cfg(args.terrain)
    generator = cfg.scene.terrain.terrain_generator
    generator.seed = args.seed
    generator.curriculum = False
    generator.difficulty_range = (args.difficulty, args.difficulty)
    generator.num_rows = 1
    generator.num_cols = 1
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.curriculum.terrain_levels = None
    cfg.terminations.terrain_out_of_bounds = None
    command = cfg.commands.base_velocity
    command.ranges.lin_vel_x = (args.velocity, args.velocity)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.rel_standing_envs = 0.0
    command.resampling_time_range = (1000.0, 1000.0)
    cfg.events.randomize_reset_base.params["pose_range"] = {
        "x": (-.1, .1), "y": (-.1, .1), "yaw": (0.0, 0.0)}
    cfg.events.randomize_reset_base.params["velocity_range"] = {}
    if not args.randomize:
        for name in ("randomize_rigid_body_mass_base", "randomize_rigid_body_mass_others",
                     "randomize_com_positions", "randomize_actuator_gains",
                     "randomize_apply_external_force_torque", "randomize_push_robot"):
            setattr(cfg.events, name, None)
        cfg.events.randomize_rigid_body_material.params.update(
            static_friction_range=(.8, .8), dynamic_friction_range=(.8, .8), restitution_range=(0., 0.))
        for actuator in cfg.scene.robot.actuators.values():
            actuator.min_delay = actuator.max_delay = 0
        cfg.observations.policy.enable_corruption = False
    agent = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent.device = args.device
    raw = gym.make(args.task, cfg=cfg)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=args.device)
    runner.load(str(args.checkpoint.resolve()), load_optimizer=False)
    policy = runner.get_inference_policy(device=args.device)
    obs = env.get_observations()
    robot = env.unwrapped.scene["robot"]
    joint_ids, _ = robot.find_joints(JOINTS, preserve_order=True)
    start_x = robot.data.root_pos_w[:, 0].clone()
    alive = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
    progress = torch.zeros(args.num_envs, device=env.device)
    sq_error = torch.zeros(args.num_envs, device=env.device)
    samples = torch.zeros(args.num_envs, device=env.device)
    peak_torque = torch.zeros((args.num_envs, 16), device=env.device)
    with torch.inference_mode():
        for _ in range(round(args.duration / env.unwrapped.step_dt)):
            # Sample before step: done environments are reset by Isaac during step.
            measured_progress = robot.data.root_pos_w[:, 0] - start_x
            progress = torch.where(alive, measured_progress, progress)
            sq_error += alive * (robot.data.root_lin_vel_b[:, 0] - args.velocity).square()
            samples += alive
            torque = robot.data.applied_torque[:, joint_ids].abs()
            peak_torque = torch.where(alive[:, None], torch.maximum(peak_torque, torque), peak_torque)
            obs, _, dones, _ = env.step(policy(obs))
            alive &= ~dones.bool()
    success = alive & (progress >= args.distance)
    results = {
        "checkpoint": str(args.checkpoint.resolve()), "task": args.task, "terrain": args.terrain,
        "difficulty": args.difficulty, "seed": args.seed, "trials": args.num_envs,
        "duration_s": args.duration, "command_vx_m_s": args.velocity, "required_progress_m": args.distance,
        "randomized": args.randomize, "success_definition": "No termination during rollout AND net +X progress >= required distance",
        "success_rate": success.float().mean().item(), "survival_rate": alive.float().mean().item(),
        "velocity_rmse_until_first_termination_m_s": (sq_error.sum()/samples.sum().clamp(min=1)).sqrt().item(),
        "mean_progress_until_first_termination_m": progress.mean().item(),
        "peak_torques_Nm": dict(zip(JOINTS, peak_torque.max(dim=0).values.cpu().tolist())),
        "note": "Samples stop before auto-reset; final pose is measured one policy step before horizon. Pyramid terrain traversal is a proxy, not a real staircase certification.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    env.close()
finally:
    app.close()
