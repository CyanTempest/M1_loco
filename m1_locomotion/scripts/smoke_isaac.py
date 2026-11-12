#!/usr/bin/env python3
"""Run on the Linux GPU host before training. Checks real imported joint/observation order."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="M1-Velocity-Flat-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=128)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
import m1_locomotion.tasks
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from m1_locomotion.contract import ACTION_CLIP, ACTOR_DIM, JOINTS, LEG_JOINTS, WHEEL_JOINTS

try:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    raw = gym.make(args.task, cfg=cfg)
    env = RslRlVecEnvWrapper(raw, clip_actions=ACTION_CLIP)
    robot = env.unwrapped.scene["robot"]
    for term, expected in (("joint_pos", LEG_JOINTS), ("joint_vel", WHEEL_JOINTS)):
        actual = env.unwrapped.action_manager.get_term(term).IO_descriptor.joint_names
        assert actual == expected, (term, actual, expected)
    actor = env.unwrapped.observation_manager.compute_group("policy")
    critic = env.unwrapped.observation_manager.compute_group("critic")
    assert actor.shape == (args.num_envs, ACTOR_DIM), actor.shape
    assert critic.shape == (args.num_envs, 243), critic.shape
    assert set(robot.joint_names) == set(JOINTS), robot.joint_names
    with torch.inference_mode():
        for _ in range(args.steps):
            obs, rewards, dones, extras = env.step(torch.zeros((args.num_envs, 16), device=env.device))
            assert torch.isfinite(rewards).all()
            assert torch.isfinite(robot.data.root_state_w).all()
            assert torch.isfinite(env.unwrapped.observation_manager.compute_group("policy")).all()
    print(f"PASS: {args.task}, actor=53 critic=243 actions=16; {args.steps} steps. This is not a trained policy test.")
    env.close()
finally:
    app.close()
