"""Record genuine checkpoint rollouts through the public MuJoCo playback API."""

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from rsl_rl.runners import OnPolicyRunner
from unilab.base.config_adapter import EnvConfigAdapter, create_env
from unilab.rl import RslRlVecEnvAdapter
from unilab.training import algo_config_dict

from .assets import PACKAGE_ROOT
from .evaluate import evaluation_configuration
from .rendering import original_visual_model

CASES = {
    "flat": ("flat", 0, "平地"),
    "uphill": ("uphill", 2, "上坡 · 约 9.9°"),
    "downhill": ("downhill", 2, "下坡 · 约 9.9°"),
    "rough": ("rough", 0, "粗糙地面 · 高度噪声 ±4 cm"),
    "stairs_up_7cm": ("stairs_up", 1, "上楼梯 · 台阶 7 cm"),
    "stairs_down_7cm": ("stairs_down", 1, "下楼梯 · 台阶 7 cm"),
    "stairs_up_9_5cm": ("stairs_up", 2, "上楼梯 · 台阶 9.5 cm"),
}


def record(
    checkpoint,
    case,
    output,
    *,
    seconds=20.0,
    width=960,
    height=540,
    appearance="visual",
    profile="basic",
):
    terrain, level, title = CASES[case]
    steps = round(seconds / 0.02)
    cfg = evaluation_configuration(
        terrain, steps=steps, count=8, level=level, profile=profile
    )
    override = EnvConfigAdapter(
        cfg, root_dir=PACKAGE_ROOT, algo_name="ppo"
    ).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=8, env_cfg_override=override)
    renderer = None
    encoder = None
    output.mkdir(parents=True, exist_ok=True)
    video = output / f"{case}.mp4"
    font_file = "/System/Library/Fonts/STHeiti Medium.ttc"
    heading = ImageFont.truetype(font_file, 29)
    small = ImageFont.truetype(font_file, 21)
    try:
        wrapped = RslRlVecEnvAdapter(env, device="cpu")
        runner = OnPolicyRunner(
            wrapped, algo_config_dict(cfg), log_dir=None, device="cpu"
        )
        runner.load(str(checkpoint), map_location="cpu")
        policy = runner.get_inference_policy(device="cpu")
        obs = wrapped.get_observations()
        # A detached render model; only appearance is changed, never physics.
        model = (
            original_visual_model(env.backend)
            if appearance == "visual"
            else copy.copy(env.backend.get_playback_model(env_index=0))
        )
        model.opt.disableflags = env.backend.get_playback_model(
            env_index=0
        ).opt.disableflags
        scene_option = mujoco.MjvOption()
        if appearance == "visual":
            scene_option.geomgroup[3] = 0
        model.vis.global_.offwidth = width
        model.vis.global_.offheight = height
        model.vis.headlight.ambient[:] = [0.25, 0.25, 0.25]
        model.vis.headlight.diffuse[:] = [0.45, 0.45, 0.45]
        model.vis.headlight.specular[:] = [0.05, 0.05, 0.05]
        model.vis.quality.shadowsize = 2048
        wheel_geoms = []
        for gid in range(model.ngeom):
            if appearance == "visual":
                continue
            body = (
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]
                )
                or ""
            )
            if "FOOT_LINK" in body:
                model.geom_rgba[gid] = [0.10, 0.12, 0.14, 1]
                wheel_geoms.append(gid)
            elif "BASE_LINK" in body:
                model.geom_rgba[gid] = [0.12, 0.42, 0.66, 1]
            elif model.geom_bodyid[gid] != 0:
                model.geom_rgba[gid] = [0.65, 0.71, 0.75, 1]
            else:
                model.geom_rgba[gid] = [0.61, 0.64, 0.62, 1]
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 2.6
        camera.azimuth = 110
        camera.elevation = -22
        root = env.backend.get_root_state_layout("BASE_LINK")
        qids = list(root.qpos_indices)
        initial = env.backend.get_state_views(("qpos",), device="cpu")["qpos"][0]
        start_x = float(initial[qids[0]])
        progress = 0.0
        sim_time = 0.0
        speed = 0.0
        stopped = False
        outcome = "行进中"
        final_image = None
        encoder = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                "25",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video),
            ],
            stdin=subprocess.PIPE,
        )
        for frame in range(round(seconds * 25)):
            if not stopped:
                for _ in range(2):
                    with torch.inference_mode():
                        action = policy(obs)
                        obs, _, done, _ = wrapped.step(action)
                    sim_time += 0.02
                    if bool(done[0]):
                        stopped, outcome = True, "触发终止 · 已暂停"
                        break
                    state = env.backend.get_state_views(("qpos", "qvel"), device="cpu")
                    assert model.nq == state["qpos"].shape[1]
                    data.qpos[:] = state["qpos"][0].numpy()
                    data.qvel[:] = state["qvel"][0].numpy()
                    progress = float(data.qpos[qids[0]]) - start_x
                    speed = (
                        float(obs["critic"][0, 0])
                        if profile == "b2w-full"
                        else float(
                            env.backend.get_sensor_view("local_linvel", device="cpu")[
                                0, 0
                            ]
                        )
                    )
                    if progress >= 4.0:
                        stopped, outcome = True, "到达 4 m · 已暂停"
                        break
                mujoco.mj_forward(model, data)
                camera.lookat[:] = data.qpos[qids[:3]] + np.array([0.22, 0, -0.16])
                renderer.update_scene(data, camera=camera, scene_option=scene_option)
                # Nonphysical spoke markers show actual wheel rotation.
                for gid in wheel_geoms:
                    radius, halfwidth = model.geom_size[gid, :2]
                    rotation = data.geom_xmat[gid].reshape(3, 3)
                    for side in (-1, 1):
                        a = data.geom_xpos[gid] + rotation @ np.array(
                            [-radius * 0.8, 0, side * (halfwidth + 0.001)]
                        )
                        b = data.geom_xpos[gid] + rotation @ np.array(
                            [radius * 0.8, 0, side * (halfwidth + 0.001)]
                        )
                        scene = renderer.scene
                        geom = scene.geoms[scene.ngeom]
                        mujoco.mjv_initGeom(
                            geom,
                            mujoco.mjtGeom.mjGEOM_LINE,
                            np.zeros(3),
                            np.zeros(3),
                            np.eye(3).ravel(),
                            np.array([0.85, 0.88, 0.9, 1.0], dtype=np.float32),
                        )
                        mujoco.mjv_connector(
                            geom, mujoco.mjtGeom.mjGEOM_LINE, 2.0, a, b
                        )
                        scene.ngeom += 1
                if terrain == "flat":
                    # Render-only floor grid makes forward motion visible.
                    for axis in (0, 1):
                        for coordinate in np.arange(-7, 7.1, 0.5):
                            scene = renderer.scene
                            geom = scene.geoms[scene.ngeom]
                            a, b = (
                                np.array([-7.0, coordinate, 0.002]),
                                np.array([7.0, coordinate, 0.002]),
                            )
                            if axis:
                                a, b = a[[1, 0, 2]], b[[1, 0, 2]]
                            mujoco.mjv_initGeom(
                                geom,
                                mujoco.mjtGeom.mjGEOM_LINE,
                                np.zeros(3),
                                np.zeros(3),
                                np.eye(3).ravel(),
                                np.array([0.3, 0.35, 0.37, 1.0], dtype=np.float32),
                            )
                            mujoco.mjv_connector(
                                geom, mujoco.mjtGeom.mjGEOM_LINE, 1.0, a, b
                            )
                            scene.ngeom += 1
                final_image = renderer.render().copy()
            if frame == round(seconds * 25) - 1 and not stopped:
                outcome = "20 秒内未到达 4 m" if seconds == 20 else "片段结束"
            image = Image.fromarray(final_image)
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, width, 91), fill=(18, 25, 33))
            draw.text((22, 12), title, font=heading, fill=(240, 246, 250))
            draw.text(
                (22, 55),
                f"M1 {'原始外观' if appearance == 'visual' else '碰撞体'}仿真  |  指令 0.5 m/s  |  原速播放",
                font=small,
                fill=(177, 194, 206),
            )
            draw.rectangle((0, height - 64, width, height), fill=(18, 25, 33))
            draw.text(
                (22, height - 54),
                f"t={sim_time:4.1f}s   前进 {progress:.2f}m   速度 {speed:.2f}m/s",
                font=small,
                fill=(224, 235, 243),
            )
            draw.text(
                (22, height - 28),
                outcome,
                font=small,
                fill=(117, 219, 170) if progress >= 4 else (244, 196, 109),
            )
            encoder.stdin.write(np.asarray(image).tobytes())
            if frame in (0, 125, round(seconds * 25) - 1):
                image.save(output / f"{case}_frame{frame:04d}.jpg")
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("ffmpeg encoding failed")
        encoder = None
        report = {
            "profile": profile,
            "video": video.name,
            "case": case,
            "terrain": terrain,
            "level": level,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "environment_index": 0,
            "evaluation_num_envs": 8,
            "seed": 123,
            "command_vx": 0.5,
            "simulated_seconds": sim_time,
            "video_seconds": seconds,
            "fps": 25,
            "playback_speed": 1,
            "progress_m": progress,
            "goal_reached": progress >= 4,
            "status": outcome,
            "appearance": (
                "17 original URDF visual meshes and material colors on a detached render model; training physics unchanged"
                if appearance == "visual"
                else "training collision geometry; colors, wheel rotation markers and floor grid adjusted only in detached playback model"
            ),
        }
        (output / f"{case}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        if encoder is not None:
            encoder.stdin.close()
            encoder.wait()
        if renderer is not None:
            renderer.close()
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=PACKAGE_ROOT / "checkpoints/mixed.pt"
    )
    parser.add_argument("--case", choices=("all", *CASES), default="all")
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT / "videos")
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument(
        "--appearance", choices=("visual", "collision"), default="visual"
    )
    parser.add_argument("--profile", choices=("basic", "b2w-full"), default="basic")
    args = parser.parse_args()
    torch.set_num_threads(2)
    for case in CASES if args.case == "all" else (args.case,):
        record(
            args.checkpoint,
            case,
            args.output,
            seconds=args.seconds,
            appearance=args.appearance,
            profile=args.profile,
        )


if __name__ == "__main__":
    main()
