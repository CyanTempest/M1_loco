"""Local browser controls for one M1 policy rollout."""

import argparse
import io
import json
import math
import threading
import time
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image
from rsl_rl.runners import OnPolicyRunner
from unilab.base.config_adapter import EnvConfigAdapter, create_env
from unilab.managers.command_manager import CommandTerm, CommandTermCfg
from unilab.rl import RslRlVecEnvAdapter
from unilab.training import algo_config_dict

from .assets import PACKAGE_ROOT
from .evaluate import evaluation_configuration
from .rendering import original_visual_model

TERRAINS = ("flat", "uphill", "downhill", "rough", "stairs_up", "stairs_down")


class ManualVelocity(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.values = torch.zeros((env.num_envs, 3), device=env.device)

    @property
    def command(self):
        return self.values

    def set_velocity(self, values):
        self.values[:] = torch.tensor(values, device=self.values.device)

    def _update_metrics(self):
        pass

    def _resample_command(self, reset_mask):
        pass

    def _update_command(self):
        pass

    def refresh_reset(self, reset_mask):
        # Commands are set by the operator and persist through environment resets.
        pass

    def post_compute_reset(self):
        pass


@dataclass(kw_only=True)
class ManualVelocityCfg(CommandTermCfg):
    def build(self, env):
        return ManualVelocity(self, env)


class BrowserState:
    def __init__(self, checkpoint):
        self.lock = threading.Lock()
        self.frame = b""
        self.controls = {
            "vx": 0.0,
            "vy": 0.0,
            "yaw": 0.0,
            "paused": True,
            "terrain": "flat",
            "level": 2,
            "reset": 0,
        }
        self.telemetry = {
            "status": "正在加载 M1 和策略…",
            "checkpoint": str(
                torch.load(checkpoint, map_location="cpu", weights_only=False).get(
                    "iter", checkpoint.stem
                )
            ),
            "measured": [0, 0, 0],
            "applied": [0, 0, 0],
            "simulation_seconds": 0,
        }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, data, content_type, code=200):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self.reply(
                Path(__file__).with_name("interactive.html").read_bytes(),
                "text/html; charset=utf-8",
            )
        elif path == "/state":
            with self.server.state.lock:
                data = {**self.server.state.telemetry, **self.server.state.controls}
            self.reply(json.dumps(data).encode(), "application/json")
        elif path == "/frame":
            with self.server.state.lock:
                frame = self.server.state.frame
            self.reply(frame, "image/jpeg", 200 if frame else 503)
        else:
            self.reply(b"Not found", "text/plain", 404)

    def do_POST(self):
        try:
            if self.path != "/control":
                raise ValueError("Unknown endpoint")
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 2048:
                raise ValueError("Invalid request length")
            update = json.loads(self.rfile.read(length))
            if not isinstance(update, dict) or set(update) - set(
                self.server.state.controls
            ):
                raise ValueError("Invalid controls")
            for key, value in update.items():
                if key in ("vx", "vy", "yaw"):
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or abs(value) > 1
                    ):
                        raise ValueError("Velocity must be finite and within [-1,1]")
                elif key == "terrain" and value not in TERRAINS:
                    raise ValueError("Invalid terrain")
                elif key == "level" and (
                    type(value) is not int or value not in range(5)
                ):
                    raise ValueError("Invalid difficulty")
                elif key == "paused" and type(value) is not bool:
                    raise ValueError("Invalid pause state")
                elif key == "reset" and (type(value) is not int or value < 0):
                    raise ValueError("Invalid reset counter")
            with self.server.state.lock:
                self.server.state.controls.update(update)
            self.reply(b"{}", "application/json")
        except (ValueError, TypeError) as error:
            self.reply(
                json.dumps({"error": str(error)}).encode(), "application/json", 400
            )


def rollout(checkpoint, state):
    env = renderer = None
    selection = None
    last_reset = -1
    last_render = 0.0
    sim_time = 0.0
    redraw = True
    try:
        while True:
            started = time.monotonic()
            with state.lock:
                controls = state.controls.copy()
            current = (controls["terrain"], controls["level"])
            if current != selection:
                with state.lock:
                    state.telemetry["status"] = "正在加载地形…"
                if renderer is not None:
                    renderer.close()
                if env is not None:
                    env.close()
                cfg = evaluation_configuration(
                    current[0],
                    count=1,
                    steps=180000,
                    level=current[1],
                    speed=0,
                    profile="b2w-full",
                )
                cfg.env.commands.twist = {
                    "_target_": "m1_unilab.interactive.ManualVelocityCfg",
                    "resampling_time_range": [3600, 3600],
                }
                override = EnvConfigAdapter(
                    cfg, root_dir=PACKAGE_ROOT, algo_name="ppo"
                ).build_task_env_cfg_override()
                env = create_env(cfg, num_envs=1, env_cfg_override=override)
                wrapped = RslRlVecEnvAdapter(env, device="cpu")
                runner = OnPolicyRunner(
                    wrapped, algo_config_dict(cfg), log_dir=None, device="cpu"
                )
                runner.load(str(checkpoint), map_location="cpu")
                policy = runner.get_inference_policy(device="cpu")
                obs = wrapped.get_observations()
                command = env.command_manager.get_term("twist")
                model = original_visual_model(env.backend)
                physical = env.backend.get_playback_model(env_index=0)
                model.opt.disableflags = physical.opt.disableflags
                model.vis.global_.offwidth = 800
                model.vis.global_.offheight = 500
                data = mujoco.MjData(model)
                renderer = mujoco.Renderer(model, height=500, width=800)
                option = mujoco.MjvOption()
                option.geomgroup[3] = 0
                camera = mujoco.MjvCamera()
                camera.distance, camera.azimuth, camera.elevation = 2.8, 110, -24
                selection = current
                last_reset = -1
            if controls["reset"] != last_reset:
                obs, _ = wrapped.reset()
                sim_time = 0
                last_reset = controls["reset"]
                redraw = True
            target = [controls[k] for k in ("vx", "vy", "yaw")]
            command.set_velocity(target)
            if not controls["paused"]:
                with torch.no_grad():
                    obs, _, done, _ = wrapped.step(policy(obs))
                sim_time += 0.02
                if bool(done[0]):
                    with state.lock:
                        state.telemetry["status"] = "已摔倒或到达边界，环境已自动复位"
            if (redraw or not controls["paused"]) and started - last_render >= 0.1:
                views = env.backend.get_state_views(("qpos", "qvel"), device="cpu")
                data.qpos[:] = views["qpos"][0].numpy()
                data.qvel[:] = views["qvel"][0].numpy()
                mujoco.mj_forward(model, data)
                camera.lookat[:] = data.qpos[:3] + np.array([0.2, 0, -0.15])
                renderer.update_scene(data, camera=camera, scene_option=option)
                buffer = io.BytesIO()
                Image.fromarray(renderer.render()).save(
                    buffer, format="JPEG", quality=85
                )
                with state.lock:
                    state.frame = buffer.getvalue()
                    state.telemetry.update(
                        status="预览已暂停" if controls["paused"] else "运行中",
                        measured=[float(obs["critic"][0, i]) for i in (0, 1, 5)],
                        applied=command.command[0].tolist(),
                        simulation_seconds=round(sim_time, 2),
                    )
                last_render = started
                redraw = False
            time.sleep(max(0, 0.02 - (time.monotonic() - started)))
    finally:
        if renderer is not None:
            renderer.close()
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PACKAGE_ROOT / "evaluations/iteration_10000/checkpoint.pt",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    torch.set_num_threads(1)
    state = BrowserState(args.checkpoint.resolve())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.state = state
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{args.port}"
    print(f"M1 interactive preview: {url}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        rollout(args.checkpoint, state)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
