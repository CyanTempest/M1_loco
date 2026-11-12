"""Compose the downstream task with UniLab's upstream PPO owner configuration."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from m1_locomotion.contract import DEFAULT_POSITION, JOINTS, LEG_JOINTS, POSITION_SCALE
from omegaconf import OmegaConf

from .assets import PACKAGE_ROOT, build_assets


def configuration(
    terrain="flat",
    *,
    num_envs=None,
    iterations=None,
    device="mps",
    run_name="",
    checkpoint=None,
    profile="basic",
):
    import unilab
    from unilab.base.registry import ensure_registries

    ensure_registries()
    upstream = Path(unilab.__file__).parent / "conf/ppo"
    with initialize_config_dir(config_dir=str(upstream), version_base="1.3"):
        cfg = compose(config_name="config")
    OmegaConf.set_struct(cfg, False)
    if profile not in ("basic", "b2w-full"):
        raise ValueError(profile)
    full = profile == "b2w-full"
    task = OmegaConf.load(
        Path(__file__).with_name("task_full.yaml" if full else "task.yaml")
    )
    # Replace Go2 task sections completely; retain only PPO/runtime defaults.
    for key in ("env", "reward"):
        OmegaConf.update(cfg, key, task[key], merge=False, force_add=True)
    for key in ("algo", "training"):
        OmegaConf.update(cfg, key, task[key], merge=True, force_add=True)
    cfg.play_profile.enabled = False
    paths = build_assets(profile)
    cfg.env.scene.model_file = str(paths[terrain])
    cfg.env.scene.entities = {
        "robot": {
            "root_body_name": "BASE_LINK",
            "body_names": ["BASE_LINK"],
            "joint_names": JOINTS,
            "actuator_names": [name + "_act" for name in JOINTS],
        }
    }
    if full:
        import xml.etree.ElementTree as ET

        from .full_drive import BODIES

        cfg.env.scene.fragment_files = [str(PACKAGE_ROOT / "assets/solver_full.xml")]
        cfg.env.scene.entities.robot.body_names = BODIES
        cfg.env.scene.entities.robot.geom_names = [
            geom.get("name")
            for geom in ET.parse(PACKAGE_ROOT / "assets/robot_full.xml").findall(
                ".//geom"
            )
        ]
    else:
        cfg.env.actions.legs.clip = {
            name: [
                DEFAULT_POSITION[name] - 3 * POSITION_SCALE[name],
                DEFAULT_POSITION[name] + 3 * POSITION_SCALE[name],
            ]
            for name in LEG_JOINTS
        }
    if num_envs is not None:
        cfg.algo.num_envs = num_envs
    if iterations is not None:
        cfg.algo.max_iterations = iterations
    if cfg.algo.num_envs < 1 or cfg.algo.max_iterations < 1:
        raise ValueError("Environment and iteration counts must be positive")
    cfg.algo.run_name = run_name or terrain
    cfg.training.device = device
    cfg.training.task_name = "M1Flat" if terrain == "flat" else "M1Mixed"
    if full:
        cfg.training.task_name += "Full"
    cfg.training.log_root = str(PACKAGE_ROOT / "runs")
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        # UniLab's PPO entry point consumes load_run (also accepts a file),
        # not the resume/resume_path fields used by some other runners.
        cfg.algo.load_run = str(checkpoint_path)
        cfg.algo.checkpoint = -1
    if terrain == "mixed":
        mixed = OmegaConf.load(Path(__file__).with_name("terrain.yaml"))
        cfg.env.scene.terrain = mixed.terrain
        cfg.env.curriculum = mixed.curriculum
        if full:
            cfg.env.scene.terrain.generator.border_width = 4.0
            cfg.env.curriculum.terrain.func = (
                "m1_unilab.full_mdp.TerrainLevelsDistanceFull"
            )
    return cfg
