"""String-only Gym registration; safe before AppLauncher (no simulator imports)."""
import gymnasium as gym

for suffix, config in {
    "Flat": "M1FlatEnvCfg", "Rough": "M1RoughEnvCfg",
    "Flat-Preview": "M1PreviewFlatEnvCfg", "Rough-Preview": "M1PreviewRoughEnvCfg",
}.items():
    gym.register(
        id=f"M1-Velocity-{suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"m1_locomotion.env_cfg:{config}",
                "rsl_rl_cfg_entry_point": "m1_locomotion.agents:M1PPORunnerCfg"},
    )
