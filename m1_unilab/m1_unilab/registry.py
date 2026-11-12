"""Register M1 with UniLab's existing Manager-Based runtime."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

for name in ("M1Flat", "M1Mixed", "M1FlatFull", "M1MixedFull"):
    registry.register_env_config(name, ManagerBasedRlEnvCfg)
    registry.register_env(name, make_manager_based_rl_env, sim_backend="mujoco")
