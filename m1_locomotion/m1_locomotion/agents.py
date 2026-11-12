from isaaclab.utils import configclass
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_b2w.agents.rsl_rl_ppo_cfg import (
    UnitreeB2WRoughPPORunnerCfg,
)
from .contract import ACTION_CLIP


@configclass
class M1PPORunnerCfg(UnitreeB2WRoughPPORunnerCfg):
    # Shared log root allows flat -> rough --resume with identical network sizes.
    experiment_name = "m1_locomotion"
    max_iterations = 10000
    save_interval = 100
    clip_actions = ACTION_CLIP
    seed = 42
