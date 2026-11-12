"""B2W rough reward equations on the M1 drive's public physics snapshot.

Reference: robot_lab 09f6a9dfdf48f32f38bb851dfa3a7d44db32b270,
``velocity/config/wheeled/unitree_b2w/rough_env_cfg.py`` and ``mdp/rewards.py``.
These are unweighted, unintegrated values: RewardManager applies weight and dt.
The caller owns physical-step acceleration, contact history, and contact events.
"""

from collections.abc import Mapping
from typing import Any

import torch
from unilab.managers.manager_base import ManagerTermBase

REWARD_NAMES = (
    "lin_vel_z_l2",
    "ang_vel_xy_l2",
    "joint_torques_l2",
    "joint_acc_l2",
    "joint_acc_wheel_l2",
    "joint_pos_limits",
    "joint_power",
    "stand_still",
    "joint_pos_penalty",
    "joint_mirror",
    "action_rate_l2",
    "undesired_contacts",
    "contact_forces",
    "track_lin_vel_xy_exp",
    "track_ang_vel_z_exp",
    "feet_contact_without_cmd",
    "upward",
)
WHEEL_BODY_IDS = (4, 8, 12, 16)
NONWHEEL_BODY_IDS = (0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15)


def _field(snapshot: Any, name: str) -> torch.Tensor:
    return snapshot[name] if isinstance(snapshot, Mapping) else getattr(snapshot, name)


def reward_values(
    s: Any,
    default_q: torch.Tensor,
    soft_limits: torch.Tensor,
    symmetry_signs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Evaluate all 17 nonzero B2W rough terms without side effects.

    ``s`` is a mapping or object with tensor fields ``q, qd, qdd, tau`` (N,16),
    ``gravity, lin_vel, ang_vel, command`` (N,3), ``action, last_action``
    (N,16), ``contact_forces`` (N,3,17,3), and ``first_contact`` (N,17).
    Velocities and unit gravity are in the body frame (upright gravity_z=-1).
    Torque is actual applied joint torque, not the policy action or PD target.
    Force history contains the last three *physics* samples; first_contact is
    the caller's Isaac-compatible first-contact predicate over policy dt.

    Joint order is FR/FL/RR/RL legs (ABAD,HIP,KNEE) followed by four wheels.
    Body order is BASE_LINK then ABAD/HIP/KNEE/FOOT for those same four legs.
    ``default_q`` ends in 12 or 16 joints; ``soft_limits`` ends in (12|16,2)
    and already includes the 0.9 soft-limit factor. ``symmetry_signs`` is a
    length-12 or length-16 tensor for the reward's geometric symmetry mapping.
    For M1's diagonal Rz(pi) symmetry this is +1 on all front-leg axes and
    -1 on all rear-leg axes; it does not change actuator/encoder coordinates.
    Mirror compares these signed absolute positions, not posture offsets.
    Every input must already be on the same Torch device and in radians/SI.
    """
    q = _field(s, "q")
    qd = _field(s, "qd")
    qdd = _field(s, "qdd")
    tau = _field(s, "tau")
    gravity = _field(s, "gravity")
    lin_vel = _field(s, "lin_vel")
    ang_vel = _field(s, "ang_vel")
    command = _field(s, "command")
    contact_forces = _field(s, "contact_forces")
    first_contact = _field(s, "first_contact")
    action = _field(s, "action")
    last_action = _field(s, "last_action")

    if q.ndim != 2 or q.shape[-1] != 16:
        raise ValueError("M1 rewards require q with shape (num_envs, 16)")
    if default_q.shape[-1] not in (12, 16):
        raise ValueError("default_q must end in 12 or 16 joints")
    if soft_limits.shape[-2:] not in ((12, 2), (16, 2)):
        raise ValueError("soft_limits must end in (12, 2) or (16, 2)")
    if symmetry_signs.ndim != 1 or symmetry_signs.shape[0] not in (12, 16):
        raise ValueError("symmetry_signs must be a 12- or 16-element vector")
    if contact_forces.shape != (q.shape[0], 3, 17, 3):
        raise ValueError("contact_forces must have shape (num_envs, 3, 17, 3)")
    if first_contact.shape != (q.shape[0], 17):
        raise ValueError("first_contact must have shape (num_envs, 17)")

    gate = (-gravity[:, 2]).clamp(0.0, 0.7) / 0.7
    command_norm = torch.linalg.vector_norm(command, dim=-1)
    standing = command_norm < 0.1
    leg_error = q[:, :12] - default_q[..., :12]
    posture = torch.linalg.vector_norm(leg_error, dim=-1)
    moving = (command_norm > 0.1) | (
        torch.linalg.vector_norm(lin_vel[:, :2], dim=-1) > 0.5
    )
    canonical_q = q[:, :12] * symmetry_signs[:12]
    mirror = (
        (canonical_q[:, :3] - canonical_q[:, 9:12]).square().sum(dim=-1)
        + (canonical_q[:, 3:6] - canonical_q[:, 6:9]).square().sum(dim=-1)
    ) * 0.5
    max_contact_force = torch.linalg.vector_norm(contact_forces, dim=-1).amax(dim=1)
    lower = soft_limits[..., :12, 0]
    upper = soft_limits[..., :12, 1]

    return {
        "lin_vel_z_l2": gate * lin_vel[:, 2].square(),
        "ang_vel_xy_l2": gate * ang_vel[:, :2].square().sum(dim=-1),
        "joint_torques_l2": tau[:, :12].square().sum(dim=-1),
        "joint_acc_l2": qdd[:, :12].square().sum(dim=-1),
        "joint_acc_wheel_l2": qdd[:, 12:].square().sum(dim=-1),
        "joint_pos_limits": (
            (lower - q[:, :12]).clamp_min(0.0) + (q[:, :12] - upper).clamp_min(0.0)
        ).sum(dim=-1),
        "joint_power": (qd[:, :12] * tau[:, :12]).abs().sum(dim=-1),
        "stand_still": gate * standing * leg_error.abs().sum(dim=-1),
        "joint_pos_penalty": gate * torch.where(moving, posture, 5.0 * posture),
        "joint_mirror": gate * mirror,
        "action_rate_l2": (action - last_action).square().sum(dim=-1),
        "undesired_contacts": gate
        * (max_contact_force[:, NONWHEEL_BODY_IDS] > 1.0).sum(dim=-1).to(q.dtype),
        "contact_forces": (max_contact_force[:, WHEEL_BODY_IDS] - 100.0)
        .clamp_min(0.0)
        .sum(dim=-1),
        "track_lin_vel_xy_exp": gate
        * torch.exp(-(command[:, :2] - lin_vel[:, :2]).square().sum(dim=-1) / 0.25),
        "track_ang_vel_z_exp": gate
        * torch.exp(-(command[:, 2] - ang_vel[:, 2]).square() / 0.25),
        "feet_contact_without_cmd": gate
        * standing
        * first_contact[:, WHEEL_BODY_IDS].sum(dim=-1).to(q.dtype),
        "upward": (1.0 - gravity[:, 2]).square(),
    }


class B2WReward(ManagerTermBase):
    """Select one term from the drive's once-per-control-step reward cache."""

    def __init__(self, cfg, env):
        super().__init__(env)
        if cfg.params.get("kind") not in REWARD_NAMES:
            raise ValueError(f"Unknown B2W reward kind: {cfg.params.get('kind')!r}")

    def __call__(self, env, kind: str) -> torch.Tensor:
        return env.action_manager.get_term("drive").reward_values()[kind]
