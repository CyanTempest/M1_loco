"""Analytic checks for the B2W equations and the public drive adapter."""

from types import SimpleNamespace

import pytest
import torch
from m1_unilab.full_rewards import REWARD_NAMES, B2WReward, reward_values


def snapshot(n=1):
    s = {
        name: torch.zeros((n, 16), dtype=torch.float64)
        for name in ("q", "qd", "qdd", "tau", "action", "last_action")
    }
    s.update(
        {
            name: torch.zeros((n, 3), dtype=torch.float64)
            for name in ("gravity", "lin_vel", "ang_vel", "command")
        }
    )
    s["gravity"][:, 2] = -1
    s["contact_forces"] = torch.zeros((n, 3, 17, 3), dtype=torch.float64)
    s["first_contact"] = torch.zeros((n, 17), dtype=torch.bool)
    return s


def calculate(s, default_q=None, limits=None, signs=None):
    return reward_values(
        s,
        torch.zeros(16, dtype=torch.float64) if default_q is None else default_q,
        torch.tensor([[-1.0, 1.0]] * 16, dtype=torch.float64)
        if limits is None
        else limits,
        torch.ones(16, dtype=torch.float64) if signs is None else signs,
    )


def test_all_seventeen_values_are_unweighted_and_not_dt_scaled():
    out = calculate(snapshot(23))  # More environments than joints; no row slicing.
    assert tuple(out) == REWARD_NAMES
    assert len(out) == 17
    for name, value in out.items():
        assert value.shape == (23,)
        assert value.dtype == torch.float64
        expected = (
            4.0 if name == "upward" else (1.0 if name.startswith("track_") else 0.0)
        )
        torch.testing.assert_close(value, torch.full_like(value, expected))


def test_upright_gate_partial_tilt_and_inversion():
    s = snapshot(3)
    s["gravity"][:, 2] = torch.tensor([-1.0, -0.35, 1.0])
    s["lin_vel"][:, 2] = 2
    s["ang_vel"][:, :2] = torch.tensor([3.0, 4.0])
    s["tau"][:, 0] = 2
    out = calculate(s)
    torch.testing.assert_close(
        out["lin_vel_z_l2"], torch.tensor([4.0, 2.0, 0.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        out["ang_vel_xy_l2"], torch.tensor([25.0, 12.5, 0.0], dtype=torch.float64)
    )
    for name in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"):
        torch.testing.assert_close(
            out[name], torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64)
        )
    torch.testing.assert_close(
        out["upward"], torch.tensor([4.0, 1.35**2, 0.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        out["joint_torques_l2"], torch.full((3,), 4.0, dtype=torch.float64)
    )


def test_actual_torque_power_and_acceleration_select_correct_joints():
    s = snapshot()
    s["tau"][0, 0] = -2
    s["qd"][0, 0] = 3
    s["tau"][0, 12:] = 1000
    s["qd"][0, 12:] = 1000
    s["qdd"][0, 0:2] = torch.tensor([3.0, 4.0])
    s["qdd"][0, 12:14] = torch.tensor([6.0, 8.0])
    # Actions are unrelated to torque. Inverting the robot must not gate these.
    s["action"][:] = 17
    s["gravity"][:, 2] = 1
    out = calculate(s)
    assert out["joint_torques_l2"].item() == 4
    assert out["joint_power"].item() == 6
    assert out["joint_acc_l2"].item() == 25
    assert out["joint_acc_wheel_l2"].item() == 100
    assert out["action_rate_l2"].item() == 16 * 17**2


def test_standstill_and_posture_strict_thresholds_and_l2_norm():
    s = snapshot(4)
    s["q"][:, :2] = torch.tensor([3.0, 4.0])
    s["command"][:, 0] = torch.tensor([0.0, 0.1, 0.1001, 0.0], dtype=torch.float64)
    s["lin_vel"][1, 0] = 0.5
    s["lin_vel"][3, 0] = 0.5001
    out = calculate(s)
    torch.testing.assert_close(
        out["joint_pos_penalty"],
        torch.tensor([25.0, 25.0, 5.0, 5.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        out["stand_still"], torch.tensor([7.0, 0.0, 0.0, 7.0], dtype=torch.float64)
    )


def test_mirror_maps_m1_front_rear_signs_and_averages_diagonals():
    s = snapshot()
    default = torch.tensor(
        [0.0, 0.65, -1.3] * 2 + [0.0, -0.65, 1.3] * 2, dtype=torch.float64
    )
    signs = torch.tensor(
        [1.0, 1.0, 1.0] * 2 + [-1.0, -1.0, -1.0] * 2, dtype=torch.float64
    )
    s["q"][0, :12] = default
    assert calculate(s, default_q=default, signs=signs)["joint_mirror"].item() == 0
    assert calculate(s, default_q=default)["joint_mirror"].item() > 0
    s["q"][0, 1] += 0.2  # one component of one pair
    out = calculate(s, default_q=default, signs=signs)
    assert out["joint_mirror"].item() == pytest.approx(0.2**2 / 2)
    # Absolute canonical joint angles are compared, even if q equals q0.
    default = s["q"][0, :12].clone()
    out = calculate(s, default_q=default, signs=signs)
    assert out["joint_pos_penalty"].item() == 0
    assert out["joint_mirror"].item() == pytest.approx(0.2**2 / 2)


def test_diagonal_geometry_mapping_flips_rear_abad_too():
    s = snapshot()
    s["q"][0, :12] = torch.tensor(
        [0.2, 0.65, -1.3, -0.3, 0.7, -1.4, 0.3, -0.7, 1.4, -0.2, -0.65, 1.3],
        dtype=torch.float64,
    )
    signs = torch.tensor([1.0] * 6 + [-1.0] * 6, dtype=torch.float64)
    assert calculate(s, signs=signs)["joint_mirror"].item() == 0
    signs[6] = signs[9] = 1
    assert calculate(s, signs=signs)["joint_mirror"].item() == pytest.approx(
        (0.4**2 + 0.6**2) / 2
    )


def test_soft_limit_violation_is_linear_and_excludes_wheels():
    s = snapshot()
    s["q"][0, :4] = torch.tensor([-1.2, 1.3, -1.0, 1.0], dtype=torch.float64)
    s["q"][0, 12:] = 10000
    assert calculate(s)["joint_pos_limits"].item() == pytest.approx(0.5)
    # Caller supplies already-softened limits; no second 0.9 multiplication.
    limits = torch.tensor([[-0.9, 0.9]] * 12, dtype=torch.float64)
    s["q"][0, :12] = 0.9
    assert calculate(s, limits=limits)["joint_pos_limits"].item() == 0


def test_contact_history_norm_thresholds_body_selection_and_linear_excess():
    s = snapshot()
    f = s["contact_forces"]
    f[0, 2, 0, 2] = 1  # strict >1, so excluded
    f[0, 0, 1, :2] = torch.tensor([3.0, 4.0])  # old sample still counts
    f[0, 1, 2, 0] = 1.00001
    f[0, 0, 4, 2] = 150
    f[0, 1, 8, 2] = 100  # exact threshold: no excess
    f[0, 2, 12, :2] = torch.tensor([-90.0, -120.0])
    out = calculate(s)
    assert out["undesired_contacts"].item() == 2
    assert out["contact_forces"].item() == 100
    s["gravity"][0, 2] = 1
    out = calculate(s)
    assert out["undesired_contacts"].item() == 0
    assert out["contact_forces"].item() == 100  # no upright gate


def test_no_command_contact_is_first_event_not_persistent_contact():
    s = snapshot(3)
    s["contact_forces"][:, :, (4, 8, 12, 16), 2] = 150
    s["first_contact"][0, (4, 8)] = True
    s["first_contact"][0, 0] = True  # non-wheel never counts
    s["first_contact"][1, (4, 8, 12, 16)] = True
    s["command"][1, 0] = 0.1  # strict <0.1
    s["gravity"][0, 2] = -0.35
    torch.testing.assert_close(
        calculate(s)["feet_contact_without_cmd"],
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
    )


def test_tracking_uses_body_xy_and_yaw_and_sigma_squared():
    s = snapshot()
    s["command"][0] = torch.tensor([0.3, 0.4, 0.5])
    s["lin_vel"][0, 2] = 99  # z is not in the xy tracking error
    s["ang_vel"][0, :2] = 99  # roll/pitch not in the yaw error
    out = calculate(s)
    assert out["track_lin_vel_xy_exp"].item() == pytest.approx(
        torch.exp(torch.tensor(-1.0)).item()
    )
    assert out["track_ang_vel_z_exp"].item() == pytest.approx(
        torch.exp(torch.tensor(-1.0)).item()
    )


def test_snapshot_object_inputs_are_not_mutated():
    s = snapshot(2)
    before = {name: value.clone() for name, value in s.items()}
    calculate(SimpleNamespace(**s))
    for name, value in s.items():
        torch.testing.assert_close(value, before[name])


def test_adapter_selects_public_drive_cached_value():
    cached = calculate(snapshot(2))
    requested = []
    drive = SimpleNamespace(reward_values=lambda: cached)

    def get_term(name):
        requested.append(name)
        return drive

    env = SimpleNamespace(num_envs=2, action_manager=SimpleNamespace(get_term=get_term))
    for kind in REWARD_NAMES:
        term = B2WReward(SimpleNamespace(params={"kind": kind}), env)
        assert term(env, kind=kind) is cached[kind]
    assert requested == ["drive"] * 17
    with pytest.raises(ValueError, match="Unknown B2W reward"):
        B2WReward(SimpleNamespace(params={"kind": "not_a_reward"}), env)


def test_wrong_contact_shape_fails_instead_of_silently_changing_history():
    s = snapshot()
    s["contact_forces"] = s["contact_forces"][:, :1]
    with pytest.raises(ValueError, match="contact_forces"):
        calculate(s)
