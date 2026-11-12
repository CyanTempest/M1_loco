"""Behavior checks on FullDrive with a tiny real MuJoCo motor plant.

Only the environment shell and deterministic RNG are test doubles. FullDrive,
EntityScene, torque dispatch, sensor reads and physical stepping are real.
The synthetic rig deliberately has zero gravity/contact so that a delayed
command's first physical effect is distinguishable from external motion.
"""

import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest
import torch
from m1_locomotion.contract import DEFAULT_POSITION, JOINTS, LEGS
from m1_unilab.full_drive import BODIES, FullDrive, FullDriveCfg
from unilab.base.entity import EntityCfg, EntityScene
from unilab.base.scene import SceneCfg
from unisim.backend.mujoco.backend import MuJoCoBackend


def write_motor_plant(path):
    root = ET.Element("mujoco", model="full_drive_behavior_test")
    ET.SubElement(
        root, "option", timestep=".002", gravity="0 0 0", integrator="implicitfast"
    )
    world = ET.SubElement(root, "worldbody")
    base = ET.SubElement(world, "body", name="BASE_LINK", pos="0 0 2")
    ET.SubElement(base, "freejoint")
    ET.SubElement(
        base,
        "geom",
        type="box",
        size=".15 .1 .05",
        mass="10",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(base, "site", name="imu")
    for index, leg in enumerate(LEGS):
        parent = base
        for kind in ("ABAD", "HIP", "KNEE", "FOOT"):
            pos = (
                f"{0.2 if index < 2 else -0.2} {0.15 if index % 2 else -0.15} 0"
                if kind == "ABAD"
                else "0 0 -.08"
            )
            body = ET.SubElement(parent, "body", name=f"{leg}_{kind}_LINK", pos=pos)
            ET.SubElement(
                body,
                "joint",
                name=f"{leg}_{kind}_JOINT",
                type="hinge",
                axis="1 0 0" if kind == "ABAD" else "0 1 0",
            )
            ET.SubElement(
                body,
                "geom",
                type="sphere",
                size=".025",
                mass=".2",
                contype="0",
                conaffinity="0",
            )
            parent = body
    actuators = ET.SubElement(root, "actuator")
    for index, joint in enumerate(JOINTS):
        limit = 180.0 if index < 12 else 28.68
        ET.SubElement(
            actuators,
            "motor",
            name=f"{joint}_act",
            joint=joint,
            gear="1",
            forcelimited="true",
            forcerange=f"{-limit} {limit}",
        )
    sensors = ET.SubElement(root, "sensor")
    ET.SubElement(sensors, "gyro", name="gyro", site="imu")
    ET.SubElement(sensors, "velocimeter", name="local_linvel", site="imu")
    for joint in JOINTS:
        ET.SubElement(sensors, "jointactuatorfrc", name=f"tau_{joint}", joint=joint)
    for body in BODIES:
        ET.SubElement(
            sensors,
            "contact",
            name=f"contact_{body}",
            body1=body,
            body2="world",
            reduce="netforce",
            data="force",
        )
    ET.ElementTree(root).write(path)


@pytest.fixture
def plant(tmp_path):
    model_path = tmp_path / "motor_plant.xml"
    write_motor_plant(model_path)
    scene_cfg = SceneCfg(
        model_file=str(model_path),
        entities={
            "robot": EntityCfg(
                root_body_name="BASE_LINK",
                body_names=tuple(BODIES),
                joint_names=tuple(JOINTS),
                actuator_names=tuple(f"{name}_act" for name in JOINTS),
            )
        },
    )
    backend = MuJoCoBackend(
        scene_cfg, 2, 0.002, base_name="BASE_LINK", add_body_sensors=True
    )
    backend.materialize()
    try:
        control = torch.zeros((2, 16))
        scene = EntityScene.from_scene_cfg(
            scene_cfg, backend, control_buffer=control, device="cpu"
        )
        env = SimpleNamespace(
            device="cpu",
            num_envs=2,
            scene=scene,
            backend=backend,
            cfg=SimpleNamespace(sim_dt=0.002, ctrl_dt=0.02),
            defer_finite_validation=False,
        )
        drive = FullDrive(FullDriveCfg(entity_name="robot", delay_max_steps=10), env)
        qpos = backend.get_state_views(("qpos",))["qpos"].clone()
        qpos[:, drive.q_indices] = torch.tensor(
            [DEFAULT_POSITION[name] for name in JOINTS]
        )
        qvel = backend.get_state_views(("qvel",))["qvel"].clone().zero_()
        backend.set_state_tensor(torch.arange(2), qpos, qvel)
        yield SimpleNamespace(
            drive=drive,
            backend=backend,
            control=control,
            default_qpos=qpos.clone(),
            default_qvel=qvel.clone(),
        )
    finally:
        backend.close()


def physical_step(plant):
    plant.drive.apply_actions()
    torque = plant.control.clone()  # Actual public entity control dispatch.
    plant.backend.step_tensor(plant.control, nsteps=1)
    velocity = plant.backend.get_state_views(("qvel",))["qvel"][
        :, plant.drive.v_indices
    ].clone()
    return torque, velocity


def test_zero_and_ten_physics_step_delays_change_when_leg_and_wheel_move(plant):
    drive = plant.drive
    drive.delay_steps[0] = 0
    drive.delay_steps[1] = 10
    actions = torch.zeros((2, 16))
    actions[:, (0, 12)] = 0.4  # ABAD position +.05 rad; wheel speed +2 rad/s.
    drive.process_actions(actions)  # One policy command, held for 11 physical steps.

    for step in range(11):
        torque, velocity = physical_step(plant)
        if step == 0:
            assert drive.delayed_targets[0, 0].item() == pytest.approx(0.05)
            assert drive.delayed_targets[0, 12].item() == pytest.approx(2.0)
            assert torch.all(torque[0, (0, 12)] > 0)
            assert torch.all(velocity[0, (0, 12)].abs() > 1e-4)
        if step < 10:
            # Initial delay slots contain the stand pose and zero wheel speed.
            torch.testing.assert_close(drive.delayed_targets[1], drive.default_q)
            torch.testing.assert_close(torque[1], torch.zeros(16), atol=1e-6, rtol=0)
            torch.testing.assert_close(velocity[1], torch.zeros(16), atol=1e-6, rtol=0)
        else:
            assert drive.delayed_targets[1, 0].item() == pytest.approx(0.05)
            assert drive.delayed_targets[1, 12].item() == pytest.approx(2.0)
            assert torch.all(torque[1, (0, 12)] > 0)
            assert torch.all(velocity[1, (0, 12)].abs() > 1e-4)
    assert drive.physics_tick == 11
    assert drive.wrench_applications == 11


class UpperEndpointRng:
    """Select the configured upper gains/delay, without reimplementing drive DR."""

    def uniform(self, low, high, shape):
        return torch.full(shape, float(high))

    def integers(self, low, high, shape):
        return torch.full(shape, high - 1, dtype=torch.long)


def test_randomized_wheel_kp_remains_zero_and_motor_torque_saturates(plant):
    drive = plant.drive
    rng = UpperEndpointRng()
    drive.randomize(None, rng)
    first_kp, first_kd = drive.kp.clone(), drive.kd.clone()
    drive.randomize(None, rng)
    # Gain DR is relative to nominal, not compounded every reset.
    torch.testing.assert_close(drive.kp, first_kp)
    torch.testing.assert_close(drive.kd, first_kd)
    torch.testing.assert_close(drive.kp[:, 12:], torch.zeros((2, 4)))
    torch.testing.assert_close(
        drive.kp[:, :12], 2 * drive.nominal_kp[:12].expand(2, -1)
    )
    torch.testing.assert_close(drive.kd, 2 * drive.nominal_kd.expand(2, -1))
    assert torch.all(drive.delay_steps == 10)
    drive.delay_steps.zero_()
    drive.force_local.zero_()
    drive.torque_local.zero_()

    # A multi-turn wheel position alone must produce no restoring torque.
    qpos = plant.default_qpos.clone()
    qpos[:, drive.q_indices[12:]] = 100
    plant.backend.set_state_tensor(torch.arange(2), qpos, plant.default_qvel)
    drive.process_actions(torch.zeros((2, 16)))
    torque, _ = physical_step(plant)
    torch.testing.assert_close(torque[:, 12:], torch.zeros((2, 4)))

    # Large commands must reach the actuator-specific physical torque caps.
    plant.backend.set_state_tensor(
        torch.arange(2), plant.default_qpos, plant.default_qvel
    )
    drive.reset()
    actions = torch.ones((2, 16)) * 1e6
    actions[1] *= -1
    drive.process_actions(actions)
    torque, _ = physical_step(plant)
    expected = torch.tensor([180.0] * 12 + [28.68] * 4)
    torch.testing.assert_close(torque[0], expected)
    torch.testing.assert_close(torque[1], -expected)
    # Validate force at the joint after the native motor, not just our buffer.
    actual = torch.cat(
        [plant.backend.get_sensor_view(f"tau_{joint}") for joint in JOINTS], dim=1
    )
    torch.testing.assert_close(actual, torque)


def test_subset_reset_flushes_delayed_command_without_touching_other_env(plant):
    drive = plant.drive
    drive.delay_steps[:] = 10
    actions = torch.zeros((2, 16))
    actions[0, (0, 12)] = 0.4
    actions[1, (0, 12)] = -0.4
    drive.process_actions(actions)
    for _ in range(14):  # Populate every slot and wrap the ring head.
        physical_step(plant)
    saved = {
        "history": drive.delay_history[:, 1].clone(),
        "raw": drive.raw_action[1].clone(),
        "targets": drive.targets[1].clone(),
        "delayed": drive.delayed_targets[1].clone(),
        "torque": drive.commanded_torque[1].clone(),
        "force_history": drive.force_history[1].clone(),
        "age": drive.contact_age[1].clone(),
    }
    old_head, old_tick = drive.delay_head, drive.physics_tick
    plant.backend.set_state_tensor(
        torch.tensor([0]), plant.default_qpos[:1], plant.default_qvel[:1]
    )
    drive.reset(torch.tensor([0]))
    assert drive.delay_head == old_head
    assert drive.physics_tick == old_tick
    torch.testing.assert_close(
        drive.delay_history[:, 0], drive.default_q.expand(11, -1)
    )
    torch.testing.assert_close(drive.delay_history[:, 1], saved["history"])
    for current, name in (
        (drive.raw_action, "raw"),
        (drive.targets, "targets"),
        (drive.delayed_targets, "delayed"),
        (drive.commanded_torque, "torque"),
        (drive.force_history, "force_history"),
        (drive.contact_age, "age"),
    ):
        torch.testing.assert_close(current[1], saved[name])
    assert drive.reset_pending.tolist() == [True, False]
    assert not drive.commanded_torque[0].any()
    assert not drive.force_history[0].any()
    assert not drive.contact_age[0].any()

    actions[0] = 0
    drive.process_actions(actions)
    for _ in range(12):  # Old nonzero commands must never reappear after reset.
        torque, velocity = physical_step(plant)
        torch.testing.assert_close(drive.delayed_targets[0], drive.default_q)
        torch.testing.assert_close(torque[0], torch.zeros(16), atol=1e-6, rtol=0)
        torch.testing.assert_close(velocity[0], torch.zeros(16), atol=1e-6, rtol=0)
        assert drive.delayed_targets[1, 12].item() == pytest.approx(-2.0)
