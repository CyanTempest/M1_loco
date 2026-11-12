#!/usr/bin/env python3
"""CPU model sanity check: stand, then roll at fixed leg pose. Not an RL policy evaluation.

Uses the prepared URDF's collision geometry and inertia with a floating base.
MuJoCo contact dynamics and this zero-delay PD check do not validate Isaac PhysX training.
"""
import argparse
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from m1_locomotion.contract import DEFAULT_POSITION, JOINTS, ROOT, WHEEL_LINKS
from m1_locomotion.gains import load_gains


def build_model(temp_dir):
    path = ROOT / "assets/m1/m1_cad.urdf"
    robot = ET.parse(path).getroot()
    # Only collision geometry is required for this check.
    for link in robot.findall("link"):
        for visual in list(link.findall("visual")):
            link.remove(visual)
    for mesh in robot.findall(".//mesh"):
        mesh.set("filename", str(path.parent / mesh.get("filename")))
    ET.SubElement(robot, "link", name="world")
    joint = ET.SubElement(robot, "joint", name="floating_base", type="floating")
    ET.SubElement(joint, "parent", link="world")
    ET.SubElement(joint, "child", link="BASE_LINK")
    imported = mujoco.MjModel.from_xml_string(ET.tostring(robot, encoding="unicode"))
    mjcf_path = Path(temp_dir) / "imported.xml"
    mujoco.mj_saveLastXML(str(mjcf_path), imported)
    mjcf = ET.parse(mjcf_path).getroot()
    ET.SubElement(mjcf, "option", timestep="0.001", integrator="implicitfast", gravity="0 0 -9.81")
    ET.SubElement(mjcf.find("worldbody"), "geom", name="ground", type="plane", size="20 20 .1",
                  friction="0.8 0.005 0.0001", rgba=".7 .7 .7 1")
    return mujoco.MjModel.from_xml_string(ET.tostring(mjcf, encoding="unicode"))


def run(gain_profile="user_wheels"):
    with tempfile.TemporaryDirectory(prefix="m1_cpu_") as tmp:
        model = build_model(tmp)
    data = mujoco.MjData(model)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS]
    qids = model.jnt_qposadr[joint_ids]
    vids = model.jnt_dofadr[joint_ids]
    q0 = np.array([DEFAULT_POSITION[n] for n in JOINTS])
    from m1_locomotion.audit import audit_model
    data.qpos[2] = audit_model()["standing_base_height_m"] + 0.015
    data.qpos[qids] = q0
    mujoco.mj_forward(model, data)
    gains = load_gains(gain_profile)
    kp = np.array([gains[n]["kp"] for n in JOINTS])
    kd = np.array([gains[n]["kd"] for n in JOINTS])
    limits = np.array([180.0]*12 + [28.68]*4)
    max_torque = np.zeros(16)
    min_height = 10.0
    max_tilt = 0.0
    nonwheel_ground_contacts = set()
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "BASE_LINK")
    rolling_start_x = None
    for _ in range(round(10.0 / model.opt.timestep)):
        desired_velocity = np.zeros(16)
        if data.time >= 5.0:
            if rolling_start_x is None:
                rolling_start_x = float(data.qpos[0])
            desired_velocity[12:] = 5.0
        # The configured wheel kp is zero, so wheel angle does not produce torque.
        torque = np.clip(kp * (q0 - data.qpos[qids]) + kd * (desired_velocity - data.qvel[vids]), -limits, limits)
        data.qfrc_applied[vids] = torque
        mujoco.mj_step(model, data)
        max_torque = np.maximum(max_torque, np.abs(torque))
        min_height = min(min_height, float(data.qpos[2]))
        max_tilt = max(max_tilt, float(np.arccos(np.clip(data.xmat[base_id].reshape(3, 3)[2, 2], -1, 1))))
        for contact in data.contact:
            pair = (contact.geom1, contact.geom2)
            if ground_id in pair:
                other = pair[1] if pair[0] == ground_id else pair[0]
                body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[other])
                if body not in WHEEL_LINKS:
                    nonwheel_ground_contacts.add(body)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError("Non-finite physics state")
    displacement = float(data.qpos[0]) - rolling_start_x
    result = {"gain_profile": gain_profile, "wheel_gains": {n: gains[n] for n in JOINTS[12:]},
              "wheel_control": "pure velocity feedback; no wheel position target ramp",
              "engine": f"MuJoCo {mujoco.__version__}", "simulated_seconds": data.time,
              "mass_kg": float(model.body_mass.sum()), "min_base_height_m": min_height,
              "final_base_height_m": float(data.qpos[2]), "forward_displacement_over_last_5s_m": displacement,
              "max_leg_torque_Nm": float(max_torque[:12].max()), "max_wheel_torque_Nm": float(max_torque[12:].max()),
              "max_tilt_rad": max_tilt, "nonwheel_ground_contacts": sorted(nonwheel_ground_contacts),
              "scope": "nominal PD asset test only; no trained policy, no domain randomization or actuator delay"}
    assert min_height > 0.3, result
    assert max_tilt < 0.5, result
    assert not nonwheel_ground_contacts, result
    assert displacement > 0.3, result
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gain-profile", choices=("preview", "user_wheels"), default="user_wheels")
    args = parser.parse_args()
    text = json.dumps(run(args.gain_profile), indent=2) + "\n"
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
