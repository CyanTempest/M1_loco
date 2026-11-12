"""Convert the prepared M1 URDF to native MuJoCo position/velocity servos."""

import json
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from m1_locomotion.contract import DEFAULT_POSITION, JOINTS, LEG_JOINTS, ROOT
from m1_locomotion.gains import load_gains

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def build_assets(profile="basic") -> dict[str, Path]:
    if profile not in ("basic", "b2w-full"):
        raise ValueError(profile)
    full = profile == "b2w-full"
    suffix = "_full" if full else ""
    timestep = "0.001" if full else "0.002"
    integrator = "implicit" if full else "implicitfast"
    source = ROOT / "assets/m1/m1_cad.urdf"
    out = PACKAGE_ROOT / "assets"
    out.mkdir(exist_ok=True)
    if full:
        (out / "solver_full.xml").write_text(
            '<mujoco><option integrator="implicit"><flag autoreset="disable" '
            'nativeccd="disable"/></option></mujoco>\n'
        )
    urdf = ET.parse(source).getroot()
    # Keep physics compilation lean. Playback independently restores the original
    # URDF visual meshes through rendering.original_visual_model.
    for link in urdf.findall("link"):
        for visual in list(link.findall("visual")):
            link.remove(visual)
    for mesh in urdf.findall(".//mesh"):
        mesh.set("filename", str((source.parent / mesh.get("filename")).resolve()))
    ET.SubElement(urdf, "link", name="world")
    joint = ET.SubElement(urdf, "joint", name="floating_base", type="floating")
    ET.SubElement(joint, "parent", link="world")
    ET.SubElement(joint, "child", link="BASE_LINK")
    model = mujoco.MjModel.from_xml_string(ET.tostring(urdf, encoding="unicode"))
    with tempfile.TemporaryDirectory(prefix="m1_unilab_") as tmp:
        converted = Path(tmp) / "converted.xml"
        mujoco.mj_saveLastXML(str(converted), model)
        robot = ET.parse(converted).getroot()
    robot.set("model", "M1")
    meshdir = os.path.relpath(source.parent / "meshes", out)
    robot.find("compiler").set("meshdir", meshdir)
    for mesh in robot.findall("asset/mesh"):
        mesh.set("file", Path(mesh.get("file")).name)
    option = robot.find("option")
    if option is None:
        option = ET.SubElement(robot, "option")
    option.attrib.update(timestep=timestep, integrator=integrator, gravity="0 0 -9.81")
    for geom in robot.findall(".//geom"):
        geom.set("friction", "0.8 0.005 0.0001")
        geom.set("rgba", "0.3 0.4 0.5 1")
    if full:
        for body in robot.findall(".//body"):
            for i, geom in enumerate(body.findall("geom")):
                geom.set("name", f"{body.get('name')}_collision_{i}")
                # Robot/world contact only; robot DR owns contact material.
                geom.set("contype", "1")
                geom.set("conaffinity", "2")
                geom.set("priority", "1")
    base = robot.find(".//body[@name='BASE_LINK']")
    ET.SubElement(base, "site", name="imu", size="0.01", pos="0 0 0")
    sensor = ET.SubElement(robot, "sensor")
    ET.SubElement(sensor, "gyro", name="gyro", site="imu")
    ET.SubElement(sensor, "velocimeter", name="local_linvel", site="imu")
    ET.SubElement(sensor, "framezaxis", name="upvector", objtype="site", objname="imu")
    if full:
        for name in JOINTS:
            ET.SubElement(sensor, "jointactuatorfrc", name=f"tau_{name}", joint=name)
        from .full_drive import BODIES

        for name in BODIES:
            ET.SubElement(
                sensor,
                "contact",
                name=f"contact_{name}",
                body1=name,
                data="force",
                reduce="netforce",
            )
    gains = load_gains("user_wheels")
    physical = json.loads((ROOT / "config/parameter_sources.json").read_text())[
        "m1_cad"
    ]
    actuator = ET.SubElement(robot, "actuator")
    for name in JOINTS:
        leg = name in LEG_JOINTS
        limit = physical["leg_effort_limit_Nm" if leg else "wheel_effort_limit_Nm"]
        attributes = {
            "name": name + "_act",
            "joint": name,
            "forcelimited": "true",
            "forcerange": f"{-limit} {limit}",
        }
        if full:
            ET.SubElement(
                actuator,
                "motor",
                **attributes,
                gear="1",
                ctrllimited="true",
                ctrlrange=f"{-limit} {limit}",
            )
        elif leg:
            ET.SubElement(
                actuator,
                "position",
                **attributes,
                kp=str(gains[name]["kp"]),
                kv=str(gains[name]["kd"]),
            )
        else:
            if gains[name]["kp"] != 0:
                raise ValueError("UniLab wheel velocity servos require kp=0")
            ET.SubElement(
                actuator,
                "velocity",
                **attributes,
                kv=str(gains[name]["kd"]),
                ctrllimited="true",
                ctrlrange="-110 110",
            )
    robot_path = out / f"robot{suffix}.xml"
    ET.indent(robot)
    ET.ElementTree(robot).write(robot_path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(robot_path))
    qpos = model.qpos0.copy()
    qpos[2] = 0.54
    for name in JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos[model.jnt_qposadr[jid]] = DEFAULT_POSITION[name]
    ctrl = [DEFAULT_POSITION[name] if name in LEG_JOINTS else 0.0 for name in JOINTS]
    if full:
        ctrl = [0.0] * 16
    paths = {}
    for terrain in ("flat", "mixed"):
        scene = ET.Element("mujoco", model=f"M1_{terrain}")
        ET.SubElement(scene, "compiler", meshdir=meshdir, angle="radian")
        ET.SubElement(
            scene,
            "option",
            timestep=timestep,
            integrator=integrator,
            gravity="0 0 -9.81",
        )
        ET.SubElement(scene, "include", file=robot_path.name)
        world = ET.SubElement(scene, "worldbody")
        ET.SubElement(world, "light", pos="0 0 6", directional="true", dir="0 0 -1")
        if terrain == "flat":
            ET.SubElement(
                world,
                "geom",
                name="floor",
                type="plane",
                size="100 100 .1",
                friction="1 0.005 0.0001" if full else "0.8 0.005 0.0001",
                contype="2" if full else "1",
                conaffinity="1",
                rgba=".65 .65 .65 1",
            )
        keyframes = ET.SubElement(scene, "keyframe")
        ET.SubElement(
            keyframes,
            "key",
            name="home",
            qpos=" ".join(map(str, qpos)),
            ctrl=" ".join(map(str, ctrl)),
        )
        paths[terrain] = out / f"scene_{terrain}{suffix}.xml"
        ET.indent(scene)
        ET.ElementTree(scene).write(paths[terrain], encoding="unicode")
        loaded = mujoco.MjModel.from_xml_path(str(paths[terrain]))
        assert loaded.nu == 16
        assert np.isclose(loaded.body_mass.sum(), physical["mass_kg"], atol=1e-5)
    return paths
