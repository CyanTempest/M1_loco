"""CPU checks of topology, inertias, joint limits and the nominal standing pose."""
import xml.etree.ElementTree as ET
import numpy as np

from .contract import DEFAULT_POSITION, JOINTS, LEG_JOINTS, ROOT, WHEEL_JOINTS, WHEEL_LINKS


def rotation(rpy):
    r, p, y = map(float, rpy)
    cr, cp, cy = np.cos([r, p, y])
    sr, sp, sy = np.sin([r, p, y])
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def forward_kinematics(root, joint_positions=None):
    q = DEFAULT_POSITION if joint_positions is None else joint_positions
    frames = {"BASE_LINK": np.eye(4)}
    pending = list(root.findall("joint"))
    while pending:
        progress = False
        for joint in pending[:]:
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent not in frames:
                continue
            transform = np.eye(4)
            origin = joint.find("origin")
            if origin is not None:
                transform[:3, :3] = rotation(origin.get("rpy", "0 0 0").split())
                transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            if joint.get("type") != "fixed":
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                axis /= np.linalg.norm(axis)
                x, y, z = axis
                cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                angle = q[joint.get("name")]
                turn = np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*(cross@cross)
                transform[:3, :3] = transform[:3, :3] @ turn
            frames[child] = frames[parent] @ transform
            pending.remove(joint)
            progress = True
        if not progress:
            raise ValueError("Disconnected or cyclic kinematic tree")
    return frames


def audit_model():
    path = ROOT / "assets/m1/m1_cad.urdf"
    root = ET.parse(path).getroot()
    joints = {j.get("name"): j for j in root.findall("joint") if j.get("type") != "fixed"}
    assert set(joints) == set(JOINTS), "Expected exactly 12 leg + 4 wheel joints"
    masses = []
    min_eigenvalue = float("inf")
    for link in root.findall("link"):
        mass = float(link.find("inertial/mass").get("value"))
        assert mass > 0
        masses.append(mass)
        v = {k: float(x) for k, x in link.find("inertial/inertia").attrib.items()}
        tensor = np.array([[v['ixx'], v['ixy'], v['ixz']], [v['ixy'], v['iyy'], v['iyz']],
                           [v['ixz'], v['iyz'], v['izz']]])
        eig = np.linalg.eigvalsh(tensor)
        assert eig[0] > 0, f"Non-positive inertia in {link.get('name')}"
        assert eig[0] + eig[1] >= eig[2] - 1e-8, f"Invalid principal moments in {link.get('name')}"
        min_eigenvalue = min(min_eigenvalue, eig[0])
    for name in LEG_JOINTS:
        limit = joints[name].find("limit")
        assert float(limit.get("lower")) < DEFAULT_POSITION[name] < float(limit.get("upper"))
    for name in WHEEL_JOINTS:
        assert joints[name].get("type") == "continuous"
    for mesh in root.findall(".//mesh"):
        assert (path.parent / mesh.get("filename")).is_file()
    frames = forward_kinematics(root)
    centers = np.array([frames[n][:3, 3] for n in WHEEL_LINKS])
    assert np.ptp(centers[:, 2]) < 1e-6, "Standing wheel centers must have equal height"
    axes = np.array([frames[n][:3, :3] @ np.array([0, 1, 0]) for n in WHEEL_LINKS])
    assert np.allclose(axes, [0, 1, 0]), "Wheel axes must align with body +Y at stand"
    assert abs(sum(masses) - 42.338) < 1e-7
    return {"profile": "m1_cad", "links": len(masses), "actuated_joints": len(joints),
            "mass_kg": sum(masses), "minimum_inertia_eigenvalue": min_eigenvalue,
            "wheel_centers_body_m": centers.tolist(), "standing_base_height_m": float(-centers[0, 2] + .0975)}
