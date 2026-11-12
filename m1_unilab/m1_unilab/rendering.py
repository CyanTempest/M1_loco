"""Original M1 URDF appearance on a detached playback model."""

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from m1_locomotion.contract import ROOT
from unisim.utils.math import np_quat_from_euler_xyz


@lru_cache(maxsize=32)
def stl_triangles(path):
    """Read binary CAD STL without MuJoCo's 200k-face STL decoder limit.

    Pass every original triangle as mesh data; no decimation or shape changes.
    """
    raw = Path(path).read_bytes()
    count = int.from_bytes(raw[80:84], "little")
    if len(raw) != 84 + 50 * count:
        raise ValueError(f"Unexpected binary STL structure: {path}")
    dtype = np.dtype(
        [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
    )
    triangles = np.frombuffer(raw, dtype=dtype, offset=84, count=count)["vertices"]
    vertices = np.ascontiguousarray(triangles.reshape(-1, 3))
    return vertices.ravel(), np.arange(count * 3, dtype=np.int32)


def original_visual_model(backend):
    """Add the supplied URDF's visual meshes without changing training physics."""
    scene = backend.get_scene_visual_model_file() or backend.get_scene_model_file()
    spec = mujoco.MjSpec.from_file(scene)
    urdf = ET.parse(ROOT / "assets/m1/original.urdf").getroot()
    count = 0
    for link in urdf.findall("link"):
        visuals = link.findall("visual")
        if not visuals:
            continue
        body = spec.body(link.get("name"))
        if body is None:
            raise ValueError(f"Missing render body for {link.get('name')}")
        # Hide original robot colliders only in the rendering option's group 3.
        for geom in body.geoms:
            geom.group = 3
        for index, visual in enumerate(visuals):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                raise ValueError("Expected the supplied M1 visual mesh geometry")
            mesh_file = ROOT / "assets/m1/meshes" / Path(mesh.get("filename")).name
            mesh_name = f"m1_visual_{link.get('name')}_{index}"
            vertices, faces = stl_triangles(str(mesh_file))
            spec.add_mesh(
                name=mesh_name,
                uservert=vertices,
                userface=faces,
                scale=np.fromstring(mesh.get("scale", "1 1 1"), sep=" "),
            )
            origin = visual.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            color = visual.find("material/color")
            rgba = (
                np.fromstring(color.get("rgba"), sep=" ")
                if color is not None
                else [0.7, 0.7, 0.7, 1]
            )
            body.add_geom(
                name=mesh_name,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                pos=xyz,
                quat=np_quat_from_euler_xyz(*rpy),
                rgba=rgba,
                group=2,
                contype=0,
                conaffinity=0,
                mass=0,
                density=0,
            )
            count += 1
    assert count == 17, f"Unexpected original visual count: {count}"
    model = spec.compile()
    physical = backend.get_playback_model(env_index=0)
    assert (model.nq, model.nv, model.nbody, model.njnt) == (
        physical.nq,
        physical.nv,
        physical.nbody,
        physical.njnt,
    )
    for field in (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "body_pos",
        "body_quat",
        "jnt_type",
        "jnt_qposadr",
        "jnt_dofadr",
        "jnt_pos",
        "jnt_axis",
        "jnt_range",
    ):
        np.testing.assert_allclose(
            getattr(model, field), getattr(physical, field), rtol=1e-6, atol=1e-9
        )
    for i in range(model.njnt):
        assert model.joint(i).name == physical.joint(i).name
    return model
