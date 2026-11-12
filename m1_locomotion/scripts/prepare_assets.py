#!/usr/bin/env python3
"""Reproducibly normalize the supplied M1 URDF; no external downloads."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def prepare(root=ROOT):
    asset = root / "assets/m1"
    source = asset / "original.urdf"
    source_mass = sum(float(x.get("value")) for x in ET.parse(source).findall("link/inertial/mass"))
    report = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "source_mass_kg": source_mass,
              "changes": ["FOOT joints changed to continuous; effort/velocity retained",
                          "Relative mesh paths normalized",
                          "12 sensor-only BASE_LINK mesh collisions removed; 11 original base primitive colliders retained"],
              "profiles": {}}
    for profile, target_mass in (("m1_cad", source_mass),):
        tree = ET.parse(source)
        robot = tree.getroot()
        robot.set("name", f"M1_{profile}")
        for mesh in robot.findall(".//mesh"):
            filename = Path(mesh.get("filename")).name
            if not (asset / "meshes" / filename).is_file():
                raise FileNotFoundError(filename)
            mesh.set("filename", f"meshes/{filename}")
        base = robot.find("link[@name='BASE_LINK']")
        removed = 0
        for collision in list(base.findall("collision")):
            if collision.find("geometry/mesh") is not None:
                base.remove(collision)
                removed += 1
        assert removed == 12, "Unexpected source model; inspect before changing collisions"
        for joint in robot.findall("joint"):
            if joint.get("name").endswith("_FOOT_JOINT"):
                joint.set("type", "continuous")
                joint.find("limit").attrib.pop("lower", None)
                joint.find("limit").attrib.pop("upper", None)
        # Preserve all original masses, centers of mass and inertia tensors verbatim.
        ET.indent(tree, space="  ")
        output = asset / f"{profile}.urdf"
        tree.write(output, encoding="utf-8", xml_declaration=True)
        report["profiles"][profile] = {"mass_kg": target_mass, "mass_inertia_scale": 1.0,
                                         "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    (asset / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
