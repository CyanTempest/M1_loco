#!/usr/bin/env python3
"""Render actual pinned Isaac Lab terrain generators without importing Isaac Sim.

Loads only the named pure geometry functions using AST, unchanged; all configuration
arguments come from this project's terrain_cfg. This is a mesh preview, not a physics run.
Requires numpy, scipy, trimesh and matplotlib and an IsaacLab v2.3.2 source checkout.
"""
import argparse
import ast
import copy
import functools
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from collections.abc import Callable

import numpy as np
import scipy.interpolate as interpolate
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]


def load_functions(path, names, namespace):
    parsed = ast.parse(path.read_text())
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == set(names), path
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + functions, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)


def config_facade(type_name):
    def make(**kwargs):
        defaults = dict(size=(8.0, 8.0), horizontal_scale=.1, vertical_scale=.005,
                        slope_threshold=.75, downsampled_scale=None, border_width=0.0,
                        inverted="Inverted" in type_name)
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)
    return make


def preview(isaac_root, output, difficulty):
    revision = subprocess.check_output(["git", "-C", str(isaac_root), "rev-parse", "HEAD"], text=True).strip()
    assert revision == "37ddf626871758333d6ed89cf64ad702aef127d0", revision
    terrain_path = isaac_root / "source/isaaclab/isaaclab/terrains"
    namespace = dict(np=np, trimesh=trimesh, interpolate=interpolate, copy=copy,
                     functools=functools, Callable=Callable)
    sources = {
        terrain_path / "trimesh/utils.py": ("make_plane", "make_border"),
        terrain_path / "height_field/utils.py": ("height_field_to_mesh", "convert_height_field_to_mesh"),
        terrain_path / "trimesh/mesh_terrains.py": ("flat_terrain", "pyramid_stairs_terrain", "inverted_pyramid_stairs_terrain"),
        terrain_path / "height_field/hf_terrains.py": ("random_uniform_terrain", "pyramid_sloped_terrain"),
    }
    for path, names in sources.items():
        load_functions(path, names, namespace)
    config_names = ("MeshPlaneTerrainCfg", "HfInvertedPyramidSlopedTerrainCfg", "HfPyramidSlopedTerrainCfg",
                    "HfRandomUniformTerrainCfg", "MeshInvertedPyramidStairsTerrainCfg", "MeshPyramidStairsTerrainCfg")
    cfg_namespace = {"terrain_gen": SimpleNamespace(**{name: config_facade(name) for name in config_names}),
                     "TerrainGeneratorCfg": lambda **kwargs: SimpleNamespace(**kwargs)}
    load_functions(ROOT / "m1_locomotion/terrains.py", ["terrain_cfg"], cfg_namespace)
    config = cfg_namespace["terrain_cfg"]()
    panels = [
        ("flat", "flat_terrain", "FLAT", "20% of training tiles"),
        ("slope_up", "pyramid_sloped_terrain", "UPHILL", "10% | slope ratio 0.00–0.35"),
        ("slope_down", "pyramid_sloped_terrain", "DOWNHILL", "10% | slope ratio 0.00–0.35"),
        ("rough", "random_uniform_terrain", "ROUGH GROUND", "20% | height samples 0.5–4 cm"),
        ("stairs_up", "inverted_pyramid_stairs_terrain", "STAIRS UP", "20% | rise 3–16 cm / tread 30 cm"),
        ("stairs_down", "pyramid_stairs_terrain", "STAIRS DOWN", "20% | rise 3–16 cm / tread 30 cm"),
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": "#203247"})
    fig = plt.figure(figsize=(15, 9), facecolor="#f5f7fa")
    fig.suptitle("M1  /  TRAINING TERRAIN GEOMETRY", x=.045, y=.975, ha="left", fontsize=22, fontweight="bold")
    fig.text(.045, .925, f"Isaac Lab v2.3.2 generators  •  difficulty {difficulty:.2f}  •  8 × 8 m tiles  •  z displayed 2× for readability",
             fontsize=11, color="#596b7e")
    manifest = {"isaaclab_commit": revision, "difficulty": difficulty, "seed": config.seed,
                "sources_sha256": {str(path.relative_to(isaac_root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                "terrain_config_sha256": hashlib.sha256((ROOT / "m1_locomotion/terrains.py").read_bytes()).hexdigest(), "tiles": {}}
    for i, (key, function, title, subtitle) in enumerate(panels):
        cfg = config.sub_terrains[key]
        np.random.seed(config.seed + i)
        meshes, origin = namespace[function](difficulty, cfg)
        mesh = trimesh.util.concatenate(meshes)
        mesh.apply_translation(-origin)
        # Only exterior upward-facing and vertical surfaces; omit hidden undersides.
        keep = mesh.face_normals[:, 2] > -.1
        faces = mesh.triangles[keep]
        normals = mesh.face_normals[keep]
        ax = fig.add_subplot(2, 3, i+1, projection="3d", facecolor="#f5f7fa", computed_zorder=False)
        z = faces.mean(axis=1)[:, 2]
        spread = max(float(np.ptp(z)), .005)
        colors = plt.get_cmap("viridis")((z-z.min())/spread * .65 + .15)
        shade = .55 + .45 * LightSource(azdeg=315, altdeg=50).shade_normals(normals)
        colors[:, :3] *= shade[:, None]
        ax.add_collection3d(Poly3DCollection(faces, facecolors=colors, edgecolors="none", rasterized=True, zorder=1))
        ax.scatter([0], [0], [.025], color="#e85c40", s=32, depthshade=False, zorder=10)
        ax.quiver(0, 0, .04, 1.4, 0, 0, color="#e85c40", linewidth=2, arrow_length_ratio=.18, zorder=11)
        ax.set(xlim=(-4, 4), ylim=(-4, 4), zlim=(-1.4, 1.4), xticks=[-4, 0, 4], yticks=[-4, 0, 4], zticks=[-1, 0, 1])
        ax.set_box_aspect((8, 8, 5.6))
        ax.view_init(elev=48, azim=-57)
        ax.set_xlabel("x (m)", labelpad=-3); ax.set_ylabel("y (m)", labelpad=-3)
        ax.set_zlabel("z (m)", labelpad=-1)
        ax.set_title(title + "\n" + subtitle, loc="left", fontsize=11, pad=-8)
        ax.tick_params(labelsize=8, pad=-1)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1, 1, 1, 0))
            axis._axinfo["grid"]["color"] = (.65, .7, .75, .25)
        manifest["tiles"][key] = {"vertices": len(mesh.vertices), "triangles": len(mesh.faces),
                                  "spawn_origin_before_recentering_m": origin.tolist(),
                                  "parameters": vars(cfg)}
    fig.subplots_adjust(left=.035, right=.98, bottom=.075, top=.87, wspace=.08, hspace=.15)
    fig.text(.045, .025, "Red marker: spawn platform and outward travel  |  CPU mesh preview; no Isaac Sim rollout or trained policy shown.", fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2)+"\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaaclab-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/terrain_preview.png")
    parser.add_argument("--difficulty", type=float, default=1.0)
    args = parser.parse_args()
    if not 0 <= args.difficulty <= 1:
        parser.error("difficulty must be in [0,1]")
    preview(args.isaaclab_root.resolve(), args.output, args.difficulty)
