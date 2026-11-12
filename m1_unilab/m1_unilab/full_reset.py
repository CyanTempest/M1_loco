"""Prevent terrain penetration when spawning M1 at arbitrary root orientations.

This reset-only event preserves the sampled pose except for the minimum
upward correction required by conservative collision-geometry bounds. The
current task resets joints to their declared defaults and does not randomize
geometry; both are prerequisites for caching the root-relative envelope.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from unilab.base.reset_selectors import resolve_reset_env_ids
from unilab.managers.manager_base import ManagerTermBase
from unilab.utils.math import np_matrix_from_quat


def _rotation(quaternions):
    """wxyz rotation matrices; normalize for calculation without changing poses."""
    q = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(q).all() or (norms <= 0).any():
        raise ValueError("TerrainClearance requires finite nonzero root quaternions")
    return np_matrix_from_quat(q / norms)


@dataclass(frozen=True)
class _Heightfield:
    heights: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    half_size: np.ndarray

    def max_under_spheres(self, centers, radii):
        # Supported terrain rotations preserve vertical, so raising world Z
        # also raises terrain-local Z. Sphere XY footprints remain circles.
        local = (centers - self.position) @ self.rotation
        ny, nx = self.heights.shape
        dx, dy = 2 * self.half_size / np.array([nx - 1, ny - 1])
        lower = (local[:, :2] - radii[:, None] + self.half_size) / [dx, dy]
        upper = (local[:, :2] + radii[:, None] + self.half_size) / [dx, dy]
        lo = np.floor(lower).astype(np.int64)
        hi = np.ceil(upper).astype(np.int64)
        result = np.full(len(centers), -np.inf)
        for index, (left, right) in enumerate(zip(lo, hi, strict=True)):
            x0, y0 = np.maximum(left, 0)
            x1, y1 = np.minimum(right, [nx - 1, ny - 1])
            if x0 <= x1 and y0 <= y1:
                # Include every vertex of every cell intersecting the square
                # footprint. Triangle heights cannot exceed their vertex max.
                result[index] = self.heights[y0 : y1 + 1, x0 : x1 + 1].max()
        return result


class TerrainClearance(ManagerTermBase):
    """Place after the root reset event, before drive randomization.

    Only public Entity transaction reads/writes and a detached copy of the
    public playback model are used. No XML loads, native forwards or RNG draws
    occur during reset. The playback model is necessary because the public
    scene-file path can precede generated-terrain injection.
    """

    def __init__(self, cfg, env):
        super().__init__(env)
        if cfg.mode != "reset" or set(cfg.params) - {"clearance"}:
            raise ValueError(
                "TerrainClearance requires mode='reset' and optional clearance only"
            )
        self.clearance = self._validate_clearance(cfg.params.get("clearance", 0.005))
        self.entity = env.scene["robot"]
        if not callable(getattr(self.entity, "read_reset_root_pose", None)):
            raise NotImplementedError(
                "TerrainClearance requires public Entity.read_reset_root_pose"
            )
        model = copy.copy(env.backend.get_playback_model())
        if not isinstance(model, mujoco.MjModel):
            raise NotImplementedError(
                "TerrainClearance requires a MuJoCo playback model"
            )
        data = mujoco.MjData(model)
        keyframe = env.cfg.scene.default_keyframe_name
        if keyframe is not None:
            key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
            if key < 0:
                raise ValueError(
                    f"TerrainClearance cannot find default keyframe {keyframe!r}"
                )
            mujoco.mj_resetDataKeyframe(model, data, key)
        # Use public per-entity joint defaults, in their declared name order.
        defaults = self.entity.data.default_joint_pos(torch.device(env.device))
        q0 = defaults.detach().cpu().numpy()
        if not np.allclose(q0, q0[0], rtol=0, atol=0):
            raise NotImplementedError(
                "TerrainClearance needs identical default joint poses across environments"
            )
        for name, value in zip(self.entity.joint_names, q0[0], strict=True):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0:
                raise ValueError(f"TerrainClearance missing playback joint {name}")
            data.qpos[model.jnt_qposadr[joint]] = value
        mujoco.mj_forward(model, data)
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "BASE_LINK")
        if root < 1:
            raise ValueError("TerrainClearance cannot find floating BASE_LINK")
        descendants = np.zeros(model.nbody, dtype=bool)
        descendants[root] = True
        for body in range(root + 1, model.nbody):
            descendants[body] = descendants[model.body_parentid[body]]
        collidable = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        robot_ids = np.flatnonzero(descendants[model.geom_bodyid] & collidable)
        if not len(robot_ids):
            raise ValueError("TerrainClearance found no robot collision geometry")
        root_rotation = data.xmat[root].reshape(3, 3)
        self.centers = (
            (data.geom_xpos[robot_ids] - data.xpos[root]) @ root_rotation
        ).copy()
        self.radii = model.geom_rbound[robot_ids].copy()
        self.num_geoms = len(robot_ids)
        if not np.isfinite(self.radii).all() or (self.radii <= 0).any():
            raise ValueError(
                "TerrainClearance requires finite positive collision bounds"
            )
        self.heightfields = []
        self.plane_heights = []
        for geom in np.flatnonzero(~descendants[model.geom_bodyid] & collidable):
            interacts = (
                (model.geom_contype[robot_ids] & model.geom_conaffinity[geom])
                | (model.geom_contype[geom] & model.geom_conaffinity[robot_ids])
            ).any()
            if not interacts:
                continue
            rotation = data.geom_xmat[geom].reshape(3, 3).copy()
            if not np.allclose(rotation[:, 2], [0, 0, 1], rtol=0, atol=1e-10):
                raise NotImplementedError(
                    "TerrainClearance requires upright static terrain"
                )
            position = data.geom_xpos[geom].copy()
            kind = model.geom_type[geom]
            if kind == mujoco.mjtGeom.mjGEOM_PLANE:
                self.plane_heights.append(float(position[2]))
            elif kind == mujoco.mjtGeom.mjGEOM_HFIELD:
                field = model.geom_dataid[geom]
                ny, nx = int(model.hfield_nrow[field]), int(model.hfield_ncol[field])
                address = int(model.hfield_adr[field])
                size = model.hfield_size[field]
                heights = (
                    model.hfield_data[address : address + nx * ny]
                    .astype(np.float64)
                    .reshape(ny, nx)
                )
                heights = position[2] + size[2] * heights
                self.heightfields.append(
                    _Heightfield(heights, position, rotation, size[:2].copy())
                )
            else:
                raise NotImplementedError(
                    "TerrainClearance supports only plane/heightfield ground"
                )
        if (
            env.cfg.scene.terrain is not None
            and env.cfg.scene.terrain.generator is not None
            and not self.heightfields
        ):
            raise ValueError(
                "TerrainClearance playback model omitted the generated heightfield"
            )
        if not self.heightfields and not self.plane_heights:
            raise ValueError("TerrainClearance found no ground collision surface")
        self.last_lift = np.zeros(env.num_envs, dtype=np.float64)

    @staticmethod
    def _validate_clearance(value):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("clearance must be finite and non-negative")
        return value

    def __call__(self, env, env_ids, clearance=0.005):
        clearance = self._validate_clearance(clearance)
        ids = resolve_reset_env_ids(
            env_ids, num_envs=env.num_envs, device=env.device, owner="TerrainClearance"
        )
        if not ids.numel():
            return
        pose = np.asarray(
            self.entity.read_reset_root_pose(ids), dtype=np.float64
        ).copy()
        if pose.shape != (ids.numel(), 7) or not np.isfinite(pose).all():
            raise ValueError("TerrainClearance received an invalid pending root pose")
        centers = np.einsum("nij,gj->ngi", _rotation(pose[:, 3:7]), self.centers)
        centers += pose[:, None, :3]
        radii = np.broadcast_to(self.radii, centers.shape[:2]).reshape(-1)
        flat_centers = centers.reshape(-1, 3)
        heights = np.full(len(flat_centers), max(self.plane_heights, default=-np.inf))
        for surface in self.heightfields:
            heights = np.maximum(
                heights, surface.max_under_spheres(flat_centers, radii)
            )
        lift = np.maximum(
            0,
            (heights + radii + clearance - flat_centers[:, 2])
            .reshape(len(pose), -1)
            .max(axis=1),
        )
        pose[:, 2] += lift
        self.last_lift[ids.detach().cpu().numpy()] = lift
        if (lift > 0).any():
            self.entity.write_root_link_pose_to_sim(
                torch.as_tensor(pose, dtype=torch.float64, device=env.device), ids
            )
