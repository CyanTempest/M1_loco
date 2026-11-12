"""M1 domain randomization through public UniLab event/Entity contracts.

MuJoCo has one sliding-friction coefficient, not PhysX's independent static
and dynamic coefficients. MaterialBuckets deliberately samples that single
coefficient in [0.3, 1.0]. Its restitution-to-damping-ratio conversion is a
continuous second-order contact approximation, not an exact MuJoCo restitution
coefficient or a claim of PhysX contact equivalence.
"""

from __future__ import annotations

import math

import torch
from unilab.managers.manager_base import ManagerTermBase
from unilab.managers.scene_entity_config import SceneEntityCfg

MATERIAL_MAPPING = (
    "MuJoCo single sliding friction U[0.3,1.0] replaces the unavailable "
    "independent PhysX static/dynamic pair; torsional=0.005, rolling=0.0001. "
    "Requested restitution e is mapped approximately to solref=[0.02/zeta,zeta], "
    "zeta=-ln(e)/sqrt(pi^2+ln(e)^2), with e=0 mapped to zeta=1. "
    "Keeping timeconst*zeta=0.02 prevents restitution DR from increasing contact stiffness. "
    "Actual restitution depends on timestep, solver, impedance and geometry."
)


def restitution_to_damping_ratio(restitution: torch.Tensor) -> torch.Tensor:
    """Return an approximate continuous-contact damping ratio for 0 <= e < 1."""
    if not restitution.is_floating_point():
        raise TypeError("restitution must be a floating-point tensor")
    if not bool(torch.isfinite(restitution).all()) or bool(
        ((restitution < 0) | (restitution >= 1)).any()
    ):
        raise ValueError("restitution must contain finite values in [0, 1)")
    # Keep the zero branch finite as well: where() evaluates both arguments.
    log_e = torch.log(restitution.clamp_min(torch.finfo(restitution.dtype).tiny))
    zeta = -log_e / torch.sqrt(math.pi**2 + log_e.square())
    return torch.where(restitution == 0, torch.ones_like(zeta), zeta)


def _bounds(value, name, *, minimum=None, maximum=None):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a (minimum, maximum) pair")
    low, high = map(float, value)
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"{name} must be finite and ordered")
    if minimum is not None and low < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and high > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return low, high


def _asset(cfg, env, *, mode, allowed):
    if cfg.mode != mode:
        raise ValueError(f"{cfg.func} requires mode={mode!r}")
    unknown = set(cfg.params) - set(allowed)
    if unknown:
        raise ValueError(f"unsupported event parameters: {sorted(unknown)}")
    asset_cfg = cfg.params.get("asset_cfg")
    if not isinstance(asset_cfg, SceneEntityCfg):
        raise TypeError("asset_cfg must be a resolved SceneEntityCfg")
    return asset_cfg, env.scene[asset_cfg.name]


def _all_rows(env, env_ids):
    expected = torch.arange(env.num_envs, dtype=torch.int64, device=env.device)
    if env_ids is None:
        return expected
    if not isinstance(env_ids, torch.Tensor) or env_ids.dtype != torch.int64:
        raise TypeError("startup env_ids must be an int64 tensor")
    if env_ids.device != expected.device or not torch.equal(env_ids, expected):
        raise ValueError("startup randomization requires all environment rows in order")
    return env_ids


def _default_rows(defaults, num_envs, canonical_ndim, device):
    # Public bindings can return read-only NumPy or Torch defaults; always copy.
    values = (
        defaults.detach().to(device=device, dtype=torch.float64).clone()
        if isinstance(defaults, torch.Tensor)
        else torch.tensor(defaults, device=device, dtype=torch.float64)
    )
    if values.ndim == canonical_ndim:
        return values.unsqueeze(0).expand(num_envs, *values.shape).clone()
    if values.ndim == canonical_ndim + 1 and values.shape[0] == num_envs:
        return values
    raise ValueError("default binding has an invalid canonical/per-environment shape")


class MaterialBuckets(ManagerTermBase):
    """Sample one fixed material bucket per selected geom and environment.

    The 64 bucket values are shared across environments; assignments are
    independent per env/geom. Writes occur in the environment-owned startup
    transaction. Reset events never resample or mutate this table.
    """

    uses_startup_rows = True

    def __init__(self, cfg, env):
        super().__init__(env)
        asset_cfg, self.entity = _asset(
            cfg,
            env,
            mode="startup",
            allowed=(
                "asset_cfg",
                "num_buckets",
                "friction_range",
                "restitution_range",
                "time_constant",
            ),
        )
        self.num_buckets = cfg.params.get("num_buckets", 64)
        if isinstance(self.num_buckets, bool) or not isinstance(self.num_buckets, int):
            raise TypeError("num_buckets must be an integer")
        if self.num_buckets <= 0:
            raise ValueError("num_buckets must be positive")
        self.friction_range = _bounds(
            cfg.params.get("friction_range", (0.3, 1.0)), "friction_range", minimum=0
        )
        self.restitution_range = _bounds(
            cfg.params.get("restitution_range", (0.0, 0.5)),
            "restitution_range",
            minimum=0,
            maximum=1,
        )
        if self.restitution_range[1] >= 1:
            raise ValueError("restitution_range maximum must be less than 1")
        self.time_constant = float(cfg.params.get("time_constant", 0.02))
        if not math.isfinite(self.time_constant) or self.time_constant <= 0:
            raise ValueError("time_constant must be positive and finite")
        self.geom_ids, defaults = self.entity.bind_geom_friction_write(
            asset_cfg.geom_ids, term_name="m1_material_buckets"
        )
        solref_ids, _ = self.entity.bind_geom_solref_write(
            asset_cfg.geom_ids, term_name="m1_material_buckets"
        )
        if self.geom_ids.tolist() != solref_ids.tolist():
            raise ValueError("friction and solref bindings selected different geoms")
        self.num_geoms = len(self.geom_ids)
        if self.num_geoms == 0:
            raise ValueError("MaterialBuckets must select at least one geom")
        self.friction_values = _default_rows(defaults, env.num_envs, 2, env.rng.device)
        if self.friction_values.shape != (env.num_envs, self.num_geoms, 3):
            raise ValueError(
                "friction defaults must have three values per selected geom"
            )
        self.bucket_values = None  # [num_buckets, sliding friction / requested e]
        self.bucket_ids = None  # [num_envs, selected_geoms]
        self.solref_values = None

    def __call__(
        self,
        env,
        env_ids,
        asset_cfg,
        num_buckets=64,
        friction_range=(0.3, 1.0),
        restitution_range=(0.0, 0.5),
        time_constant=0.02,
    ):
        del asset_cfg, num_buckets, friction_range, restitution_range, time_constant
        ids = _all_rows(env, env_ids)
        if self.bucket_values is None:
            friction = env.rng.uniform(
                *self.friction_range, (self.num_buckets,), dtype=torch.float64
            )
            restitution = env.rng.uniform(
                *self.restitution_range, (self.num_buckets,), dtype=torch.float64
            )
            self.bucket_values = torch.stack((friction, restitution), dim=-1)
            self.bucket_ids = env.rng.integers(
                0, self.num_buckets, (env.num_envs, self.num_geoms)
            )
            selected = self.bucket_values[self.bucket_ids]
            self.friction_values[..., 0] = selected[..., 0]
            self.friction_values[..., 1] = 0.005
            self.friction_values[..., 2] = 0.0001
            zeta = restitution_to_damping_ratio(selected[..., 1])
            # MuJoCo k is proportional to 1 / (timeconst * zeta)^2.
            # Holding timeconst alone fixed unintentionally stiffens bouncy
            # contacts (up to ~21.5x at e=.5), which triggered the captured NaN.
            # Interpret the configured time_constant as the zeta=1 reference.
            self.solref_values = torch.stack(
                (
                    self.time_constant / zeta,
                    zeta,
                ),
                dim=-1,
            )
        self.entity.write_geom_friction_to_sim(
            self.friction_values, self.geom_ids, ids, term_name="m1_material_buckets"
        )
        self.entity.write_geom_solref_to_sim(
            self.solref_values, self.geom_ids, ids, term_name="m1_material_buckets"
        )


class BodyMassInertia(ManagerTermBase):
    """Fixed per-env M1 mass DR with matching principal-inertia scaling.

    BASE_LINK gets an additive mass offset; other selected bodies get an
    independent scale. Inertia scales by final_mass / original_mass, retaining
    the body's source geometry, COM and principal axes.
    """

    uses_startup_rows = True

    def __init__(self, cfg, env):
        super().__init__(env)
        asset_cfg, self.entity = _asset(
            cfg,
            env,
            mode="startup",
            allowed=(
                "asset_cfg",
                "base_body_name",
                "base_mass_range",
                "other_mass_scale_range",
            ),
        )
        self.base_body_name = cfg.params.get("base_body_name", "BASE_LINK")
        self.base_mass_range = _bounds(
            cfg.params.get("base_mass_range", (-1.0, 3.0)), "base_mass_range"
        )
        self.other_mass_scale_range = _bounds(
            cfg.params.get("other_mass_scale_range", (0.7, 1.3)),
            "other_mass_scale_range",
            minimum=0,
        )
        if self.other_mass_scale_range[0] <= 0:
            raise ValueError("other_mass_scale_range must be strictly positive")
        self.body_ids, mass0 = self.entity.bind_body_mass_write(
            asset_cfg.body_ids, term_name="m1_mass_inertia"
        )
        inertia_ids, inertia0 = self.entity.bind_body_inertia_write(
            asset_cfg.body_ids, term_name="m1_mass_inertia"
        )
        if self.body_ids.tolist() != inertia_ids.tolist():
            raise ValueError("mass and inertia bindings selected different bodies")
        names = [self.entity.body_names[int(i)] for i in self.body_ids]
        if names.count(self.base_body_name) != 1:
            raise ValueError("BodyMassInertia must select the base body exactly once")
        self.base_column = names.index(self.base_body_name)
        self.mass0 = _default_rows(mass0, env.num_envs, 1, env.rng.device)
        self.inertia0 = _default_rows(inertia0, env.num_envs, 2, env.rng.device)
        if self.mass0.shape != (env.num_envs, len(names)) or self.inertia0.shape != (
            env.num_envs,
            len(names),
            3,
        ):
            raise ValueError("mass/inertia defaults have incompatible shapes")
        if not bool(torch.isfinite(self.mass0).all()) or bool((self.mass0 <= 0).any()):
            raise ValueError("selected source body masses must be finite and positive")
        if not bool(torch.isfinite(self.inertia0).all()) or bool(
            (self.inertia0 < 0).any()
        ):
            raise ValueError("selected source inertias must be finite and nonnegative")
        if bool((self.mass0[:, self.base_column] + self.base_mass_range[0] <= 0).any()):
            raise ValueError("base mass range could produce non-positive mass")
        self.mass_values = None
        self.inertia_values = None

    def __call__(
        self,
        env,
        env_ids,
        asset_cfg,
        base_body_name="BASE_LINK",
        base_mass_range=(-1.0, 3.0),
        other_mass_scale_range=(0.7, 1.3),
    ):
        del asset_cfg, base_body_name, base_mass_range, other_mass_scale_range
        ids = _all_rows(env, env_ids)
        if self.mass_values is None:
            scale = env.rng.uniform(
                *self.other_mass_scale_range, self.mass0.shape, dtype=torch.float64
            )
            mass = self.mass0 * scale
            mass[:, self.base_column] = self.mass0[
                :, self.base_column
            ] + env.rng.uniform(
                *self.base_mass_range, (env.num_envs,), dtype=torch.float64
            )
            self.mass_values = mass
            self.inertia_values = self.inertia0 * (mass / self.mass0)[..., None]
        self.entity.write_body_mass_to_sim(
            self.mass_values, self.body_ids, ids, term_name="m1_mass_inertia"
        )
        self.entity.write_body_inertia_to_sim(
            self.inertia_values, self.body_ids, ids, term_name="m1_mass_inertia"
        )


class RandomizeDrive(ManagerTermBase):
    """Make reset events the sole owner of drive gain/delay/wrench sampling."""

    def __init__(self, cfg, env):
        super().__init__(env)
        if cfg.mode != "reset" or cfg.params:
            raise ValueError("RandomizeDrive requires mode='reset' and no params")

    def __call__(self, env, env_ids):
        env.action_manager.get_term("drive").randomize(env_ids, env.rng)
