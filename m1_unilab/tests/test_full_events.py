"""Material semantics and selected-column DR using only a public Entity fake."""

from types import SimpleNamespace

import pytest
import torch
from m1_unilab.full_events import (
    BodyMassInertia,
    MaterialBuckets,
    RandomizeDrive,
    restitution_to_damping_ratio,
)
from unilab.managers.event_manager import EventTermCfg
from unilab.managers.rng import ManagerRng
from unilab.managers.scene_entity_config import SceneEntityCfg


class PublicEntity:
    body_names = ("BASE_LINK", "UNSELECTED", "FRONT", "REAR")

    def __init__(self):
        self.calls = []

    def bind_geom_friction_write(self, ids, *, term_name):
        assert ids == [2, 0]
        return torch.tensor(ids), torch.tensor([[0.8, 0.005, 0.0001]] * 2)

    def bind_geom_solref_write(self, ids, *, term_name):
        return torch.tensor(ids), torch.tensor([[0.02, 1.0]] * 2)

    def bind_body_mass_write(self, ids, *, term_name):
        assert ids == [2, 0, 3]
        return torch.tensor(ids), torch.tensor([3.0, 20.0, 5.0])

    def bind_body_inertia_write(self, ids, *, term_name):
        return torch.tensor(ids), torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [2.0, 3.0, 4.0]]
        )

    def _write(self, field, values, columns, env_ids):
        self.calls.append(
            (field, values.clone(), torch.as_tensor(columns).clone(), env_ids.clone())
        )

    def write_geom_friction_to_sim(self, values, columns, env_ids, *, term_name):
        self._write("friction", values, columns, env_ids)

    def write_geom_solref_to_sim(self, values, columns, env_ids, *, term_name):
        self._write("solref", values, columns, env_ids)

    def write_body_mass_to_sim(self, values, columns, env_ids, *, term_name):
        self._write("mass", values, columns, env_ids)

    def write_body_inertia_to_sim(self, values, columns, env_ids, *, term_name):
        self._write("inertia", values, columns, env_ids)


def setup_term(cls, *, count=137, **params):
    entity = PublicEntity()
    env = SimpleNamespace(
        num_envs=count,
        device=torch.device("cpu"),
        rng=ManagerRng.seeded(17, device="cpu"),
        scene={"robot": entity},
    )
    asset = SceneEntityCfg("robot", geom_ids=[2, 0], body_ids=[2, 0, 3])
    cfg = EventTermCfg(func=cls, mode="startup", params={"asset_cfg": asset, **params})
    return cls(cfg, env), env, entity, cfg.params


def test_restitution_mapping_zero_endpoint_formula_and_validation():
    values = torch.tensor([0.0, 0.1, 0.5], dtype=torch.float64)
    actual = restitution_to_damping_ratio(values)
    assert actual[0] == 1
    assert actual[2] == pytest.approx(0.2154537619662468)
    assert torch.all(actual[:-1] > actual[1:])
    for bad in (-0.01, 1.0, float("nan")):
        with pytest.raises(ValueError):
            restitution_to_damping_ratio(torch.tensor([bad]))


def test_material_buckets_match_shared_table_and_keep_nonselected_geoms_untouched():
    term, env, entity, params = setup_term(MaterialBuckets)
    term(env, torch.arange(env.num_envs), **params)
    assert term.bucket_values.shape == (64, 2)
    assert term.bucket_ids.shape == (env.num_envs, 2)
    assert term.bucket_ids.min() >= 0 and term.bucket_ids.max() < 64
    assert (term.bucket_ids[:, 0] != term.bucket_ids[:, 1]).any()
    selected = term.bucket_values[term.bucket_ids]
    friction, solref = entity.calls
    assert friction[0] == "friction" and solref[0] == "solref"
    for _, _, columns, rows in entity.calls:
        assert columns.tolist() == [
            2,
            0,
        ]  # entity-local order, not natural/global order
        assert rows.tolist() == list(range(env.num_envs))
    torch.testing.assert_close(friction[1][..., 0], selected[..., 0])
    assert ((selected[..., 0] >= 0.3) & (selected[..., 0] <= 1)).all()
    assert ((selected[..., 1] >= 0) & (selected[..., 1] <= 0.5)).all()
    torch.testing.assert_close(
        friction[1][..., 1], torch.full((env.num_envs, 2), 0.005, dtype=torch.float64)
    )
    torch.testing.assert_close(
        friction[1][..., 2], torch.full((env.num_envs, 2), 0.0001, dtype=torch.float64)
    )
    torch.testing.assert_close(
        solref[1][..., 0] * solref[1][..., 1],
        torch.full((env.num_envs, 2), 0.02, dtype=torch.float64),
    )
    torch.testing.assert_close(
        solref[1][..., 1], restitution_to_damping_ratio(selected[..., 1])
    )
    # A hypothetical full model retains the geometry column omitted by the selector.
    full = torch.full((env.num_envs, 3, 3), -99.0, dtype=torch.float64)
    full[:, friction[2]] = friction[1]
    assert (full[:, 1] == -99).all()


def test_startup_materials_replay_without_resampling_and_reject_partial_rows():
    term, env, entity, params = setup_term(MaterialBuckets, count=7)
    term(env, None, **params)
    rng_state = env.rng.generator.get_state().clone()
    term(env, torch.arange(7), **params)
    assert torch.equal(env.rng.generator.get_state(), rng_state)
    torch.testing.assert_close(entity.calls[0][1], entity.calls[2][1])
    with pytest.raises(ValueError, match="all environment rows"):
        term(env, torch.tensor([1, 4]), **params)


def test_body_mass_additive_base_and_scaled_other_inertia_follow_selected_order():
    term, env, entity, params = setup_term(BodyMassInertia)
    term(env, None, **params)
    mass, inertia = entity.calls
    assert mass[0] == "mass" and inertia[0] == "inertia"
    assert mass[2].tolist() == [2, 0, 3]
    assert (mass[1][:, 1] >= 19).all() and (mass[1][:, 1] <= 23).all()
    for column, baseline in ((0, 3.0), (2, 5.0)):
        ratio = mass[1][:, column] / baseline
        assert (ratio >= 0.7).all() and (ratio <= 1.3).all()
    expected_ratio = mass[1] / torch.tensor([3.0, 20.0, 5.0], dtype=torch.float64)
    expected_inertia = (
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [2.0, 3.0, 4.0]])
        * expected_ratio[..., None]
    )
    torch.testing.assert_close(inertia[1], expected_inertia)
    rng_state = env.rng.generator.get_state().clone()
    term(env, None, **params)
    torch.testing.assert_close(entity.calls[2][1], mass[1])
    assert torch.equal(env.rng.generator.get_state(), rng_state)


@pytest.mark.parametrize(
    "params",
    [
        {"num_buckets": 0},
        {"friction_range": [1, 0.3]},
        {"restitution_range": [0, 1]},
        {"time_constant": 0},
    ],
)
def test_invalid_material_ranges_fail_before_writes(params):
    with pytest.raises((ValueError, TypeError)):
        setup_term(MaterialBuckets, **params)


def test_nonpositive_mass_range_is_rejected_without_clamping():
    with pytest.raises(ValueError, match="non-positive mass"):
        setup_term(BodyMassInertia, base_mass_range=(-21, 3))


def test_randomize_drive_only_delegates_selected_rows_and_manager_rng():
    calls = []
    drive = SimpleNamespace(randomize=lambda ids, rng: calls.append((ids, rng)))

    def get_term(name):
        assert name == "drive"
        return drive

    env = SimpleNamespace(
        action_manager=SimpleNamespace(get_term=get_term), rng=object()
    )
    term = RandomizeDrive(EventTermCfg(func=RandomizeDrive, mode="reset"), env)
    ids = torch.tensor([2, 5])
    term(env, ids)
    assert calls == [(ids, env.rng)]
