import importlib.util
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from m1_locomotion.audit import audit_model, forward_kinematics
from m1_locomotion.contract import (ACTOR_DIM, DEFAULT_POSITION, LEG_JOINTS, ROOT, WHEEL_JOINTS,
                                    WHEEL_LINKS, action_targets)


def test_physical_asset():
    result = audit_model()
    assert result["actuated_joints"] == 16
    assert .50 < result["standing_base_height_m"] < .55


def test_continuous_wheel_rotations_do_not_move_axles():
    model = ET.parse(ROOT / "assets/m1/m1_cad.urdf").getroot()
    nominal = forward_kinematics(model)
    q = dict(DEFAULT_POSITION)
    for i, joint in enumerate(WHEEL_JOINTS):
        q[joint] = math.pi * (i + 1.7)
    spun = forward_kinematics(model, q)
    for wheel in WHEEL_LINKS:
        assert np.allclose(nominal[wheel][:3, 3], spun[wheel][:3, 3])


def test_action_zero_and_unit_wheel_targets():
    q, dq = action_targets([0.0] * 16)
    assert q == [DEFAULT_POSITION[n] for n in LEG_JOINTS]
    assert dq == [0.0] * 4
    q, dq = action_targets([0.0] * 12 + [1.0] * 4)
    assert dq == [5.0] * 4
    assert ACTOR_DIM == 53


@pytest.mark.parametrize("action", [[float("nan")] * 16, [0.0] * 12, [float("inf")] * 16])
def test_invalid_actions_are_rejected(action):
    with pytest.raises(ValueError):
        action_targets(action)


def test_action_bounds():
    q, dq = action_targets([100.0] * 16)
    assert dq == [15.0] * 4
    assert abs(q[0] - .45) < 1e-9


def test_asset_preparation_is_reproducible():
    spec = importlib.util.spec_from_file_location("prepare_assets", ROOT / "scripts/prepare_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = (ROOT / "assets/m1/m1_cad.urdf").read_bytes()
    module.prepare()
    after = (ROOT / "assets/m1/m1_cad.urdf").read_bytes()
    assert before == after


def test_m1_inertia_is_preserved_exactly():
    original = ET.parse(ROOT / "assets/m1/original.urdf").getroot()
    prepared = ET.parse(ROOT / "assets/m1/m1_cad.urdf").getroot()
    for old, new in zip(original.findall("link"), prepared.findall("link")):
        assert old.get("name") == new.get("name")
        for tag in ("inertial/origin", "inertial/mass", "inertial/inertia"):
            assert old.find(tag).attrib == new.find(tag).attrib
