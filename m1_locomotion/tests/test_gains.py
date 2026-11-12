import json
import pytest
import m1_locomotion.gains as module
from m1_locomotion.contract import JOINTS, LEG_JOINTS, ROOT, WHEEL_JOINTS


def test_a2_leg_gain_mapping():
    gains = module.load_gains("preview")
    assert set(gains) == set(JOINTS)
    for name in LEG_JOINTS:
        assert gains[name] == {"kp": 150.0, "kd": 6.0} if "KNEE" in name else gains[name] == {"kp": 100.0, "kd": 4.0}


def test_user_wheel_values_never_silently_fall_back(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config/a2_leg_gains.json").write_bytes((ROOT / "config/a2_leg_gains.json").read_bytes())
    data = {"source": "test", "motors": {n: {"kp": None, "kd": None} for n in WHEEL_JOINTS}}
    target = tmp_path / "config/wheel_gains.json"
    target.write_text(json.dumps(data))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="Missing/invalid"):
        module.load_gains("user_wheels")
    for i, n in enumerate(WHEEL_JOINTS):
        data["motors"][n] = {"kp": 0.0, "kd": 0.2 + i * .1}
    target.write_text(json.dumps(data))
    gains = module.load_gains("user_wheels")
    assert [gains[n]["kd"] for n in WHEEL_JOINTS] == [0.2 + i*.1 for i in range(4)]
    data["motors"][WHEEL_JOINTS[0]] = {"kp": 100.0, "kd": 4.0}
    target.write_text(json.dumps(data))
    assert module.load_gains("user_wheels")[WHEEL_JOINTS[0]] == {"kp": 100.0, "kd": 4.0}
    data["motors"][WHEEL_JOINTS[0]]["kd"] = -1.0
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Missing/invalid"):
        module.load_gains("user_wheels")
