from pathlib import Path

import pytest
from m1_unilab.config import configuration
from unilab.training.run import resolve_configured_checkpoint_path


def test_mixed_resume_uses_the_field_consumed_by_unilab_ppo(tmp_path):
    checkpoint = tmp_path / "model_999.pt"
    checkpoint.touch()
    cfg = configuration("mixed", checkpoint=checkpoint)
    resolved, _ = resolve_configured_checkpoint_path(cfg, root_dir=Path.cwd())
    assert cfg.algo.load_run != "-1"
    assert resolved == checkpoint


def test_missing_checkpoint_fails_instead_of_silently_training_from_scratch(tmp_path):
    with pytest.raises(FileNotFoundError):
        configuration("mixed", checkpoint=tmp_path / "missing.pt")
