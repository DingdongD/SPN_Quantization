from pathlib import Path

import pytest

from scripts import nlspn_validated_output_promotion as promotion


def _make_tree(path: Path, marker):
    path.mkdir(parents=True)
    (path / "marker").write_text(marker, encoding="utf-8")
    return path


def _paths(tmp_path):
    target = _make_tree(tmp_path / "cross_scene_motion_windows", "old")
    staging = _make_tree(
        tmp_path / "cross_scene_motion_windows_raft_staging", "new")
    backup = tmp_path / "cross_scene_motion_windows_pre_raft_backup"
    return staging, target, backup


def _marker_validator(path):
    value = (Path(path) / "marker").read_text(encoding="utf-8")
    if value not in ("old", "new"):
        raise RuntimeError("invalid marker")
    return value


def test_promote_keeps_validated_new_target_and_old_backup(tmp_path):
    staging, target, backup = _paths(tmp_path)

    result = promotion.promote_validated_output(
        staging, target, backup, _marker_validator)

    assert result["target"] == str(target.resolve())
    assert result["backup"] == str(backup.resolve())
    assert (target / "marker").read_text(encoding="utf-8") == "new"
    assert (backup / "marker").read_text(encoding="utf-8") == "old"
    assert not staging.exists()


def test_failed_post_promotion_validation_restores_old_target(tmp_path):
    staging, target, backup = _paths(tmp_path)
    calls = []

    def validator(path):
        calls.append(Path(path))
        if len(calls) == 2:
            raise RuntimeError("post validation failed")

    with pytest.raises(RuntimeError, match="post validation"):
        promotion.promote_validated_output(
            staging, target, backup, validator)

    assert (target / "marker").read_text(encoding="utf-8") == "old"
    assert (staging / "marker").read_text(encoding="utf-8") == "new"
    assert not backup.exists()


def test_prevalidation_failure_leaves_both_trees_untouched(tmp_path):
    staging, target, backup = _paths(tmp_path)

    with pytest.raises(RuntimeError, match="pre validation"):
        promotion.promote_validated_output(
            staging, target, backup,
            lambda path: (_ for _ in ()).throw(
                RuntimeError("pre validation failed")))

    assert (target / "marker").read_text(encoding="utf-8") == "old"
    assert (staging / "marker").read_text(encoding="utf-8") == "new"
    assert not backup.exists()


def test_existing_backup_stops_before_validation_or_moves(tmp_path):
    staging, target, backup = _paths(tmp_path)
    _make_tree(backup, "backup")
    calls = []

    with pytest.raises(FileExistsError, match="backup"):
        promotion.promote_validated_output(
            staging, target, backup, lambda path: calls.append(path))

    assert calls == []
    assert (target / "marker").read_text(encoding="utf-8") == "old"
    assert (staging / "marker").read_text(encoding="utf-8") == "new"


def test_promotion_requires_distinct_sibling_directories(tmp_path):
    staging, target, backup = _paths(tmp_path / "one")
    elsewhere = tmp_path / "two" / backup.name

    with pytest.raises(ValueError, match="siblings"):
        promotion.promote_validated_output(
            staging, target, elsewhere, _marker_validator)
    with pytest.raises(ValueError, match="distinct"):
        promotion.promote_validated_output(
            staging, target, target, _marker_validator)
