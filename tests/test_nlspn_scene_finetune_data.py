import csv
from copy import deepcopy

import pytest

from scripts import nlspn_scene_finetune_data as data


@pytest.fixture(scope="module")
def fake_data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("scene_depth_data")
    for scenes in data.SPLIT_SCENES.values():
        for scene in scenes:
            (root / scene / "rgb").mkdir(parents=True)
            (root / scene / "depth").mkdir()
            for frame_id in range(1, data.SCENE_FRAME_COUNTS[scene] + 1):
                (root / scene / "rgb" / f"{frame_id:04d}.jpg").touch()
                (root / scene / "depth" / f"Image{frame_id:04d}.exr").touch()
    return root


@pytest.fixture(scope="module")
def manifests(fake_data_root):
    return data.build_manifests(fake_data_root)


def test_manifest_contract_constants_are_exact():
    assert data.SPLIT_SCENES == {
        "train": (
            "BeachApartmentInterior_My_ir",
            "bedroom_ir",
            "livingroom_ir",
            "room4",
        ),
        "val": ("room6",),
        "test": ("room3", "room7"),
    }
    assert data.SCENE_FRAME_COUNTS == {
        "BeachApartmentInterior_My_ir": 2000,
        "bedroom_ir": 2000,
        "livingroom_ir": 4000,
        "room3": 4000,
        "room4": 2000,
        "room6": 2000,
        "room7": 4000,
    }
    assert data.MANIFEST_FIELDS == (
        "split", "scene", "frame_id", "rgb_path", "depth_path")


def test_build_manifests_has_exact_scene_disjoint_geometry(manifests):
    assert tuple(manifests) == ("train", "val", "test")
    assert [len(manifests[name]) for name in manifests] == [10000, 2000, 8000]
    assert {row["scene"] for row in manifests["train"]} == {
        "BeachApartmentInterior_My_ir", "bedroom_ir", "livingroom_ir", "room4"}
    assert {row["scene"] for row in manifests["val"]} == {"room6"}
    assert {row["scene"] for row in manifests["test"]} == {"room3", "room7"}
    assert manifests["train"][0] == data.canonical_row(
        "train", "BeachApartmentInterior_My_ir", 1)
    assert manifests["test"][-1] == data.canonical_row("test", "room7", 4000)


def test_build_manifests_rejects_missing_pair(fake_data_root):
    path = fake_data_root / "room6/depth/Image0007.exr"
    path.unlink()
    try:
        with pytest.raises(ValueError, match=r"room6.*0007.*pair"):
            data.build_manifests(fake_data_root)
    finally:
        path.touch()


def test_build_manifests_rejects_extra_frame_id(fake_data_root):
    path = fake_data_root / "room6/rgb/2001.jpg"
    path.touch()
    try:
        with pytest.raises(ValueError, match=r"room6.*extra.*2001"):
            data.build_manifests(fake_data_root)
    finally:
        path.unlink()


def test_validate_manifest_rejects_duplicate_identity(manifests):
    rows = list(manifests["val"])
    rows.append(dict(rows[-1]))
    with pytest.raises(ValueError, match="duplicate.*identity"):
        data.validate_manifest(rows, "val")


def test_validate_split_isolation_rejects_scene_overlap(manifests):
    changed = deepcopy(manifests)
    changed["val"][0] = data.canonical_row(
        "val", "BeachApartmentInterior_My_ir", 1)
    with pytest.raises(ValueError, match="split.*overlap"):
        data.validate_split_isolation(changed)


def test_validate_manifest_rejects_path_traversal(manifests):
    rows = deepcopy(manifests["val"])
    rows[0]["rgb_path"] = "../room6/rgb/0001.jpg"
    with pytest.raises(ValueError, match="canonical.*path"):
        data.validate_manifest(rows, "val")


def test_validate_manifest_rejects_noncanonical_order(manifests):
    rows = deepcopy(manifests["val"])
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError, match="canonical.*order"):
        data.validate_manifest(rows, "val")


def test_validate_manifest_rejects_wrong_row_count(manifests):
    with pytest.raises(ValueError, match="row count"):
        data.validate_manifest(manifests["val"][:-1], "val")


def test_csv_round_trip_is_atomic_and_reconstructs_canonical_rows(
        manifests, fake_data_root, tmp_path):
    paths = data.write_manifests(manifests, tmp_path)
    assert set(paths) == {"train", "val", "test"}
    assert not list(tmp_path.glob("*.tmp"))
    loaded = data.read_manifests(tmp_path, fake_data_root)
    assert loaded == manifests


def test_read_manifest_rejects_changed_csv_fields(
        manifests, fake_data_root, tmp_path):
    paths = data.write_manifests(manifests, tmp_path)
    rows = list(csv.reader(paths["val"].open(newline="")))
    rows[0][-1] = "changed_depth_path"
    with paths["val"].open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    with pytest.raises(ValueError, match="CSV fields"):
        data.read_manifests(tmp_path, fake_data_root)


def test_read_manifest_rejects_noncanonical_csv_path(
        manifests, fake_data_root, tmp_path):
    paths = data.write_manifests(manifests, tmp_path)
    rows = list(csv.reader(paths["val"].open(newline="")))
    rows[1][3] = "../room6/rgb/0001.jpg"
    with paths["val"].open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    with pytest.raises(ValueError, match="canonical.*path"):
        data.read_manifests(tmp_path, fake_data_root)


def test_validate_canonical_digest_rejects_changed_digest(manifests):
    digest = data.canonical_manifest_sha256(manifests)
    assert len(digest) == 64
    assert digest == data.canonical_manifest_sha256(deepcopy(manifests))
    with pytest.raises(ValueError, match="digest"):
        data.validate_canonical_digest(manifests, "0" * 64)
