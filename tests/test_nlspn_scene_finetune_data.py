import csv
from copy import deepcopy

import numpy as np
import pytest
import torch

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


def test_preprocess_matches_inference_geometry_and_rgb_scale():
    rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    depth = np.full((480, 640), 2.5, dtype=np.float32)
    rgb_out, gt, valid = data.preprocess_arrays(rgb, depth)
    assert rgb_out.shape == (3, 228, 304)
    assert gt.shape == valid.shape == (228, 304)
    assert rgb_out.min() == pytest.approx(128.0 / 255.0)
    assert rgb_out.max() == pytest.approx(128.0 / 255.0)
    assert np.all(gt == 2.5) and valid.all()


def test_preprocess_does_not_apply_imagenet_normalization():
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.ones((480, 640), dtype=np.float32)
    rgb[:, :, 0] = 255
    rgb_out, _, _ = data.preprocess_arrays(rgb, depth)
    np.testing.assert_allclose(rgb_out[0], 1.0)
    np.testing.assert_allclose(rgb_out[1:], 0.0)


def test_depth_from_exr_array_accepts_equal_three_channels():
    single = np.arange(12, dtype=np.float32).reshape(3, 4)
    three = np.repeat(single[:, :, None], 3, axis=2)
    np.testing.assert_array_equal(data.depth_from_exr_array(three), single)


@pytest.mark.parametrize("array", [
    np.zeros((2, 3, 2), dtype=np.float32),
    np.zeros((2, 3, 3, 1), dtype=np.float32),
])
def test_depth_from_exr_array_rejects_malformed_shape(array):
    with pytest.raises(ValueError, match="EXR.*shape"):
        data.depth_from_exr_array(array)


def test_depth_from_exr_array_rejects_unequal_channels():
    array = np.zeros((2, 3, 3), dtype=np.float32)
    array[0, 0, 1] = 1.0
    with pytest.raises(ValueError, match="EXR.*channels"):
        data.depth_from_exr_array(array)


def test_sanitize_depth_rejects_nonfinite_nonpositive_and_over_ten_metres():
    depth = np.array(
        [[np.nan, np.inf, -1.0, 0.0, 0.1, 10.0, 10.1]],
        dtype=np.float32)
    gt, valid = data.sanitize_depth(depth)
    np.testing.assert_array_equal(
        valid, [[False, False, False, False, True, True, False]])
    np.testing.assert_allclose(gt, [[0.0, 0.0, 0.0, 0.0, 0.1, 10.0, 0.0]])


def test_validation_sparse_is_fixed_and_training_sparse_changes_by_epoch():
    valid = np.ones((228, 304), dtype=bool)
    gt = np.arange(valid.size, dtype=np.float32).reshape(valid.shape) + 1.0
    va = data.build_sparse(gt, valid, "val", "room6", 7, 0, 2026)
    vb = data.build_sparse(gt, valid, "val", "room6", 7, 9, 2026)
    ta = data.build_sparse(gt, valid, "train", "room4", 7, 0, 2026)
    tb = data.build_sparse(gt, valid, "train", "room4", 7, 1, 2026)
    assert np.array_equal(va, vb)
    assert not np.array_equal(ta, tb)
    assert all(np.count_nonzero(x) == 500 for x in (va, ta, tb))


def test_build_sparse_rejects_fewer_than_500_valid_pixels():
    gt = np.ones((20, 25), dtype=np.float32)
    valid = np.ones_like(gt, dtype=bool)
    valid[0, 0] = False
    with pytest.raises(ValueError, match="fewer than 500 valid pixels"):
        data.build_sparse(gt, valid, "val", "room6", 1, 0, 2026)


def test_train_augmentation_is_deterministic_and_joint_for_geometry():
    rgb = np.linspace(0.1, 0.9, 3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    gt = np.arange(20, dtype=np.float32).reshape(4, 5) + 1.0
    valid = np.ones((4, 5), dtype=bool)
    first = data.augment_arrays(rgb, gt, valid, "train", "room4", 7, 3, 2026)
    again = data.augment_arrays(rgb, gt, valid, "train", "room4", 7, 3, 2026)
    later = data.augment_arrays(rgb, gt, valid, "train", "room4", 7, 4, 2026)
    for left, right in zip(first, again):
        np.testing.assert_array_equal(left, right)
    assert not np.array_equal(first[0], later[0])
    assert (
        np.array_equal(first[1], gt) or
        np.array_equal(first[1], gt[:, ::-1]))
    assert np.array_equal(first[2], first[1] > 0)


def test_validation_augmentation_is_identity():
    rgb = np.full((3, 4, 5), 0.25, dtype=np.float32)
    gt = np.ones((4, 5), dtype=np.float32)
    valid = np.ones((4, 5), dtype=bool)
    actual = data.augment_arrays(
        rgb, gt, valid, "val", "room6", 1, 99, 2026)
    for source, result in zip((rgb, gt, valid), actual):
        np.testing.assert_array_equal(source, result)


def test_scene_depth_dataset_returns_tensors_and_identity(monkeypatch, tmp_path):
    row = data.canonical_row("val", "room6", 7)
    rgb_path = tmp_path / row["rgb_path"]
    depth_path = tmp_path / row["depth_path"]
    rgb_path.parent.mkdir(parents=True)
    depth_path.parent.mkdir(parents=True)
    rgb_path.touch()
    depth_path.touch()
    monkeypatch.setattr(
        data, "read_rgb_jpg",
        lambda path: np.full((480, 640, 3), 128, dtype=np.uint8))
    monkeypatch.setattr(
        data, "read_depth_exr",
        lambda path: np.full((480, 640), 2.5, dtype=np.float32))

    dataset = data.SceneDepthDataset([row], tmp_path, "val", seed=2026)
    dataset.set_epoch(12)
    sample = dataset[0]

    assert sample["scene"] == "room6"
    assert sample["frame_id"] == 7
    assert sample["rgb"].shape == (3, 228, 304)
    assert sample["dep"].shape == sample["gt"].shape == (1, 228, 304)
    assert sample["valid"].shape == (1, 228, 304)
    assert all(isinstance(sample[name], torch.Tensor)
               for name in ("rgb", "dep", "gt", "valid"))
    assert torch.count_nonzero(sample["dep"]).item() == 500
    assert torch.all(sample["dep"][sample["dep"] > 0] == 2.5)
