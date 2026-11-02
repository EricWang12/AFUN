import json
import pickle

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.ragnet import RAGNetDataset
from src.data.robot import RobotDataset, _match_segments_to_trajectories, _pixel_to_3d


def test_ragnet_public_schemas_and_class_zero(tmp_path):
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(tmp_path / "rgb.png")
    raw = np.ones((3, 4), dtype=np.uint8)
    raw[0, 0] = 0
    Image.fromarray(raw).save(tmp_path / "mask.png")
    record = {"frame_path": "./data/rgb.png", "mask_path": "./data/mask.png",
              "task_object_class": "handle", "graspnet_object_id": 0}
    with (tmp_path / "graspnet_train.pkl").open("wb") as handle:
        pickle.dump([record], handle)
    # Records that only name the object class are not training samples.
    with pytest.raises(ValueError, match="No RAGNet samples"):
        RAGNetDataset(tmp_path, source="graspnet")
    record.update(question="Open it.", answer="Use the handle.")
    with (tmp_path / "graspnet_easy_reasoning_train.pkl").open("wb") as handle:
        pickle.dump([record], handle)
    data = RAGNetDataset(tmp_path, source="graspnet", reasoning=["easy"])
    assert len(data) == 1
    sample = data[0]
    assert sample["query"] == ["Open it. Use the handle."]
    assert sample["mask"].sum() == 1
    assert sample["mask"][0, 0, 0] == 1
    (tmp_path / "rgb.png").unlink()
    with pytest.raises(FileNotFoundError):
        data[0]


def test_motion_matching_uses_metric_depth():
    k = {"fx": 100., "fy": 100., "cx": 2., "cy": 2.}
    depth_m = np.full((5, 5), 1000., dtype=np.float32) * 0.001
    np.testing.assert_allclose(_pixel_to_3d(depth_m, [3, 2], k), [0.01, 0, 1])
    left, right = torch.zeros(5, 5, dtype=torch.bool), torch.zeros(5, 5, dtype=torch.bool)
    left[1:4, :2], right[1:4, 3:] = True, True
    trajectories = [{"motion_2d": {"start_pixel": [4, 2]}}, {"motion_2d": {"start_pixel": [0, 2]}}]
    order, error = _match_segments_to_trajectories([left, right], trajectories, depth_m, k, 5, 5)
    assert order == [1, 0]
    assert error is None


def test_robot_skips_unmatched_samples_in_strict_mode(tmp_path):
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:20, 4:20] = True

    def sample(name, trajectories):
        folder = tmp_path / "source" / name
        folder.mkdir(parents=True)
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(folder / "obs_frame.png")
        Image.fromarray(mask.astype(np.uint8) * 255).save(folder / "sam_mask.png")
        np.save(folder / "obs_frame_depth.npy", np.full((32, 32), 1000, dtype=np.float32))
        trajectory = {"spline_params": {"ctrl": [[0, 0, 0.1]] * 3}, "motion_2d": {"start_pixel": [8, 8]}}
        (folder / "trajectory.json").write_text(json.dumps({
            "camera_info": {"intrinsics": {"fx": 30, "fy": 30, "cx": 16, "cy": 16}},
            "trajectories": [trajectory] * trajectories}))
        return {"path": f"source/{name}", "language": "Open it."}

    # Two trajectories but one mask component: the sample cannot be paired and is skipped.
    samples = [sample("unmatched", 2), sample("valid", 1)]
    (tmp_path / "manifest.json").write_text(json.dumps({"samples": samples}))
    data = RobotDataset(tmp_path, dataset_cfg={"min_area": 1, "curve_T": 4}, strict=True)
    assert data[0]["data_path"].endswith("valid")
