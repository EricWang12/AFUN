import numpy as np
import pytest

from src.eval.__main__ import evaluate
from src.eval.metrics import anchor_point, mask_iou, motion_metrics, resample, summarize


def test_iou_aggregations_weight_pixels_differently():
    rows = [mask_iou(np.ones((1, 1)), np.ones((1, 1))),
            mask_iou(np.zeros((2, 2)), np.ones((2, 2)))]
    summary = summarize(rows, "segmentation")
    assert summary["giou"] == 50
    assert summary["ciou"] == 20
    assert mask_iou(np.zeros((2, 2)), np.zeros((2, 2)))["iou"] == 1


def test_samples_without_prediction_are_left_out_of_segmentation_scores():
    def score(sample):
        if sample["id"] == "missed":
            raise RuntimeError("no detection above the confidence threshold")
        return mask_iou(np.ones((2, 2)), np.ones((2, 2)))
    out = evaluate(score, [{"id": "hit", "query": "x"}, {"id": "missed", "query": "x"}], "segmentation")
    assert out["giou"] == out["ciou"] == 100
    assert out["num_samples"] == 2 and out["num_failures"] == 1


def test_absolute_relative_and_contact_metrics():
    target = np.array([[0., 0., 1.], [1., 0., 1.]])
    pred = target + np.array([0., 1., 0.])
    metric = motion_metrics(pred, target, {"fx": 1., "fy": 1., "cx": 0., "cy": 0.}, np.ones((3, 3)))
    assert metric["ade_abs"] == pytest.approx(1)
    assert metric["fde_abs"] == pytest.approx(1)
    assert metric["ade_rel"] == metric["fde_rel"] == 0
    assert metric["contact_in_mask"] == 1


def test_motion_failures_stay_in_contact_denominator():
    valid = {k: 0.1 for k in ("ade_abs", "fde_abs", "ade_rel", "fde_rel")}
    valid["contact_in_mask"] = 1
    out = summarize([valid, {"error": "no detection"}], "motion")
    assert out["contact_in_mask_pct"] == 50
    assert out["ade_abs"] == 0.1
    assert out["num_failures"] == 1
    assert summarize([{"error": "no prediction"}], "motion")["ade_abs"] is None


def test_curve_start_is_median_of_masked_depth():
    depth = np.array([[1000., 2000.], [0., 4000.]])  # mm; zero is missing
    intrinsic = {"fx": 1., "fy": 1., "cx": 2., "cy": 1.}
    assert anchor_point(np.ones((2, 2), bool), depth, intrinsic) == pytest.approx([-2, -1, 2])
    # Without valid depth the curve starts at the camera center, which projects to the principal point.
    start = anchor_point(np.array([[False, False], [True, False]]), depth, intrinsic)
    mask = np.zeros((2, 3), bool)
    mask[1, 2] = True
    assert motion_metrics(np.stack([start, start + 1]), np.ones((2, 3)), intrinsic, mask)["contact_in_mask"] == 1


def test_invalid_trajectory_rejected():
    with pytest.raises(ValueError):
        resample(np.array([[np.nan, 0, 1], [0, 0, 1]]))
    with pytest.raises(ValueError):
        resample(np.zeros((1, 3)))
