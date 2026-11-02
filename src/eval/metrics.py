"""Segmentation and 3D motion metrics of the paper's benchmarks (Sec. 5.2)."""
import numpy as np

NUM_POINTS = 50  # predicted and ground-truth curves are linearly resampled to 50 points


def mask_iou(prediction, target):
    prediction, target = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if prediction.shape != target.shape or target.ndim != 2:
        raise ValueError("Prediction and target must be masks with the same (H, W) shape.")
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    return {"intersection": intersection, "union": union, "iou": intersection / union if union else 1.0}


def resample(points, count=NUM_POINTS):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("A trajectory must have shape (T, 3), with T >= 2.")
    if not np.isfinite(points).all():
        raise ValueError("Trajectory contains non-finite coordinates.")
    return np.stack([np.interp(np.linspace(0, 1, count), np.linspace(0, 1, len(points)), points[:, d])
                     for d in range(3)], axis=1)


def anchor_point(mask, depth, intrinsic):
    """Curve start: per-axis median of the masked pixels back-projected with depth (mm), in meters."""
    v, u = np.nonzero(mask)
    z = depth[v, u].astype(np.float64)
    valid = z > 0  # as in the paper's evaluation, only zero depth counts as missing (+inf passes)
    if not valid.any():
        return np.zeros(3)
    u, v, z = u[valid], v[valid], z[valid] / 1000
    return np.array([np.median((u - intrinsic["cx"]) * z / intrinsic["fx"]),
                     np.median((v - intrinsic["cy"]) * z / intrinsic["fy"]), np.median(z)])


def motion_metrics(prediction, target, intrinsic, mask):
    prediction, target = resample(prediction), resample(target)
    result = {}
    for suffix, pred, gt in (("abs", prediction, target),
                             ("rel", prediction - prediction[0], target - target[0])):
        distances = np.linalg.norm(pred - gt, axis=1)
        result.update({f"ade_{suffix}": float(distances.mean()), f"fde_{suffix}": float(distances[-1])})
    # Contact-in-mask: the curve start, projected with depth clamped to 1 mm, lies in the GT mask.
    x, y, z = prediction[0]
    z = max(z, 1e-3)
    u = int(round(x * intrinsic["fx"] / z + intrinsic["cx"]))
    v = int(round(y * intrinsic["fy"] / z + intrinsic["cy"]))
    result["contact_in_mask"] = int(0 <= v < mask.shape[0] and 0 <= u < mask.shape[1] and bool(mask[v, u]))
    return result


def summarize(records, task):
    if not records:
        raise ValueError("Cannot evaluate an empty dataset.")
    valid = [row for row in records if "error" not in row]
    result = {"num_samples": len(records), "num_failures": len(records) - len(valid)}
    if task == "segmentation":
        # Samples without a prediction (no detection above the confidence threshold) are left out
        # of both averages, as in the paper's evaluation.
        intersection = sum(row["intersection"] for row in valid)
        union = sum(row["union"] for row in valid)
        result.update(giou=100 * float(np.mean([row["iou"] for row in valid])) if valid else None,
                      ciou=100 * intersection / union if union else None)
    else:
        for name in ("ade_abs", "fde_abs", "ade_rel", "fde_rel"):
            result[name] = float(np.mean([row[name] for row in valid])) if valid else None
        # Failures stay in the contact-in-mask denominator as misses.
        result["contact_in_mask_pct"] = 100 * sum(row["contact_in_mask"] for row in valid) / len(records)
    return result
