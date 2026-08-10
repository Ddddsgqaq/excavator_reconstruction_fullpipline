import numpy as np

from gravity_alignment import estimate_gravity


def _extrinsics_from_centers(centers):
    extrinsic = np.zeros((len(centers), 3, 4), dtype=np.float64)
    extrinsic[:, :3, :3] = np.eye(3)
    extrinsic[:, :3, 3] = -np.asarray(centers)
    return extrinsic


def test_strong_semantic_ground_overrides_tilted_orbit_trajectory():
    angle = np.deg2rad(20.0)
    trajectory_normal = np.array([0.0, np.cos(angle), np.sin(angle)])
    basis_x = np.array([1.0, 0.0, 0.0])
    basis_v = np.cross(trajectory_normal, basis_x)
    centers = [
        np.array([0.0, 1.0, 0.0]) + x * basis_x + v * basis_v
        for x in (-1.0, 0.0, 1.0)
        for v in (-1.0, 0.0, 1.0)
    ]

    gx, gz = np.meshgrid(np.linspace(-2.0, 2.0, 40), np.linspace(-2.0, 2.0, 40))
    ground = np.stack([gx, np.zeros_like(gx), gz], axis=-1)[None, ...]
    ground_mask = np.ones(ground.shape[:-1], dtype=bool)
    confidence = np.ones(ground.shape[:-1], dtype=np.float64)

    result = estimate_gravity(
        extrinsic=_extrinsics_from_centers(centers),
        world_points=ground,
        ground_mask=ground_mask,
        conf=confidence,
        conf_thres=0.1,
    )

    assert result.source == "ground_mask"
    assert result.debug["selection_reason"] == "ground_override_on_disagreement"
    assert result.debug["ground_mask"]["ground_inlier_ratio"] > 0.99
    assert abs(float(np.dot(result.n_grav, np.array([0.0, 1.0, 0.0])))) > 0.999
