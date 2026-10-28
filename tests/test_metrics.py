"""Sanity tests on synthetic frames: the metrics must separate smoothing,
invention and faithful restoration."""
import numpy as np
import pytest

from invention_gate import (
    hf_energy, hf_retention, invention_score, hf_pump, luma_flicker, evaluate,
)

rng = np.random.default_rng(23)


def base_video(n=6, size=64, detail=12.0, motionless=True):
    """Frames with known high-frequency texture; optionally static per frame."""
    tex = rng.normal(128, detail, (size, size)).astype(np.float32)
    frames = []
    for t in range(n):
        f = np.stack([tex] * 3, axis=-1)
        if not motionless:
            f = np.roll(f, t, axis=1)
        frames.append(f)
    return np.array(frames)


def blur_video(video, k=5):
    import cv2
    return np.array([cv2.GaussianBlur(f, (0, 0), k) for f in video])


def test_identity_has_unit_retention():
    v = base_video()
    r, per = hf_retention(v, v)
    assert abs(r - 1.0) < 1e-6
    assert all(abs(p - 1.0) < 1e-6 for p in per)


def test_blur_destroys_detail_retention_below_one():
    v = base_video()
    r, _ = hf_retention(blur_video(v), v)
    assert r < 0.5, r


def test_added_noise_pushes_retention_above_one():
    v = base_video()
    noisy = v + rng.normal(0, 20, v.shape).astype(np.float32)
    r, _ = hf_retention(noisy, v)
    assert r > 1.2, r


def test_invention_detects_manufactured_texture_on_static():
    v = base_video()
    out = v.copy()
    # manufacture texture the reference does not contain, on a stable region
    mask = np.zeros(v.shape[1:3], bool)
    mask[8:24, 8:24] = True
    out[:, mask] += rng.normal(0, 10, (len(v), mask.sum(), 3)).astype(np.float32)
    inv = invention_score(out, v, static_mask=mask)
    assert inv > 0.2, inv
    # a faithful restoration scores ~0 on the same mask
    assert invention_score(v, v, static_mask=mask) < 1e-6


def test_invention_stays_low_for_smoothing():
    # smoothing REMOVES energy, it does not add it: invention must stay ~0
    v = base_video()
    inv = invention_score(blur_video(v), v)
    assert inv < 0.01, inv


def test_hf_pump_separates_pumping_from_stable_detail():
    stable = base_video(n=8)
    pumping = base_video(n=8).copy()
    for t in range(0, 8, 2):
        pumping[t] += rng.normal(0, 8, pumping.shape[1:]).astype(np.float32)
    assert hf_pump(pumping) > 3 * hf_pump(stable)


def test_flicker_finite_and_masked():
    v = base_video()
    mask = np.ones(v.shape[1:3], bool)
    assert luma_flicker(v, mask) is not None
    assert abs(luma_flicker(v) - luma_flicker(v, mask)) < 1e-6


def test_evaluate_report_shape():
    v = base_video()
    rep = evaluate(v, v)
    for k in ("hf_retention", "invention", "hf_pump_out", "hf_pump_ref",
              "luma_flicker_out", "luma_flicker_ref", "n_frames"):
        assert k in rep
    assert rep["n_frames"] == len(v)


def test_shape_mismatch_raises():
    with pytest.raises(AssertionError):
        hf_retention(base_video(4), base_video(5))
