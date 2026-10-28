"""Reference-based invention metrics on luma. Frames are HxWx3 RGB uint8 or HxW float."""
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False


def _luma(frame):
    if frame.ndim == 2:
        y = frame.astype(np.float32)
    else:
        if not _HAS_CV2:
            # BT.601 luma fallback so numpy-only installs still work
            y = frame.astype(np.float32)
            return 0.299 * y[..., 0] + 0.587 * y[..., 1] + 0.114 * y[..., 2]
        y = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)
    return y


def _blur(y, sigma=2.0):
    if _HAS_CV2:
        return cv2.GaussianBlur(y, (0, 0), sigma)
    # separable gaussian via numpy when opencv is absent
    k = int(4 * sigma + 0.5)
    xs = np.arange(-k, k + 1, dtype=np.float32)
    g = np.exp(-(xs ** 2) / (2 * sigma * sigma))
    g /= g.sum()
    return np.apply_along_axis(lambda m: np.convolve(m, g, mode="same"), 0,
                               np.apply_along_axis(lambda m: np.convolve(m, g, mode="same"), 1, y))


def hf_energy(frame, sigma=2.0):
    """std of the highpass (luma - blur) of one frame: its detail energy."""
    y = _luma(frame)
    return float((y - _blur(y, sigma)).std())


def _check_pair(out, ref):
    out = np.asarray(out)
    ref = np.asarray(ref)
    assert out.shape == ref.shape, f"shape mismatch {out.shape} vs {ref.shape}"
    return out, ref


def hf_retention(out, ref, mask=None, sigma=2.0):
    """Mean over frames of std(hp(out)) / std(hp(ref)) on the mask.

    ~1.0 detail preserved; <1 smoothing; >1 energy the reference lacks.
    Returns (value, per_frame) where value is None when nothing was measurable.
    """
    out, ref = _check_pair(out, ref)
    per_frame = []
    for o, r in zip(out, ref):
        yo, yr = _luma(o), _luma(r)
        ho = yo - _blur(yo, sigma)
        hr = yr - _blur(yr, sigma)
        if mask is not None:
            m = np.broadcast_to(mask, ho.shape).astype(bool)
            if m.sum() < 32:
                continue
            ho, hr = ho[m], hr[m]
        if hr.std() < 1e-3 and ho.std() < 1e-3:
            continue  # nothing measurable on a detail-free reference
        per_frame.append(ho.std() / max(hr.std(), 1.0))
    return (float(np.mean(per_frame)) if per_frame else None), per_frame


def invention_score(out, ref, static_mask=None, sigma=2.0):
    """Relative ADDED high-frequency energy on pixels the reference holds stable.

    Per frame: std( max(|hp(out)[m]| - |hp(ref)[m]|, 0) ) / std(hp(ref)[m]).
    Only energy the restoration puts where the reference has less counts, so
    smoothing scores ~0 (removal is hf_retention's job) while manufactured
    texture and injected noise score high. With no mask this measures added
    energy everywhere, which confounds real motion with invention: pass a
    static mask for an honest number.
    """
    out, ref = _check_pair(out, ref)
    vals = []
    for o, r in zip(out, ref):
        yo, yr = _luma(o), _luma(r)
        ho = yo - _blur(yo, sigma)
        hr = yr - _blur(yr, sigma)
        m = np.broadcast_to(static_mask, ho.shape).astype(bool) if static_mask is not None \
            else np.ones(ho.shape, bool)
        if m.sum() < 32:
            continue
        excess = np.maximum(np.abs(ho[m]) - np.abs(hr[m]), 0.0)
        vals.append(excess.std() / max(hr[m].std(), 1.0))
    return float(np.mean(vals)) if vals else None


def hf_pump(frames, sigma=2.0):
    """Temporal std of per-frame high-frequency energy: detail that pumps is not detail."""
    frames = np.asarray(frames)
    e = np.array([hf_energy(f, sigma) for f in frames])
    return float(e.std())


def luma_flicker(frames, static_mask=None):
    """Mean absolute temporal derivative of luma, optionally on a static mask only."""
    frames = np.asarray(frames)
    if len(frames) < 2:
        return None
    ys = np.stack([_luma(f) for f in frames])
    if static_mask is not None:
        m = np.broadcast_to(static_mask, ys.shape[1:]).astype(bool)
        if m.sum() < 32:
            return None
        ys = ys[:, m]
    d = np.abs(np.diff(ys, axis=0))
    return float(d.mean())


def evaluate(out, ref, static_mask=None, sigma=2.0):
    """Full report for a restored clip vs its reference."""
    retention, per_frame = hf_retention(out, ref, static_mask, sigma)
    return dict(
        hf_retention=retention,
        hf_retention_per_frame=[round(v, 4) for v in per_frame],
        invention=invention_score(out, ref, static_mask, sigma),
        hf_pump_out=hf_pump(out, sigma),
        hf_pump_ref=hf_pump(ref, sigma),
        luma_flicker_out=luma_flicker(out, static_mask),
        luma_flicker_ref=luma_flicker(ref, static_mask),
        n_frames=int(len(out)),
    )


def evaluate_files(out_path, ref_path, static_mask_path=None, sigma=2.0, max_frames=None):
    """Evaluate two video files (mp4/avi/npy). Returns (report, meta)."""
    from .video import read_frames
    out = read_frames(out_path, max_frames)
    ref = read_frames(ref_path, max_frames)
    n = min(len(out), len(ref))
    out, ref = out[:n], ref[:n]
    meta = dict(out_frames=len(out), ref_frames=len(ref), compared=n)
    mask = None
    if static_mask_path:
        m = read_frames(static_mask_path, 1)[0]
        mask = (np.asarray(m) > 127).any(axis=-1) if m.ndim == 3 else (np.asarray(m) > 127)
        meta["static_fraction"] = round(float(mask.mean()), 4)
    return evaluate(out, ref, mask, sigma), meta
