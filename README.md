# invention-gate

Reference-based metrics that catch restored video **inventing or destroying detail**, where
PSNR, SSIM and LPIPS stay blind. Built out of an analog FPV restoration project where the
rule was: restoration is concealment, never invention. The same failure mode applies to any
denoiser, upscaler or frame-interpolation pipeline, classical or learned.

## Why

Averaged metrics reward a pipeline that manufactures plausible texture. A per-pixel
comparison on the pixels that *should not change* does not. invention_gate reports four
numbers per clip, all on luma, all GT-assisted (evaluation-side only):

| metric | question it answers |
|---|---|
| `hf_retention` | did observed detail survive? ~1.0 yes, <1 smoothing, >1 added energy |
| `invention` | how much high-frequency energy was manufactured on pixels the reference holds stable |
| `hf_pump` | does detail pump in and out frame to frame (pumping detail is not detail) |
| `luma_flicker` | temporal luma derivative, optionally masked to the stable region |

Honest caveat: these metrics bound invention on the static mask. They cannot certify that
nothing was invented outside it. Nothing can, that is the point of the gate: measure what
is measurable, flag the rest.

## Install

```bash
pip install invention-gate            # numpy only
pip install invention-gate[video]     # + opencv for mp4/avi input
```

## Use

```bash
# CLI: restored vs reference, with a static-region mask (recommended)
invention-gate restored.mp4 reference.mp4 --static-mask stable.png
```

```python
from invention_gate import evaluate

report = evaluate(restored_frames, reference_frames, static_mask=mask)
# {"hf_retention": 0.97, "invention": 0.08, "hf_pump_out": 0.4, ...}
```

Inputs: video files (mp4/avi/mov via opencv), `.npy` frame stacks `(T,H,W,3)` or `(T,H,W)`,
or lists of HxWx3 arrays. Mask: HxW boolean array or a PNG (nonzero = stable).

## Reading the numbers

- `hf_retention` in `[0.9, 1.1]` is usually fine for denoisers; a sharpening pipeline will
  drift above, a heavy denoiser below.
- `invention` should sit near 0 on a static mask. Values around 1 mean the restoration adds
  as much detail energy as the reference actually contains.
- `hf_pump_out` vs `hf_pump_ref`: if the restored side pumps several times harder than the
  reference, the pipeline is temporal-unstable regardless of per-frame scores.
- Per-frame `hf_retention` is included so a single bad segment cannot hide in the mean.

## Provenance-aware gating (the stronger form)

The metrics above are what you can measure from two videos alone. If you control the
restoration pipeline, the stronger gate is to make it emit provenance per frame
(observed / warped / interpolated pixel fractions), leave-one-out fill instability on
inpaint regions, and mask-aware retention on observed pixels only. That closes the loop
between what the pipeline claims it did and what landed in the pixels. This repo keeps to
the two-video contract; the pipeline-side pattern is documented here because it motivated
the design.

## Development

```bash
pip install -e .[video,test]
pytest
```

## License

MIT
