"""invention_gate: reference-based metrics that catch restored video inventing detail.

Restoration pipelines (classical or learned) are usually scored with PSNR, SSIM
and LPIPS. Those average errors over the whole frame and stay blind to the one
failure mode that matters for archival or evidentiary use: a pipeline that
MANUFACTURES plausible detail, or destroys observed detail while keeping the
average close. invention_gate reports:

  hf_retention  - high-frequency energy of the restoration relative to the
                  reference, per frame (luma). 1.0 means the observed detail
                  level survived, below 1.0 means smoothing, above 1.0 means
                  added energy that the reference does not contain.
  invention     - relative added high-frequency energy on a stable/static mask.
                  On pixels the reference holds constant, any energy the
                  restoration adds beyond the reference is invented.
  hf_pump       - temporal std of the per-frame high-frequency energy. Detail
                  that pumps in and out frame to frame is not detail.
  luma_flicker  - mean absolute temporal derivative of luma on the static mask.

All metrics are GT-assisted and evaluation-side only: they compare a restored
video against a reference. They are NOT a no-reference quality score and they
do not certify that nothing was invented outside the mask; they bound it.
"""
from .metrics import (
    hf_energy, hf_retention, invention_score, hf_pump, luma_flicker,
    evaluate, evaluate_files,
)
from .video import read_frames

__version__ = "0.1.0"
