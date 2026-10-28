"""CLI: invention-gate RESTORED REFERENCE [--static-mask MASK] [--max-frames N]"""
import argparse
import json
import sys

from . import __version__
from .metrics import evaluate_files


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="invention-gate",
        description="Reference-based invention metrics for restored video.")
    p.add_argument("restored", help="restored video (mp4/avi/npy)")
    p.add_argument("reference", help="reference video (mp4/avi/npy)")
    p.add_argument("--static-mask", default=None,
                   help="optional PNG of the stable region; strongly recommended "
                        "so invention is measured where the reference is constant")
    p.add_argument("--max-frames", type=int, default=None)
    a = p.parse_args(argv)
    report, meta = evaluate_files(a.restored, a.reference, a.static_mask, max_frames=a.max_frames)
    json.dump({"tool": "invention-gate", "version": __version__, **meta, **report},
              sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
