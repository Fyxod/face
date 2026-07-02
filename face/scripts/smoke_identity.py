"""Run a minimal FACE identity smoke test."""
from __future__ import annotations

import argparse
from pathlib import Path

from face.core.runner import RunConfig, run_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FACE ArcFace identity smoke.")
    parser.add_argument("--mat-root", required=True)
    parser.add_argument("--arcface-checkpoint", required=True)
    parser.add_argument("--geometry-config", default="configs/geometry_default.json")
    parser.add_argument("--output-root", default="outputs/smoke")
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--init", choices=["neutral", "small_random"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig(
        mat_root=args.mat_root,
        arcface_checkpoint=args.arcface_checkpoint,
        output_root=args.output_root,
        geometry_config_path=args.geometry_config,
        iters=args.iters,
        lr=args.lr,
        init=args.init,
        quick=True,
        mode="smoke_identity",
        skip_downstream_eval=True,
        skip_deepface=True,
    )
    summary = run_matrix(cfg)
    print(f"[face-smoke] wrote: {summary['output_root']}")


if __name__ == "__main__":
    main()
