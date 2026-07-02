"""Timing smoke for FACE."""
from __future__ import annotations

import argparse

from face.core.runner import RunConfig, run_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FACE timing smoke.")
    parser.add_argument("--mat-root", required=True)
    parser.add_argument("--arcface-checkpoint", required=True)
    parser.add_argument("--geometry-config", default="configs/geometry_default.json")
    parser.add_argument("--output-root", default="outputs/smoke_timing")
    parser.add_argument("--iters", type=int, default=2)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--all-cases", action="store_true")
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
        quick=args.quick,
        all_cases=args.all_cases,
        mode="smoke_timing",
        skip_downstream_eval=True,
        skip_deepface=True,
    )
    summary = run_matrix(cfg)
    estimates = summary.get("time_estimates", {})
    print(f"[face-timing] wrote: {summary['output_root']}")
    for key in ("estimated_runtime_seconds_for_50_iterations", "estimated_runtime_seconds_for_100_iterations", "estimated_runtime_seconds_for_150_iterations", "estimated_runtime_seconds_for_400_iterations"):
        value = estimates.get(key)
        if value is not None:
            print(f"[face-timing] {key}: {value / 60:.2f} min")


if __name__ == "__main__":
    main()
