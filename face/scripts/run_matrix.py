"""Run the full FACE ArcFace identity matrix."""
from __future__ import annotations

import argparse

from face.core.runner import RunConfig, run_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FACE ArcFace identity optimization matrix.")
    parser.add_argument("--mat-root", required=True)
    parser.add_argument("--arcface-checkpoint", required=True)
    parser.add_argument("--geometry-config", default="configs/geometry_default.json")
    parser.add_argument("--output-root", default="outputs/arcface_identity_dct_image")
    parser.add_argument("--iters", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--init", choices=["neutral", "small_random"], default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-downstream-eval", action="store_true")
    parser.add_argument("--skip-deepface", action="store_true")
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
        quick=False,
        all_cases=True,
        mode="run_matrix",
        skip_downstream_eval=args.skip_downstream_eval,
        skip_deepface=args.skip_deepface,
        force=args.force,
    )
    summary = run_matrix(cfg)
    print(f"[face-run] wrote: {summary['output_root']}")


if __name__ == "__main__":
    main()
