"""Run orchestration for FACE smoke and ArcFace identity jobs."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .cases import RunSpec, build_matrix, print_resolved_cases, resolve_image_path
from .identity import face_loss, identity_objective, prepare_identity_reference
from .image_metrics import delta_to_pil, flow_to_pil, image_metrics, pil_to_tensor, save_sheet, tensor_pair_metrics, tensor_to_pil
from .logging import append_jsonl, nvidia_smi_memory_gb, read_json, write_csv, write_json
from .runtime import torch_device, torch_peak_gb
from .utils import save_input_difference


@dataclass
class RunConfig:
    mat_root: str
    arcface_checkpoint: str
    output_root: str
    geometry_config_path: str = "configs/geometry_default.json"
    iters: int = 150
    lr: float = 0.05
    seed: int = 1234
    init: str | None = None
    quick: bool = False
    all_cases: bool = False
    mode: str = "run_matrix"
    skip_downstream_eval: bool = False
    skip_deepface: bool = False
    force: bool = False
    source_url: str | None = None
    checkpoint_every: int = 25


REQUIRED_HISTORY_FIELDS = {
    "iter",
    "Z",
    "loss",
    "best_Z_so_far",
    "best_iter_so_far",
    "learning_rate",
    "seed",
    "face_id",
    "prompt",
    "case_id",
    "seconds_iter",
    "seconds_elapsed",
    "peak_vram_gb",
    "identity_cosine_similarity_raw",
    "identity_cosine_distance",
    "identity_similarity_score_pct",
    "identity_l2_embedding_distance",
    "identity_angle_radians",
    "identity_angle_degrees",
    "original_embedding_norm",
    "perturbed_embedding_norm",
    "psnr_to_original",
    "ssim_to_original",
    "mse_to_original",
    "l2_to_original",
    "combined_max_disp_px",
    "combined_mean_disp_px",
    "combined_p95_disp_px",
    "jacobian_det_min",
    "foldover_fraction",
    "smoothness_tv",
    "tps_mean_disp",
    "tps_max_disp",
    "tps_p95_disp",
    "tps_param_min",
    "tps_param_max",
    "tps_param_mean_abs",
    "tps_grad_norm",
    "tps_num_at_min",
    "tps_num_at_max",
    "delaunay_mean_disp",
    "delaunay_max_disp",
    "delaunay_p95_disp",
    "delaunay_param_min",
    "delaunay_param_max",
    "delaunay_param_mean_abs",
    "delaunay_grad_norm",
    "delaunay_num_at_min",
    "delaunay_num_at_max",
    "rolling_mean_disp",
    "rolling_max_disp",
    "rolling_p95_disp",
    "rolling_param_min",
    "rolling_param_max",
    "rolling_param_mean_abs",
    "rolling_grad_norm",
    "rolling_num_at_min",
    "rolling_num_at_max",
    "dct_mean_disp",
    "dct_max_disp",
    "dct_p95_disp",
    "dct_param_min",
    "dct_param_max",
    "dct_param_mean_abs",
    "dct_grad_norm",
    "dct_num_at_min",
    "dct_num_at_max",
    "fft_phase_norm",
    "fft_phase_mean_abs",
    "fft_phase_max_abs",
    "fft_phase_grad_norm",
    "fft_phase_num_at_min",
    "fft_phase_num_at_max",
    "legacy_fft_strength_equivalent",
    "fft_spatial_delta_mse",
    "num_total_params",
    "num_clamped_total",
    "fraction_clamped_total",
    "num_at_min_total",
    "num_at_max_total",
    "components_at_boundary",
    "total_geometry_grad_norm",
}


def _run_dir(root: Path, spec: RunSpec) -> Path:
    return root / "runs" / "arcface_identity" / spec.model / spec.case.slug


def _float_terms(terms: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in terms.items():
        if hasattr(value, "detach"):
            out[key] = float(value.detach().float().cpu())
        elif isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _history_fields_ok(row: dict[str, Any]) -> bool:
    return REQUIRED_HISTORY_FIELDS.issubset(set(row))


def _component_flow_images(aux: dict[str, Any], out_dir: Path, scale_px: float) -> None:
    flow_to_pil(aux["displacement"], scale_px).save(out_dir / "combined_flow.png")
    for name, field in aux["fields"].items():
        flow_to_pil(field, scale_px).save(out_dir / f"{name}_flow.png")
    delta_to_pil(aux["fft_delta"]).save(out_dir / "fft_phase_visualization.png")


def _save_checkpoint(run_dir: Path, iteration: int, perturbed, aux: dict[str, Any], row: dict[str, Any], geometry) -> None:
    ckpt = run_dir / "checkpoints" / f"iter_{iteration:03d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(perturbed).save(ckpt / "perturbed.png")
    flow_to_pil(aux["displacement"], geometry.component_limit_for_flow).save(ckpt / "combined_flow.png")
    write_json(ckpt / "metrics.json", row)
    write_json(
        ckpt / "geometry_params.json",
        {
            "limits": geometry.limits_dict(),
            "parameter_diagnostics": geometry.parameter_diagnostics(),
        },
    )


def _arcface(device, checkpoint_path: str, source_url: str | None):
    from face.models.arcface import ArcFaceIResNet100

    return ArcFaceIResNet100(checkpoint_path, device, source_url=source_url)


def optimize_one(spec: RunSpec, cfg: RunConfig, arcface, device, output_dir: Path) -> dict[str, Any]:
    import torch

    from face.core.geometry.combined_face import CombinedFacePerturbation, FaceGeometryConfig, load_face_geometry_config

    output_dir.mkdir(parents=True, exist_ok=True)
    done_path = output_dir / "DONE.json"
    if done_path.exists() and not cfg.force:
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            print(f"[face] skip completed run: {output_dir}")
            return read_json(summary_path)
        raise RuntimeError(f"DONE.json exists but summary.json is missing: {output_dir}. Use --force after inspecting.")

    started = time.monotonic()
    mat_root = Path(cfg.mat_root)
    image_path = resolve_image_path(mat_root, spec.case.face_id)
    print(f"[face] running {spec.slug} image={image_path}")

    original = Image.open(image_path).convert("RGB")
    original.save(output_dir / "original.png")
    original_tensor = pil_to_tensor(original, device)

    geometry_config = load_face_geometry_config(cfg.geometry_config_path) if cfg.geometry_config_path else FaceGeometryConfig()
    if cfg.init:
        geometry_config.init = cfg.init
    torch.manual_seed(spec.run_seed)
    geometry = CombinedFacePerturbation(
        original_tensor.shape[-2],
        original_tensor.shape[-1],
        original_tensor.shape[1],
        device,
        seed=spec.run_seed,
        config=geometry_config,
    )
    optimizer = torch.optim.Adam([p for p in geometry.parameters() if p.requires_grad], lr=cfg.lr)
    projection = geometry.project_()
    reference = prepare_identity_reference(arcface, original_tensor)
    reference.embedding_original.detach().cpu().numpy().astype("float32").tofile(output_dir / "embedding_original.raw")
    np.save(output_dir / "embedding_original.npy", reference.embedding_original.detach().cpu().numpy().astype("float32"))

    config_payload = {
        **asdict(cfg),
        "spec": {
            "experiment": "arcface_identity",
            "model": spec.model,
            "face_id": spec.case.face_id,
            "prompt": spec.case.prompt,
            "case_id": spec.case.slug,
            "seed": spec.run_seed,
            "image_path": str(image_path),
        },
        "experiment_description": "White-box geometric identity optimization against frozen ArcFace iResNet-100, with downstream InstructPix2Pix edit evaluation.",
        "objective": "Z = 1 - cosine_similarity(ArcFace(original), ArcFace(perturbed))",
        "loss": "loss = -Z",
        "arcface_objective_prompt_conditioned": False,
        "prompt_usage": "Prompt is used only for downstream InstructPix2Pix edit evaluation.",
        "no_landmarks_alignment_or_detection": True,
        "no_visual_counter_loss": True,
        "model_weights_frozen": True,
        "optimized_parameters": "geometry_only",
        "arcface": arcface.metadata(),
        "geometry_config_path": cfg.geometry_config_path,
        "geometry_config_resolved": geometry_config.__dict__.copy(),
        "geometry_limits": geometry.limits_dict(),
    }
    write_json(output_dir / "config_resolved.json", config_payload)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    with torch.no_grad():
        p0, aux0 = geometry(original_tensor)
        metrics0 = tensor_pair_metrics(p0, original_tensor, prefix="")
        Z0, terms0 = identity_objective(arcface, p0, reference)
        row0 = {
            "iter": 0,
            "Z": float(Z0.detach().float().cpu()),
            "loss": float(face_loss(Z0).detach().float().cpu()),
            "best_Z_so_far": float(Z0.detach().float().cpu()),
            "best_iter_so_far": 0,
            "learning_rate": cfg.lr,
            "seed": spec.run_seed,
            "face_id": spec.case.face_id,
            "prompt": spec.case.prompt,
            "case_id": spec.case.slug,
            "seconds_iter": 0.0,
            "seconds_elapsed": 0.0,
            "peak_vram_gb": torch_peak_gb(),
            "psnr_to_original": metrics0["psnr"],
            "ssim_to_original": metrics0["ssim"],
            "mse_to_original": metrics0["mse"],
            "l2_to_original": metrics0["l2"],
            **_float_terms(terms0),
            **aux0["diagnostics"],
            **{key: 0.0 for key in geometry.grad_norms()},
            **geometry.parameter_diagnostics(),
            **projection,
        }
        row0["total_geometry_grad_norm"] = 0.0
        _save_checkpoint(output_dir, 0, p0, aux0, row0, geometry)

    for iteration in range(1, cfg.iters + 1):
        iter_started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        perturbed, aux = geometry(original_tensor)
        Z, terms = identity_objective(arcface, perturbed, reference)
        loss = face_loss(Z)
        finite = bool(torch.isfinite(loss).item() and torch.isfinite(perturbed).all().item())
        if not finite:
            raise FloatingPointError(f"Non-finite Z/loss at iteration {iteration}")
        loss.backward()
        grad_norms = geometry.grad_norms()
        optimizer.step()
        projection = geometry.project_()

        with torch.no_grad():
            metrics_original = tensor_pair_metrics(perturbed, original_tensor, prefix="")
        seconds_iter = time.monotonic() - iter_started
        prev_best = best["row"]["Z"] if best is not None else -1e30
        row: dict[str, Any] = {
            "iter": iteration,
            "Z": float(Z.detach().float().cpu()),
            "loss": float(loss.detach().float().cpu()),
            "best_Z_so_far": float(max(prev_best, float(Z.detach().float().cpu()))),
            "best_iter_so_far": iteration if best is None or float(Z.detach().float().cpu()) > prev_best else best["row"]["iter"],
            "learning_rate": cfg.lr,
            "seed": spec.run_seed,
            "face_id": spec.case.face_id,
            "prompt": spec.case.prompt,
            "case_id": spec.case.slug,
            "seconds_iter": seconds_iter,
            "seconds_elapsed": time.monotonic() - started,
            "peak_vram_gb": torch_peak_gb(),
            "psnr_to_original": metrics_original["psnr"],
            "ssim_to_original": metrics_original["ssim"],
            "mse_to_original": metrics_original["mse"],
            "l2_to_original": metrics_original["l2"],
            **_float_terms(terms),
            **aux["diagnostics"],
            **grad_norms,
            **geometry.parameter_diagnostics(),
            **projection,
        }
        row["total_geometry_grad_norm"] = row.get("total_grad_norm", 0.0)
        rows.append(row)
        append_jsonl(output_dir / "history.jsonl", row)
        if best is None or row["Z"] > best["row"]["Z"]:
            best = {
                "row": row,
                "theta_state": geometry.theta_state(),
                "perturbed": perturbed.detach().clone(),
                "aux": {
                    "displacement": aux["displacement"].detach().clone(),
                    "fields": {k: v.detach().clone() for k, v in aux["fields"].items()},
                    "fft_delta": aux["fft_delta"].detach().clone(),
                },
            }
        if iteration % max(1, cfg.checkpoint_every) == 0 or iteration == cfg.iters:
            _save_checkpoint(output_dir, iteration, perturbed.detach(), aux, row, geometry)

    if not rows or best is None:
        raise RuntimeError("No finite optimization iteration completed.")

    with torch.no_grad():
        final_perturbed_tensor, final_aux = geometry(original_tensor)
        final_Z, final_terms = identity_objective(arcface, final_perturbed_tensor, reference)

    final_perturbed = tensor_to_pil(final_perturbed_tensor)
    best_perturbed = tensor_to_pil(best["perturbed"])
    final_perturbed.save(output_dir / "perturbed_final.png")
    best_perturbed.save(output_dir / "perturbed_best.png")
    _component_flow_images(final_aux, output_dir, geometry.component_limit_for_flow)
    flow_to_pil(best["aux"]["displacement"], geometry.component_limit_for_flow).save(output_dir / "combined_flow_best.png")
    (output_dir / "combined_flow.png").replace(output_dir / "combined_flow_final.png")
    # Keep a conventional alias for report scripts.
    flow_to_pil(final_aux["displacement"], geometry.component_limit_for_flow).save(output_dir / "combined_flow.png")
    save_input_difference(output_dir / "original.png", output_dir / "perturbed_best.png", output_dir / "input_difference_best.png")
    save_input_difference(output_dir / "original.png", output_dir / "perturbed_final.png", output_dir / "input_difference_final.png")

    torch.save(geometry.theta_state(), output_dir / "theta_final.pt")
    torch.save(best["theta_state"], output_dir / "theta_best.pt")
    write_json(output_dir / "geometry_params_final.json", {"limits": geometry.limits_dict(), "parameter_diagnostics": geometry.parameter_diagnostics(), "last_projection": projection})
    write_json(output_dir / "geometry_params_best.json", {"best_iter_by_Z": best["row"]["iter"], "best_Z": best["row"]["Z"]})
    np.save(output_dir / "embedding_perturbed_final.npy", arcface.embedding(final_perturbed_tensor).detach().cpu().numpy().astype("float32"))
    np.save(output_dir / "embedding_perturbed_best.npy", arcface.embedding(best["perturbed"]).detach().cpu().numpy().astype("float32"))

    edit_metadata: dict[str, Any] = {"downstream_eval_skipped": cfg.skip_downstream_eval}
    if cfg.skip_downstream_eval:
        original.copy().save(output_dir / "original_edited.png")
        best_perturbed.copy().save(output_dir / "perturbed_best_edited.png")
        final_perturbed.copy().save(output_dir / "perturbed_final_edited.png")
    else:
        try:
            from face.evaluation.instruct import InstructPix2PixEvaluator

            evaluator = InstructPix2PixEvaluator(device)
            clean_edit = evaluator.generate_edit(original, spec.case.prompt, spec.run_seed)
            pert_best_edit = evaluator.generate_edit(best_perturbed, spec.case.prompt, spec.run_seed)
            pert_final_edit = evaluator.generate_edit(final_perturbed, spec.case.prompt, spec.run_seed)
            clean_edit.save(output_dir / "original_edited.png")
            pert_best_edit.save(output_dir / "perturbed_best_edited.png")
            pert_final_edit.save(output_dir / "perturbed_final_edited.png")
            edit_metadata = {**evaluator.metadata(), "downstream_eval_skipped": False}
        except Exception as error:
            edit_metadata = {"downstream_eval_skipped": False, "downstream_eval_error": repr(error)}
            original.copy().save(output_dir / "original_edited.png")
            best_perturbed.copy().save(output_dir / "perturbed_best_edited.png")
            final_perturbed.copy().save(output_dir / "perturbed_final_edited.png")

    input_metrics_best = image_metrics(original, best_perturbed)
    input_metrics_final = image_metrics(original, final_perturbed)
    output_metrics_best = image_metrics(Image.open(output_dir / "original_edited.png"), Image.open(output_dir / "perturbed_best_edited.png"))
    output_metrics_final = image_metrics(Image.open(output_dir / "original_edited.png"), Image.open(output_dir / "perturbed_final_edited.png"))
    save_sheet(
        output_dir / "comparison_sheet.png",
        [
            ("Original", original),
            ("Perturbed Best", best_perturbed),
            ("Abs Difference x8", Image.open(output_dir / "input_difference_best.png")),
            ("Combined Flow", Image.open(output_dir / "combined_flow_best.png")),
            ("Clean Edit", Image.open(output_dir / "original_edited.png")),
            ("Perturbed Edit", Image.open(output_dir / "perturbed_best_edited.png")),
        ],
    )

    if not cfg.skip_deepface:
        from face.evaluation.deepface_panel import run_deepface_panel, write_identity_panel

        panel_rows = run_deepface_panel(output_dir)
        write_identity_panel(output_dir, panel_rows)
    else:
        panel_rows = []
        write_json(output_dir / "identity_panel.json", [{"status": "skipped"}])
        (output_dir / "identity_panel.csv").write_text("status\nskipped\n", encoding="utf-8")

    write_csv(output_dir / "history.csv", rows)
    elapsed = time.monotonic() - started
    final_row = rows[-1]
    summary = {
        "status": "done",
        "experiment": "arcface_identity",
        "model": spec.model,
        "face_id": spec.case.face_id,
        "prompt": spec.case.prompt,
        "case_id": spec.case.slug,
        "seed": spec.run_seed,
        "iters": cfg.iters,
        "Z_definition": "1 - ArcFace cosine similarity between original and perturbed full-image embeddings",
        "loss": "loss = -Z",
        "arcface_objective_prompt_conditioned": False,
        "final_Z": float(final_Z.detach().float().cpu()),
        "final_loss": -float(final_Z.detach().float().cpu()),
        "best_iter_by_Z": best["row"]["iter"],
        "best_Z": best["row"]["Z"],
        "final_identity_cosine_similarity_raw": float(_float_terms(final_terms)["identity_cosine_similarity_raw"]),
        "final_identity_similarity_score_pct": float(_float_terms(final_terms)["identity_similarity_score_pct"]),
        "best_identity_cosine_similarity_raw": best["row"].get("identity_cosine_similarity_raw"),
        "best_identity_similarity_score_pct": best["row"].get("identity_similarity_score_pct"),
        "mean_seconds_iter": float(sum(row["seconds_iter"] for row in rows) / max(len(rows), 1)),
        "elapsed_seconds": elapsed,
        "final_psnr_to_original": final_row["psnr_to_original"],
        "final_ssim_to_original": final_row["ssim_to_original"],
        "final_mse_to_original": final_row["mse_to_original"],
        "input_best_ssim": input_metrics_best["ssim"],
        "input_best_psnr": input_metrics_best["psnr"],
        "input_best_l2": input_metrics_best["l2"],
        "input_final_ssim": input_metrics_final["ssim"],
        "input_final_psnr": input_metrics_final["psnr"],
        "input_final_l2": input_metrics_final["l2"],
        "best_output_ssim": output_metrics_best["ssim"],
        "best_output_psnr": output_metrics_best["psnr"],
        "best_output_l2": output_metrics_best["l2"],
        "final_output_ssim": output_metrics_final["ssim"],
        "final_output_psnr": output_metrics_final["psnr"],
        "final_output_l2": output_metrics_final["l2"],
        "final_combined_max_disp_px": final_row["combined_max_disp_px"],
        "final_combined_mean_disp_px": final_row["combined_mean_disp_px"],
        "final_combined_p95_disp_px": final_row["combined_p95_disp_px"],
        "final_fraction_clamped_total": final_row["fraction_clamped_total"],
        "all_required_history_fields_populated": _history_fields_ok(final_row),
        "clamp_project_logic_active": final_row["num_total_params"] > 0,
        "arcface": arcface.metadata(),
        "downstream_instructpix2pix": edit_metadata,
        "deepface_rows": len(panel_rows),
        "peak_vram_gb": torch_peak_gb(),
        "nvidia_smi_memory_gb": nvidia_smi_memory_gb(),
        "run_dir": str(output_dir),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "DONE.json", {"status": "done", "elapsed_seconds": elapsed, "final_Z": summary["final_Z"]})
    return summary


def _time_estimates(summaries: list[dict[str, Any]], wall_seconds: float, observed_iters: int) -> dict[str, Any]:
    mean_seconds = [float(row.get("mean_seconds_iter", 0.0)) for row in summaries]
    observed_iter_seconds = sum(mean_seconds)
    fixed_overhead = max(0.0, float(wall_seconds) - float(observed_iters) * observed_iter_seconds)
    completed = max(len(summaries), 1)
    scale_to_full = 4.0 / completed
    full_iter_seconds = observed_iter_seconds * scale_to_full
    full_overhead = fixed_overhead * scale_to_full
    return {
        "observed_completed_runs": len(summaries),
        "observed_mean_seconds_per_iteration_per_run": float(sum(mean_seconds) / max(len(mean_seconds), 1)),
        "estimated_full_matrix_seconds_per_iteration": full_iter_seconds,
        "estimated_fixed_overhead_seconds": full_overhead,
        "estimated_runtime_seconds_for_50_iterations": full_overhead + 50 * full_iter_seconds,
        "estimated_runtime_seconds_for_100_iterations": full_overhead + 100 * full_iter_seconds,
        "estimated_runtime_seconds_for_150_iterations": full_overhead + 150 * full_iter_seconds,
        "estimated_runtime_seconds_for_400_iterations": full_overhead + 400 * full_iter_seconds,
    }


def _write_top_summary(run_root: Path, cfg: RunConfig, started: float, summaries: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    wall = time.monotonic() - started
    status = "done" if not failures else "failed"
    estimates = _time_estimates(summaries, wall, cfg.iters)
    payload = {
        "status": status,
        "mode": cfg.mode,
        "experiment": "arcface_identity",
        "iters": cfg.iters,
        "quick": cfg.quick,
        "all_cases": cfg.all_cases,
        "execution": "sequential",
        "wall_seconds": wall,
        "num_runs_attempted": len(summaries) + len(failures),
        "num_runs_completed": len(summaries),
        "num_failures": len(failures),
        "failures": failures,
        "summaries": summaries,
        "time_estimates": estimates,
        "peak_vram_gb": torch_peak_gb(),
        "nvidia_smi_memory_gb": nvidia_smi_memory_gb(),
        "all_per_iteration_logging_fields_populated": all(s.get("all_required_history_fields_populated", False) for s in summaries),
        "clamp_project_logic_active": all(s.get("clamp_project_logic_active", False) for s in summaries),
        "output_root": str(run_root),
    }
    write_json(run_root / "summary.json", payload)
    lines = [
        f"# FACE {cfg.mode} summary",
        "",
        f"- status: {status}",
        "- experiment: arcface_identity",
        "- execution: sequential",
        f"- iterations per run: {cfg.iters}",
        f"- runs attempted: {payload['num_runs_attempted']}",
        f"- runs completed: {payload['num_runs_completed']}",
        f"- failures: {payload['num_failures']}",
        f"- wall seconds: {wall:.2f}",
        f"- observed mean seconds/iteration/run: {estimates['observed_mean_seconds_per_iteration_per_run']:.3f}",
        f"- estimated 150-iteration full matrix: {estimates['estimated_runtime_seconds_for_150_iterations'] / 60:.1f} min",
        f"- peak VRAM GB: {payload.get('peak_vram_gb')}",
        f"- all required per-iteration fields populated: {payload['all_per_iteration_logging_fields_populated']}",
        f"- clamp/project logic active: {payload['clamp_project_logic_active']}",
        "",
    ]
    if failures:
        lines.extend(["## Failures", ""])
        for failure in failures:
            lines.append(f"- {failure.get('spec')}: {failure.get('error')}")
    (run_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def run_matrix(cfg: RunConfig) -> dict[str, Any]:
    started = time.monotonic()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    label = "quick" if cfg.quick else "all"
    root = Path(cfg.output_root) / f"{run_id}_arcface_identity_{label}_sequential"
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "launcher_config.json", asdict(cfg))
    print_resolved_cases(Path(cfg.mat_root))
    specs = build_matrix(quick=cfg.quick)
    if cfg.all_cases:
        specs = build_matrix(quick=False)
    device = torch_device()
    arcface = _arcface(device, cfg.arcface_checkpoint, cfg.source_url)
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for spec in specs:
        run_dir = _run_dir(root, spec)
        try:
            summaries.append(optimize_one(spec, cfg, arcface, device, run_dir))
        except Exception as error:
            failures.append({"spec": spec.slug, "error": repr(error), "run_dir": str(run_dir)})
            write_json(run_dir / "FAILED.json", {"status": "failed", "error": repr(error)})
    return _write_top_summary(root, cfg, started, summaries, failures)
