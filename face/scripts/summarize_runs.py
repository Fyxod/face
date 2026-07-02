"""Summarize FACE ArcFace identity runs."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageEnhance, ImageOps


TITLE = "FACE: ArcFace White-box Geometric Identity Optimization"
SUBTITLE = "Frozen iResNet-100 identity-distance results with downstream InstructPix2Pix evaluation"
AUTHOR = "Parth Katiyar"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize FACE result folders.")
    parser.add_argument("--results-root", default="outputs/arcface_identity")
    parser.add_argument("--output-root", default="outputs/reports/arcface_identity")
    parser.add_argument("--run-folder", default=None)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--compress-images", action="store_true")
    return parser.parse_args()


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def fmt(value: Any, digits: int = 4) -> str:
    number = to_float(value)
    if number is None:
        return "" if value is None else str(value)
    if abs(number) >= 100:
        return f"{number:.2f}"
    if abs(number) >= 10:
        return f"{number:.3f}"
    return f"{number:.{digits}f}"


def slug(value: str) -> str:
    out = []
    for char in value.lower():
        if char.isalnum():
            out.append(char)
        elif char in {" ", "-", "_", "/", "."}:
            out.append("_")
    text = "".join(out)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def find_latest_run_root(results_root: Path) -> Path:
    candidates = []
    for child in sorted(results_root.iterdir()) if results_root.exists() else []:
        if not child.is_dir():
            continue
        if list(child.glob("runs/arcface_identity/arcface_iresnet100/*/summary.json")):
            candidates.append(child)
    if not candidates:
        raise FileNotFoundError(f"No FACE run roots found under {results_root}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def resolve_run_root(args: argparse.Namespace) -> Path:
    results_root = Path(args.results_root)
    if args.run_root:
        path = Path(args.run_root)
        if path.exists():
            return path
        print(f"[face-report] requested run root missing, falling back to latest: {path}")
    if args.run_folder:
        path = results_root / args.run_folder
        if path.exists():
            return path
        print(f"[face-report] requested run folder missing, falling back to latest: {path}")
    return find_latest_run_root(results_root)


def collect_runs(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    runs: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for summary_path in sorted(run_root.glob("runs/arcface_identity/arcface_iresnet100/*/summary.json")):
        run_dir = summary_path.parent
        summary = read_json(summary_path)
        config = read_json(run_dir / "config_resolved.json") if (run_dir / "config_resolved.json").exists() else {}
        history_rows = read_csv_rows(run_dir / "history.csv")
        spec = config.get("spec", {})
        face_id = str(summary.get("face_id") or spec.get("face_id") or run_dir.name.split("__")[0])
        prompt = str(summary.get("prompt") or spec.get("prompt") or "")
        images = {
            "original": run_dir / "original.png",
            "perturbed_best": run_dir / "perturbed_best.png",
            "input_difference": run_dir / "input_difference_best.png",
            "combined_flow": run_dir / "combined_flow_best.png",
            "clean_edit": run_dir / "original_edited.png",
            "perturbed_edit": run_dir / "perturbed_best_edited.png",
            "comparison_sheet": run_dir / "comparison_sheet.png",
        }
        for label, path in images.items():
            if not path.exists():
                missing.append({"case": f"{face_id} / {prompt}", "artifact": label, "path": str(path)})
        runs.append(
            {
                "face_id": face_id,
                "prompt": prompt,
                "case": f"{face_id} / {prompt}",
                "case_slug": slug(f"{face_id}_{prompt}"),
                "run_dir": str(run_dir),
                "summary": summary,
                "config": config,
                "history_rows": history_rows,
                "images": images,
            }
        )
    return runs, missing


def make_strip(run: dict[str, Any], output_root: Path, compress: bool) -> str:
    out_dir = output_root / "strips"
    out_dir.mkdir(parents=True, exist_ok=True)
    size = (360, 360) if compress else (512, 512)
    labels = [
        ("Original", run["images"]["original"]),
        ("Perturbed Best", run["images"]["perturbed_best"]),
        ("Abs Difference x8", run["images"]["input_difference"]),
        ("Clean Edit", run["images"]["clean_edit"]),
        ("Perturbed Edit", run["images"]["perturbed_edit"]),
    ]
    cells = []
    for label, path in labels:
        if path.exists():
            img = Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
        else:
            img = Image.new("RGB", size, "#f3f4f6")
        cells.append((label, img))
    label_h = 34
    canvas = Image.new("RGB", (size[0] * len(cells), size[1] + label_h), "white")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    for idx, (label, img) in enumerate(cells):
        x = idx * size[0]
        canvas.paste(img, (x, 0))
        draw.text((x + 8, size[1] + 9), label, fill="black")
    ext = "jpg" if compress else "png"
    path = out_dir / f"face_{run['case_slug']}.{ext}"
    if compress:
        canvas.save(path, quality=82, optimize=True)
    else:
        canvas.save(path, optimize=True)
    return path.relative_to(output_root).as_posix()


def plot_lines(path: Path, title: str, ylabel: str, runs: list[dict[str, Any]], key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5.2), dpi=125)
    for run in runs:
        xs, ys = [], []
        for row in run["history_rows"]:
            x = to_float(row.get("iter"))
            y = to_float(row.get(key))
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
        if xs:
            plt.plot(xs, ys, linewidth=1.8, label=f"{run['face_id']} / {run['prompt'].replace('add ', '')}")
    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_scatter(path: Path, title: str, runs: list[dict[str, Any]], x_key: str, y_key: str, xlabel: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 5.5), dpi=125)
    for run in runs:
        summary = run["summary"]
        x = to_float(summary.get(x_key))
        y = to_float(summary.get(y_key))
        if x is not None and y is not None:
            plt.scatter([x], [y], s=70, label=f"{run['face_id']} / {run['prompt'].replace('add ', '')}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_components(path: Path, runs: list[dict[str, Any]]) -> None:
    keys = ["tps_max_disp", "delaunay_max_disp", "rolling_max_disp", "dct_max_disp"]
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5.2), dpi=125)
    for key in keys:
        xs, ys = [], []
        for run in runs:
            values = [to_float(row.get(key)) for row in run["history_rows"]]
            values = [v for v in values if v is not None]
            if values:
                xs.append(run["case"].replace("add ", ""))
                ys.append(values[-1])
        if ys:
            plt.plot(xs, ys, marker="o", linewidth=1.8, label=key)
    plt.title("Final component max displacement by run")
    plt.ylabel("pixels")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def make_graphs(runs: list[dict[str, Any]], output_root: Path) -> list[dict[str, str]]:
    graph_dir = output_root / "graphs"
    graphs = []
    specs = [
        ("Z vs iteration", "Z", "Z", "z_vs_iteration.png"),
        ("Loss vs iteration", "loss", "loss", "loss_vs_iteration.png"),
        ("ArcFace cosine similarity vs iteration", "identity_cosine_similarity_raw", "cosine similarity", "cosine_similarity_vs_iteration.png"),
        ("ArcFace cosine distance vs iteration", "identity_cosine_distance", "cosine distance", "cosine_distance_vs_iteration.png"),
        ("Cosine identity similarity score (%) vs iteration", "identity_similarity_score_pct", "score (%)", "similarity_score_pct_vs_iteration.png"),
        ("PSNR to original vs iteration", "psnr_to_original", "PSNR", "psnr_vs_iteration.png"),
        ("SSIM to original vs iteration", "ssim_to_original", "SSIM", "ssim_vs_iteration.png"),
        ("Combined max displacement vs iteration", "combined_max_disp_px", "pixels", "combined_max_disp_vs_iteration.png"),
        ("Fraction clamped vs iteration", "fraction_clamped_total", "fraction", "fraction_clamped_vs_iteration.png"),
    ]
    for title, key, ylabel, name in specs:
        path = graph_dir / name
        plot_lines(path, title, ylabel, runs, key)
        graphs.append({"title": title, "path": path.relative_to(output_root).as_posix()})
    path = graph_dir / "final_z_vs_ssim.png"
    plot_scatter(path, "Final Z vs final SSIM", runs, "final_ssim_to_original", "final_Z", "SSIM to original", "final Z")
    graphs.append({"title": "Final Z vs final SSIM", "path": path.relative_to(output_root).as_posix()})
    path = graph_dir / "final_z_vs_psnr.png"
    plot_scatter(path, "Final Z vs final PSNR", runs, "final_psnr_to_original", "final_Z", "PSNR to original", "final Z")
    graphs.append({"title": "Final Z vs final PSNR", "path": path.relative_to(output_root).as_posix()})
    path = graph_dir / "component_max_displacement.png"
    plot_components(path, runs)
    graphs.append({"title": "Component max displacement", "path": path.relative_to(output_root).as_posix()})
    return graphs


def table_html(rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> str:
    if not rows:
        return "<p>No rows available.</p>"
    parts = ["<div class='table-wrap'><table><thead><tr>"]
    for _, label in cols:
        parts.append(f"<th>{html.escape(label)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for key, _ in cols:
            value = row.get(key, "")
            parts.append(f"<td>{html.escape(fmt(value) if isinstance(value, (int, float)) else str(value))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "\n".join(parts)


def build_tables(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_run = []
    identity_rows = []
    for run in runs:
        summary = run["summary"]
        per_run.append(
            {
                "face_id": run["face_id"],
                "prompt": run["prompt"],
                "final_Z": summary.get("final_Z"),
                "best_Z": summary.get("best_Z"),
                "best_iter_by_Z": summary.get("best_iter_by_Z"),
                "final_identity_cosine_similarity_raw": summary.get("final_identity_cosine_similarity_raw"),
                "final_identity_similarity_score_pct": summary.get("final_identity_similarity_score_pct"),
                "ssim_to_original": summary.get("final_ssim_to_original"),
                "psnr_to_original": summary.get("final_psnr_to_original"),
                "output_ssim": summary.get("best_output_ssim"),
                "output_l2": summary.get("best_output_l2"),
                "max_disp_px": summary.get("final_combined_max_disp_px"),
                "fraction_clamped": summary.get("final_fraction_clamped_total"),
                "seconds_per_iter": summary.get("mean_seconds_iter"),
                "run_dir": run["run_dir"],
            }
        )
        panel_path = Path(run["run_dir"]) / "identity_panel.csv"
        for row in read_csv_rows(panel_path):
            row = dict(row)
            row["face_id"] = run["face_id"]
            row["prompt"] = run["prompt"]
            identity_rows.append(row)
    return per_run, identity_rows


def build_html(data: dict[str, Any]) -> str:
    per_cols = [
        ("face_id", "face"),
        ("prompt", "prompt"),
        ("final_Z", "final Z"),
        ("best_Z", "best Z"),
        ("final_identity_cosine_similarity_raw", "ArcFace cosine sim"),
        ("final_identity_similarity_score_pct", "cosine score %"),
        ("ssim_to_original", "SSIM"),
        ("psnr_to_original", "PSNR"),
        ("output_ssim", "edit output SSIM"),
        ("max_disp_px", "max disp px"),
        ("fraction_clamped", "fraction clamped"),
    ]
    css = """
    body { margin:0; font-family: Inter, "Segoe UI", Arial, sans-serif; color:#17202a; background:white; }
    main { max-width:1180px; margin:0 auto; padding:34px 28px 70px; }
    h1 { font-size:34px; margin:0 0 6px; }
    h2 { margin-top:48px; padding-top:18px; border-top:2px solid #d7dde5; }
    h3 { margin-top:30px; }
    .subtitle,.small { color:#5d6d7e; }
    .card { border:1px solid #d7dde5; border-radius:12px; padding:18px; margin:20px 0; background:white; }
    table { border-collapse:collapse; width:100%; font-size:13px; margin:12px 0 22px; }
    th,td { border:1px solid #d7dde5; padding:7px 9px; vertical-align:top; }
    th { background:#f6f8fb; text-align:left; }
    .strip { width:100%; border:1px solid #d7dde5; border-radius:10px; display:block; }
    .graph-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(430px,1fr)); gap:18px; }
    figure { border:1px solid #d7dde5; border-radius:10px; padding:12px; margin:0; }
    figure img { width:100%; display:block; }
    figcaption { font-weight:650; margin-bottom:10px; }
    .path { font-family:Consolas, monospace; font-size:12px; word-break:break-all; }
    """
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>FACE report</title><style>",
        css,
        "</style></head><body><main>",
        f"<h1>{html.escape(TITLE)}</h1><p class='subtitle'>{html.escape(SUBTITLE)}</p><p class='small'>Author: {html.escape(AUTHOR)}</p>",
        "<div class='card'><p>FACE optimizes <code>Z = 1 - cosine_similarity</code> between original and perturbed full-image ArcFace iResNet-100 embeddings. The loss is exactly <code>loss = -Z</code>. ArcFace weights are frozen; only geometry parameters are optimized. No landmarks, face detection, alignment, visual counter-loss, VAE objective, or UNet objective is used.</p></div>",
        "<h2>1. Run matrix</h2><p>Four prompt-labeled cases are retained for compatibility. The ArcFace objective is not prompt-conditioned; prompts are used only for downstream InstructPix2Pix edit evaluation.</p>",
        table_html(data["per_run_rows"], per_cols),
        "<h2>2. Case image strips</h2>",
    ]
    for run in data["runs"]:
        parts.append(
            f"<div class='card'><h3>{html.escape(run['face_id'])} — {html.escape(run['prompt'])}</h3>"
            f"<img class='strip' src='{html.escape(run['strip_path'])}' alt='strip'>"
            f"<p class='path'>{html.escape(run['run_dir'])}</p></div>"
        )
    parts.append("<h2>3. Graphs</h2><div class='graph-grid'>")
    for graph in data["graphs"]:
        parts.append(f"<figure><figcaption>{html.escape(graph['title'])}</figcaption><a href='{html.escape(graph['path'])}'><img src='{html.escape(graph['path'])}'></a></figure>")
    parts.append("</div><h2>4. Notes</h2><ul>")
    parts.append(f"<li>Completed runs collected: {len(data['runs'])}.</li>")
    parts.append(f"<li>Run root: <span class='path'>{html.escape(data['run_root'])}</span></li>")
    parts.append(f"<li>Missing artifacts recorded: {len(data['missing'])}.</li>")
    parts.append("</ul></main></body></html>")
    return "\n".join(parts)


def build_markdown(data: dict[str, Any]) -> str:
    lines = [
        f"# {TITLE}",
        "",
        SUBTITLE,
        "",
        "FACE optimizes `Z = 1 - cosine_similarity` with `loss = -Z` against frozen ArcFace iResNet-100.",
        "",
        "## Image strips",
        "",
    ]
    for run in data["runs"]:
        lines.extend([f"### {run['face_id']} / {run['prompt']}", "", f"![strip]({run['strip_path']})", ""])
    lines.extend(["## Graphs", ""])
    for graph in data["graphs"]:
        lines.extend([f"### {graph['title']}", "", f"![{graph['title']}]({graph['path']})", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_root = resolve_run_root(args)
    output_root = Path(args.output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    runs, missing = collect_runs(run_root)
    for run in runs:
        run["strip_path"] = make_strip(run, output_root, args.compress_images)
    graphs = make_graphs(runs, output_root)
    per_run_rows, identity_rows = build_tables(runs)
    write_csv(output_root / "per_run_final_values.csv", per_run_rows)
    write_csv(output_root / "aggregate_summary.csv", per_run_rows)
    write_csv(output_root / "identity_panel_all_runs.csv", identity_rows)
    (output_root / "missing_artifacts.md").write_text(
        "# Missing artifacts\n\n" + ("\n".join(f"- {m['case']}: {m['artifact']} ({m['path']})" for m in missing) if missing else "None.\n"),
        encoding="utf-8",
    )
    (output_root / "image_index.md").write_text("\n".join(f"- {run['case']}: {run['strip_path']}" for run in runs) + "\n", encoding="utf-8")
    data = {
        "runs": runs,
        "run_root": str(run_root),
        "missing": missing,
        "graphs": graphs,
        "per_run_rows": per_run_rows,
    }
    (output_root / "report.html").write_text(build_html(data), encoding="utf-8")
    (output_root / "report.md").write_text(build_markdown(data), encoding="utf-8")
    (output_root / "report_data_summary.json").write_text(
        json.dumps(
            {
                "run_root": str(run_root),
                "num_runs": len(runs),
                "num_missing": len(missing),
                "graphs": graphs,
                "compress_images": bool(args.compress_images),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[face-report] run root: {run_root}")
    print(f"[face-report] wrote: {output_root / 'report.html'}")
    print(f"[face-report] wrote: {output_root / 'report.md'}")


if __name__ == "__main__":
    main()
