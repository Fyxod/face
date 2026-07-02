# FACE A6000 instructions

Use these on the A6000 Linux machine.

## 0. Enter repo and pull

```bash
cd /home/interns/Desktop/face
git pull origin main
```

## 1. Optional dependency check/install

```bash
$HOME/.local/bin/micromamba run \
  -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  bash scripts/install_linux_a6000.sh
```

## 2. Put ArcFace iResNet-100 checkpoint in place

Expected file:

```text
/home/interns/Desktop/face/models/arcface/iresnet100.pth
```

Do not commit this file.

## 3. Validate ArcFace checkpoint

```bash
$HOME/.local/bin/micromamba run \
  -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m face.scripts.setup_arcface \
  --checkpoint-path /home/interns/Desktop/face/models/arcface/iresnet100.pth
```

Expected report:

```text
outputs/smoke/arcface_setup_report.json
```

If this fails, do not run optimization. Fix the checkpoint first.

## 4. Identity smoke

```bash
$HOME/.local/bin/micromamba run \
  -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m face.scripts.smoke_identity \
  --mat-root /home/interns/Desktop/mat \
  --arcface-checkpoint /home/interns/Desktop/face/models/arcface/iresnet100.pth \
  --geometry-config configs/geometry_default.json
```

Expected root:

```text
outputs/smoke/
```

## 5. Quick timing smoke

```bash
$HOME/.local/bin/micromamba run \
  -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m face.scripts.smoke_timing \
  --mat-root /home/interns/Desktop/mat \
  --arcface-checkpoint /home/interns/Desktop/face/models/arcface/iresnet100.pth \
  --iters 2 \
  --quick \
  --geometry-config configs/geometry_default.json
```

Expected root:

```text
outputs/smoke_timing/
```

## 6. All-case timing smoke

```bash
$HOME/.local/bin/micromamba run \
  -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m face.scripts.smoke_timing \
  --mat-root /home/interns/Desktop/mat \
  --arcface-checkpoint /home/interns/Desktop/face/models/arcface/iresnet100.pth \
  --iters 2 \
  --all-cases \
  --geometry-config configs/geometry_default.json
```

## 7. Full 150-iteration run

Run only after checkpoint validation and smoke pass:

```bash
cd /home/interns/Desktop/face
mkdir -p logs

$HOME/.local/bin/micromamba run \
  -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m face.scripts.run_matrix \
  --mat-root /home/interns/Desktop/mat \
  --arcface-checkpoint /home/interns/Desktop/face/models/arcface/iresnet100.pth \
  --iters 150 \
  --output-root outputs/arcface_identity \
  --geometry-config configs/geometry_default.json \
  2>&1 | tee logs/face_arcface_identity_150.log
```

Completed runs are skipped if `DONE.json` exists. Use `--force` only if you intentionally want to overwrite.

## 8. Summarize

```bash
$HOME/.local/bin/micromamba run \
  -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m face.scripts.summarize_runs \
  --results-root outputs/arcface_identity \
  --output-root outputs/reports/arcface_identity
```

## 9. Build report with compressed images

```bash
$HOME/.local/bin/micromamba run \
  -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m face.scripts.build_report \
  --results-root outputs/arcface_identity \
  --output-root outputs/reports/arcface_identity \
  --compress-images
```

Expected report outputs:

```text
outputs/reports/arcface_identity/
  aggregate_summary.csv
  per_run_final_values.csv
  identity_panel_all_runs.csv
  report.md
  report.html
  report_data_summary.json
  image_index.md
  missing_artifacts.md
  graphs/
  strips/
```

## 10. Push after run

Do not add checkpoint weights or theta files.

```bash
git status -sb
git add face configs scripts README.md instructions.md requirements.txt pyproject.toml .gitignore
git add outputs/smoke outputs/smoke_timing outputs/arcface_identity outputs/reports logs/face_arcface_identity_150.log
git commit -m "Add FACE ArcFace identity results"
git push origin main
```

If Git shows `models/arcface/iresnet100.pth`, `theta_final.pt`, or `theta_best.pt`, do not add them.
