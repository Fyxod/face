# FACE A6000 instructions

These are the commands to run on the A6000 Linux machine.

Important fix: do **not** use the old `huggingface-cli download` command and do **not** use `--local-dir-use-symlinks`. Your log showed that `huggingface-cli` is deprecated and the new `hf download` no longer accepts that option.

Use the downloader script below. It tries direct `curl` first, falls back to the Python `huggingface_hub` API if needed, checks SHA256, and then runs FACE checkpoint validation.

Checkpoint source used by the script:

```text
https://huggingface.co/camenduru/show/blob/064a379f415f674051145ec4862f54bd6a65073f/models/arcface/ms1mv3_arcface_r100_fp16.pth
```

Expected SHA256:

```text
a566a62357f0c55b679d9ff2f022a294486568be0c00665d39029d0e46a8109b
```

Final local checkpoint path:

```text
/home/interns/Desktop/face/models/arcface/iresnet100.pth
```

Do not commit this checkpoint.

## 0. Enter repo and pull

```bash
cd /home/interns/Desktop/face
git pull origin main
```

## 1. Download and validate ArcFace checkpoint

Run this exactly:

```bash
cd /home/interns/Desktop/face
bash scripts/download_arcface_checkpoint_a6000.sh /home/interns/Desktop/face
```

Expected success signs:

```text
[face-ckpt] existing checkpoint is valid.
```

or:

```text
[face-ckpt] curl download ok.
[face-ckpt] final SHA:
a566a62357f0c55b679d9ff2f022a294486568be0c00665d39029d0e46a8109b  /home/interns/Desktop/face/models/arcface/iresnet100.pth
[face-setup] checkpoint ok: /home/interns/Desktop/face/models/arcface/iresnet100.pth
```

If this step fails, stop. Do not run optimization.

The setup report is written here:

```text
outputs/smoke/arcface_setup_report.json
```

## 2. Standalone image-DCT smoke

Run this before ArcFace. It does not load ArcFace or InstructPix2Pix.

```bash
cd /home/interns/Desktop/face

$HOME/.local/bin/micromamba run \
  -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m face.scripts.smoke_dct_image \
  --image /home/interns/Desktop/mat/data/face_002/instruct_512.png \
  --output-root outputs/dct_image_smoke \
  --block-size 8 \
  --frequency-mask all_ac \
  --gain-limit 0.5
```

Expected root:

```text
outputs/dct_image_smoke/
```

This checks neutral reconstruction, nonzero image-DCT effect, gradient flow into `dct_gain_raw`, disabled behavior, projection, and DC preservation.

## 3. Identity smoke

Run after checkpoint validation succeeds:

```bash
cd /home/interns/Desktop/face

$HOME/.local/bin/micromamba run \
  -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m face.scripts.smoke_identity \
  --mat-root /home/interns/Desktop/mat \
  --arcface-checkpoint /home/interns/Desktop/face/models/arcface/iresnet100.pth \
  --geometry-config configs/geometry_default.json \
  --output-root outputs/smoke_dct_image_pipeline
```

Expected root:

```text
outputs/smoke_dct_image_pipeline/
```

This checks:

- ArcFace checkpoint loads as iResNet-100
- ArcFace parameters are frozen
- full 512×512 RGB image resizes differentiably to 112×112
- embedding is finite
- same-image cosine similarity is near 1
- `Z = 1 - cosine_similarity` is finite
- `loss = -Z` is finite
- gradients reach geometry parameters
- gradients reach `dct_gain_raw`
- ArcFace receives no optimizer updates
- hard projection executes
- DCT spectrum changes
- old DCT displacement fields are not reported
- PSNR and SSIM are logged
- history/images are saved

## 4. Quick timing smoke

```bash
cd /home/interns/Desktop/face

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

## 5. All-case timing smoke

```bash
cd /home/interns/Desktop/face

$HOME/.local/bin/micromamba run \
  -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m face.scripts.smoke_timing \
  --mat-root /home/interns/Desktop/mat \
  --arcface-checkpoint /home/interns/Desktop/face/models/arcface/iresnet100.pth \
  --iters 2 \
  --all-cases \
  --geometry-config configs/geometry_default.json
```

## 6. Full 150-iteration run

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
  --output-root outputs/arcface_identity_dct_image \
  --geometry-config configs/geometry_default.json \
  2>&1 | tee logs/face_arcface_identity_dct_image_150.log
```

Completed runs are skipped if `DONE.json` exists. Use `--force` only if you intentionally want to overwrite.

## 7. Summarize

```bash
cd /home/interns/Desktop/face

$HOME/.local/bin/micromamba run \
  -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m face.scripts.summarize_runs \
  --results-root outputs/arcface_identity_dct_image \
  --output-root outputs/reports/arcface_identity_dct_image
```

## 8. Build report

```bash
cd /home/interns/Desktop/face

$HOME/.local/bin/micromamba run \
  -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m face.scripts.build_report \
  --results-root outputs/arcface_identity_dct_image \
  --output-root outputs/reports/arcface_identity_dct_image
```

Expected report outputs:

```text
outputs/reports/arcface_identity_dct_image/
  aggregate_summary.csv
  per_run_final_values.csv
  identity_panel_all_runs.csv
  report.md
  report.html
  report.pdf
  report_data_summary.json
  image_index.md
  missing_artifacts.md
  graphs/
  strips/
```

## 9. Push after run

Do not add checkpoint weights or theta files.

```bash
cd /home/interns/Desktop/face
git status -sb

git add face configs scripts README.md instructions.md requirements.txt pyproject.toml .gitignore
git add outputs/dct_image_smoke outputs/smoke_dct_image_pipeline outputs/smoke_timing outputs/arcface_identity_dct_image outputs/reports logs/face_arcface_identity_dct_image_150.log logs.txt

git commit -m "Add FACE ArcFace identity results" || true
git push origin main
```

Before committing, check that none of these are staged:

```text
models/arcface/iresnet100.pth
theta_final.pt
theta_best.pt
*.pth
*.pt
```

If any of those appear in `git status`, do not force-add them.

## Manual fallback only if the script fails

If `scripts/download_arcface_checkpoint_a6000.sh` fails because `curl` and `huggingface_hub` both fail, try this manual command. It intentionally does not use `huggingface-cli` and does not use `--local-dir-use-symlinks`.

```bash
cd /home/interns/Desktop/face
mkdir -p models/arcface

curl -L --fail --retry 5 --retry-delay 5 \
  -o models/arcface/iresnet100.pth \
  "https://huggingface.co/camenduru/show/resolve/064a379f415f674051145ec4862f54bd6a65073f/models/arcface/ms1mv3_arcface_r100_fp16.pth?download=true"

sha256sum models/arcface/iresnet100.pth

$HOME/.local/bin/micromamba run \
  -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m face.scripts.setup_arcface \
  --checkpoint-path /home/interns/Desktop/face/models/arcface/iresnet100.pth \
  --source-name "camenduru/show models/arcface/ms1mv3_arcface_r100_fp16.pth revision 064a379f415f674051145ec4862f54bd6a65073f"
```

The SHA printed by `sha256sum` must be:

```text
a566a62357f0c55b679d9ff2f022a294486568be0c00665d39029d0e46a8109b
```
