# FACE

FACE is a focused white-box ArcFace identity optimization experiment.

The white-box target is a frozen ArcFace iResNet-100 model. The complete RGB input image is resized directly to `112×112` inside PyTorch, normalized from `[0, 1]` to `[-1, 1]`, and passed through ArcFace to get an L2-normalized identity embedding.

No landmarks, face detection, face alignment, face mesh, crop boxes, or alignment matrices are used.

## Objective

For an original image `x`, differentiable perturbation pipeline `T_theta`, perturbed image `x_p = T_theta(x)`, and frozen ArcFace model `F`:

```text
e_original  = normalize(F(preprocess(x))).detach()
e_perturbed = normalize(F(preprocess(x_p)))

Z = 1 - cosine_similarity(e_original, e_perturbed)
loss = -Z
```

`Z` is ArcFace cosine identity distance. The only optimized loss is exactly:

```text
loss = -Z
```

There is no visual counter-loss, PSNR/SSIM penalty, displacement penalty, foldover penalty, VAE objective, UNet objective, CEM, SPSA, LoRA, model fine-tuning, pixel noise, or adversarial patch.

The display-only value:

```text
identity_similarity_score_pct = 100 × clamp(cosine_similarity, 0, 1)
```

is logged as `Cosine identity similarity score (%)`. It is not used in the loss and is not a literal identity-match percentage.

## Perturbation modules

FACE contains a joint differentiable perturbation system, self-contained in this repo:

- TPS / Thin Plate Spline
- Delaunay / fixed-topology piecewise affine
- Rolling shutter
- DCT-domain image frequency perturbation
- FFT phase module

TPS, Delaunay, and rolling shutter produce spatial displacement fields. These fields are summed and applied through `grid_sample`.

The active DCT module performs a true blockwise image DCT. It modifies selected AC coefficients and reconstructs the image with an inverse DCT. It is not a cosine-basis displacement field.

FFT phase, if enabled, is applied after the spatial warp and DCT image-frequency stage as a differentiable frequency-domain phase perturbation.

Because DCT now modifies image frequency coefficients directly, FACE is no longer a strictly geometry-only perturbation pipeline when DCT is enabled. It is a combined geometric and frequency-domain perturbation pipeline.

All enabled perturbation parameters are optimized jointly with Adam. After every optimizer step, parameters are projected/clamped into the JSON-configured ranges. Clamping is not permanent freezing: a parameter at a boundary can move back inside the range on a later Adam step.

Default geometry controls live in:

```text
configs/geometry_default.json
```

Each component has an `enabled` flag and manual range controls.

The default DCT settings use `8×8` blocks, `all_ac` frequency masking, DC exclusion, and `gain_limit = 0.5`. The effective selected coefficient multiplier is therefore `[0.5, 1.5]`.

## ArcFace checkpoint

FACE requires an actual pretrained ArcFace iResNet-100 checkpoint. It never uses random weights for real optimization.

Expected local path:

```text
models/arcface/iresnet100.pth
```

Validate it with:

```bash
python -m face.scripts.setup_arcface --checkpoint-path models/arcface/iresnet100.pth
```

Every run records:

- architecture
- checkpoint path
- checkpoint filename
- SHA256
- source
- input size
- embedding dimension
- preprocessing details
- parameter counts

Checkpoint files are ignored by Git and should not be committed.

## Cases

FACE keeps the same four prompt-labeled cases as GLASS/WOOD:

- `face_002` + `add black sunglasses`
- `face_002` + `add headphones`
- `face_005` + `add black sunglasses`
- `face_005` + `add headphones`

The ArcFace objective is not prompt-conditioned. The prompt only affects downstream InstructPix2Pix edit evaluation after identity optimization is complete.

## Downstream evaluation

InstructPix2Pix is downstream evaluation-only. It is not part of the optimization loss.

After optimization, FACE generates:

- original input + prompt → clean edited output
- perturbed best input + same prompt → perturbed edited output
- perturbed final input + same prompt → perturbed final edited output

DeepFace is also evaluation-only. It is never used in the gradient loop.

## Outputs

Each run saves:

```text
history.csv
history.jsonl
config_resolved.json
summary.json
DONE.json or FAILED.json
checkpoints/iter_000 ... iter_150
original.png
perturbed_best.png
perturbed_final.png
input_difference_best.png
input_difference_final.png
combined_flow_best.png
combined_flow_final.png
dct_only_perturbed.png
dct_only_difference.png
dct_only_difference_x10.png
dct_gain_heatmap.png
dct_frequency_mask.png
dct_spectrum_before.png
dct_spectrum_after.png
dct_spectrum_difference.png
comparison_sheet.png
identity_panel.json
identity_panel.csv
```

`combined_flow*.png` visualizes only TPS + Delaunay + rolling displacement. DCT is not included in the flow image because it is not a coordinate displacement field.

Local replay/debug tensors:

```text
theta_final.pt
theta_best.pt
```

are ignored by Git.

## Smoke commands

Use the A6000 environment from MAT. See `instructions.md` for the exact command set.

Do not start the full 150-iteration matrix until checkpoint validation and smoke tests pass.

## Image-DCT smoke test

The standalone DCT smoke test does not load ArcFace or InstructPix2Pix:

```bash
python -m face.scripts.smoke_dct_image \
  --image /home/interns/Desktop/mat/data/face_002/instruct_512.png \
  --output-root outputs/dct_image_smoke \
  --block-size 8 \
  --frequency-mask all_ac \
  --gain-limit 0.5
```

It checks neutral reconstruction, nonzero DCT effect, gradient flow into `dct_gain_raw`, disabled behavior, projection, and DC preservation.
