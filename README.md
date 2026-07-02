# FACE

FACE is a focused white-box geometric identity optimization experiment.

The white-box target is a frozen ArcFace iResNet-100 model. The complete RGB input image is resized directly to `112×112` inside PyTorch, normalized from `[0, 1]` to `[-1, 1]`, and passed through ArcFace to get an L2-normalized identity embedding.

No landmarks, face detection, face alignment, face mesh, crop boxes, or alignment matrices are used.

## Objective

For an original image `x`, differentiable geometric perturbation `T_theta`, perturbed image `x_p = T_theta(x)`, and frozen ArcFace model `F`:

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

## Geometry

FACE ports the latest WOOD differentiable geometry system, self-contained in this repo:

- TPS / Thin Plate Spline
- Delaunay / fixed-topology piecewise affine
- Rolling shutter
- DCT low-frequency warp
- FFT phase module

TPS, Delaunay, rolling shutter, and DCT produce spatial displacement fields. These fields are summed and applied through `grid_sample`. FFT phase, if enabled, is applied after the spatial warp as a differentiable frequency-domain stage.

All enabled geometry parameters are optimized jointly with Adam. After every optimizer step, parameters are projected/clamped into the JSON-configured ranges. Clamping is not permanent freezing: a parameter at a boundary can move back inside the range on a later Adam step.

Default geometry controls live in:

```text
configs/geometry_default.json
```

Each component has an `enabled` flag and manual range controls.

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
comparison_sheet.png
identity_panel.json
identity_panel.csv
```

Local replay/debug tensors:

```text
theta_final.pt
theta_best.pt
```

are ignored by Git.

## Smoke commands

Use the A6000 environment from MAT. See `instructions.md` for the exact command set.

Do not start the full 150-iteration matrix until checkpoint validation and smoke tests pass.
