# FACE: ArcFace White-box Spatial + Frequency Identity Optimization

Frozen iResNet-100 identity-distance results with image-DCT perturbation and downstream InstructPix2Pix evaluation

FACE optimizes `Z = 1 - cosine_similarity` with `loss = -Z` against frozen ArcFace iResNet-100. DCT is reported as an image-frequency coefficient perturbation, not a spatial flow.

## Image strips

### face_002 / add black sunglasses

![strip](strips/face_face_002_add_black_sunglasses.png)

### face_002 / add headphones

![strip](strips/face_face_002_add_headphones.png)

### face_005 / add black sunglasses

![strip](strips/face_face_005_add_black_sunglasses.png)

### face_005 / add headphones

![strip](strips/face_face_005_add_headphones.png)

## Graphs

### Z vs iteration

![Z vs iteration](graphs/z_vs_iteration.png)

### Loss vs iteration

![Loss vs iteration](graphs/loss_vs_iteration.png)

### PSNR to original vs iteration

![PSNR to original vs iteration](graphs/psnr_vs_iteration.png)

### SSIM to original vs iteration

![SSIM to original vs iteration](graphs/ssim_vs_iteration.png)

### Geometry component diagnostics vs iteration

![Geometry component diagnostics vs iteration](graphs/geometry_component_diagnostics_vs_iteration.png)

### DCT gain mean-absolute value vs iteration

![DCT gain mean-absolute value vs iteration](graphs/dct_gain_mean_abs_vs_iteration.png)

### DCT coefficient-energy change vs iteration

![DCT coefficient-energy change vs iteration](graphs/dct_relative_energy_change_vs_iteration.png)

### DCT spatial-delta MSE vs iteration

![DCT spatial-delta MSE vs iteration](graphs/dct_spatial_delta_mse_vs_iteration.png)

### DCT low/mid/high frequency energy vs iteration

![DCT low/mid/high frequency energy vs iteration](graphs/dct_frequency_band_energy_vs_iteration.png)

### Final Z vs DCT coefficient-energy change

![Final Z vs DCT coefficient-energy change](graphs/z_vs_dct_relative_energy_change.png)
