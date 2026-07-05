# FACE: ArcFace White-box Geometric Identity Optimization

Frozen iResNet-100 identity-distance results with downstream InstructPix2Pix evaluation

FACE optimizes `Z = 1 - cosine_similarity` with `loss = -Z` against frozen ArcFace iResNet-100.

## Image strips

### face_002 / add black sunglasses

![strip](strips/face_face_002_add_black_sunglasses.jpg)

### face_002 / add headphones

![strip](strips/face_face_002_add_headphones.jpg)

### face_005 / add black sunglasses

![strip](strips/face_face_005_add_black_sunglasses.jpg)

### face_005 / add headphones

![strip](strips/face_face_005_add_headphones.jpg)

## Graphs

### Z vs iteration

![Z vs iteration](graphs/z_vs_iteration.png)

### Loss vs iteration

![Loss vs iteration](graphs/loss_vs_iteration.png)

### PSNR to original vs iteration

![PSNR to original vs iteration](graphs/psnr_vs_iteration.png)

### SSIM to original vs iteration

![SSIM to original vs iteration](graphs/ssim_vs_iteration.png)

### Geometric perturbation displacement vs iteration

![Geometric perturbation displacement vs iteration](graphs/geometric_perturbation_displacement_vs_iteration.png)

### Geometry component max displacement vs iteration

![Geometry component max displacement vs iteration](graphs/geometry_component_max_displacement_vs_iteration.png)
