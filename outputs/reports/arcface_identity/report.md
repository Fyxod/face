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

### ArcFace cosine similarity vs iteration

![ArcFace cosine similarity vs iteration](graphs/cosine_similarity_vs_iteration.png)

### ArcFace cosine distance vs iteration

![ArcFace cosine distance vs iteration](graphs/cosine_distance_vs_iteration.png)

### Cosine identity similarity score (%) vs iteration

![Cosine identity similarity score (%) vs iteration](graphs/similarity_score_pct_vs_iteration.png)

### PSNR to original vs iteration

![PSNR to original vs iteration](graphs/psnr_vs_iteration.png)

### SSIM to original vs iteration

![SSIM to original vs iteration](graphs/ssim_vs_iteration.png)

### Combined max displacement vs iteration

![Combined max displacement vs iteration](graphs/combined_max_disp_vs_iteration.png)

### Fraction clamped vs iteration

![Fraction clamped vs iteration](graphs/fraction_clamped_vs_iteration.png)

### Final Z vs final SSIM

![Final Z vs final SSIM](graphs/final_z_vs_ssim.png)

### Final Z vs final PSNR

![Final Z vs final PSNR](graphs/final_z_vs_psnr.png)

### Component max displacement

![Component max displacement](graphs/component_max_displacement.png)
