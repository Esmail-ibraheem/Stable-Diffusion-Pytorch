# DDPM 🎨
"Denoising Diffusion Probabilistic Models" paper implementation. 

---

## Overview
Denoising Diffusion Probabilistic Models (DDPM) are a class of generative models that learn a diffusion process to generate samples. The model iteratively applies a diffusion process to noise, gradually transforming it into samples from the target distribution. This approach has shown promising results in generating high-quality images and has garnered attention in the field of generative modeling.

---
### Gaussing Ditribution:

```math
q(\mathbf{x}_t \vert \mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
```

```math
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})
```

```python
class GaussingDitribution:
    def __init__(self, paramenters: torch.Tensor) -> None:
        self.mean, log_variance = torch.chunk(paramenters, 2, dim = 1)
        self.log_variance = torch.clamp(log_variance, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.log_variance)
    
    def sample(self):
        return self.mean + self.std * torch.rand_like(self.std)
```
---

## Citation
```BibTex
@misc{ho2020denoising,
    title   = {Denoising Diffusion Probabilistic Models},
    author  = {Jonathan Ho and Ajay Jain and Pieter Abbeel},
    year    = {2020},
    eprint  = {2006.11239},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

---

## References
[original_paper](https://arxiv.org/abs/2006.11239): "Denoising Diffusion Probabilistic Models" by Jonathan Ho, Ajay Jain, and Pieter Abbeel.

---

