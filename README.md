# DDPM 🎨
"Denoising Diffusion Probabilistic Models" paper implementation. 

---

## Overview
Denoising Diffusion Probabilistic Models (DDPM) are a class of generative models that learn a diffusion process to generate samples. The model iteratively applies a diffusion process to noise, gradually transforming it into samples from the target distribution. This approach has shown promising results in generating high-quality images and has garnered attention in the field of generative modeling.

---
### Gaussing Ditribution:
```math
\begin{aligned}
\mathbf{x}_t 
&= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} & \text{ ;where } \boldsymbol{\epsilon}_{t-1}, \boldsymbol{\epsilon}_{t-2}, \dots \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\boldsymbol{\epsilon}}_{t-2} & \text{ ;where } \bar{\boldsymbol{\epsilon}}_{t-2} \text{ merges two Gaussians (*).} \\
&= \dots \\
&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon} \\
q(\mathbf{x}_t \vert \mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
\end{aligned

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

