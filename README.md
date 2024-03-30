# DDPM 🎨
"Denoising Diffusion Probabilistic Models" paper implementation. 

---

## Overview
Denoising Diffusion Probabilistic Models (DDPM) are a class of generative models that learn a diffusion process to generate samples. The model iteratively applies a diffusion process to noise, gradually transforming it into samples from the target distribution. This approach has shown promising results in generating high-quality images and has garnered attention in the field of generative modeling.

---
### Gaussing Ditribution:
![image](https://github.com/Esmail-ibraheem/DDPM/assets/113830751/720b733e-3fc3-4419-a36a-b7aa0cfbb400)

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

