# DDPM 🎨
![image](https://github.com/Esmail-ibraheem/DDPM/assets/113830751/55ba9e44-5096-4531-9629-061208177653)

"Denoising Diffusion Probabilistic Models" paper implementation. 

---

## Overview
Denoising Diffusion Probabilistic Models (DDPM) are a class of generative models that learn a diffusion process to generate samples. The model iteratively applies a diffusion process to noise, gradually transforming it into samples from the target distribution. This approach has shown promising results in generating high-quality images and has garnered attention in the field of generative modeling.

---
### Gaussing Ditribution:

```math
q(\mathbf{x}_t \vert \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
```

```python
def add_noise(self, 
                 original_samples: torch.FloatTensor, 
                 timestep: torch.IntTensor):

        alphas_cumlative_product = self.alphas_cumlative_product.to(device = original_samples.device, dtype = original_samples.dtype)
        timestep = timestep.to(original_samples.device)
        alphas_cumlative_product_squaroot = alphas_cumlative_product[timestep] ** 0.5 
        alphas_cumlative_product_squaroot = alphas_cumlative_product_squaroot.flatten()
        while len(alphas_cumlative_product_squaroot.shape) < len(original_samples.shape):
            alphas_cumlative_product_squaroot = alphas_cumlative_product_squaroot.unsqueeze(-1)
        
        alphas_cumlative_product_squaroot_mins_one = (1 - alphas_cumlative_product[timestep]) ** 0.5 
        alphas_cumlative_product_squaroot_mins_one = alphas_cumlative_product_squaroot_mins_one.flatten()
        while len(alphas_cumlative_product_squaroot_mins_one.shape) < len(original_samples.shape):
            alphas_cumlative_product_squaroot_mins_one = alphas_cumlative_product_squaroot_mins_one.unsqueeze(-1)
        
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = alphas_cumlative_product_squaroot * original_samples + alphas_cumlative_product_squaroot_mins_one * noise 
        return noisy_samples
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

