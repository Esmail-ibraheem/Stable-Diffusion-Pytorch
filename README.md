# DDPM 🎨
"Denoising Diffusion Probabilistic Models" paper implementation. 



---
## Adding noise (Forward Process):

```latex
$$
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})
$$

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
