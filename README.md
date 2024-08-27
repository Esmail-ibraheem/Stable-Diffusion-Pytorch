<h1 align="center">DDPM 🎨</h1>

![image](https://github.com/Esmail-ibraheem/DDPM/assets/113830751/55ba9e44-5096-4531-9629-061208177653)

🎨"Denoising Diffusion Probabilistic Models" paper implementation. 

---

## Overview
[Diffusion Models](https://huggingface.co/blog/Esmail-AGumaan/diffusion-models#diffusion-models) are **generative** models, meaning that they are used to generate data similar to the data on which they are trained. Fundamentally, Diffusion Models work by **destroying training data** through the successive addition of Gaussian noise, and then **learning to recover** the data by _reversing_ this noising process. After training, we can use the Diffusion Model to generate data by simply **passing randomly sampled noise through the learned denoising process.**

Diffusion models are inspired by **non-equilibrium thermodynamics**. They define a **Markov chain** of diffusion steps to slowly add random noise to data and then learn to reverse the diffusion process to construct desired data samples from the noise. Unlike **VAE or flow models**, diffusion models are learned with a fixed procedure and the latent variable has high dimensionality (same as the original data).

---

diffusion models consists of two processes as shown in the image below:
- Forward process (with red lines).
- Reverse process (with blue lines).
![image](https://github.com/Esmail-ibraheem/DDPM/assets/113830751/55ba9e44-5096-4531-9629-061208177653)
As mentioned above, a Diffusion Model consists of a **forward process** (or **diffusion process**), in which a datum (generally an image) is progressively noised, and a **reverse process** (or **reverse diffusion process**), in which noise is transformed back into a sample from the target distribution.

In a bit more detail for images, the set-up consists of 2 processes:

- a fixed (or predefined) forward diffusion process q of our choosing, that gradually adds Gaussian noise to an image, until you end up with pure noise
- a learned reverse denoising diffusion process p_θ​, where a neural network is trained to gradually denoise an image starting from pure noise until you end up with an actual image.

## 1. Forward Process (Fixed):

The sampling chain transitions in the forward process can be set to conditional Gaussians when the noise level is sufficiently low. Combining this fact with the Markov assumption leads to a simple parameterization of the forward process:

![Pasted image 20240317140123](https://github.com/Esmail-ibraheem/DDPM/assets/113830751/e9638ab8-2172-45be-b46d-90a67e35d425)


## 2. Reverse Process (Learned)

Ultimately, the image is asymptotically transformed to pure Gaussian noise. The **goal** of training a diffusion model is to learn the **reverse** process - i.e. training. By traversing backwards along this chain, we can generate new data.

![](https://www.assemblyai.com/blog/content/images/2022/05/image-1.png)

where the time-dependent parameters of the Gaussian transitions are learned. Note in particular that the Markov formulation asserts that a given reverse diffusion transition distribution depends only on the previous timestep (or following timestep, depending on how you look at it).

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

@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{https://doi.org/10.48550/arxiv.2204.11824,
  doi = {10.48550/ARXIV.2204.11824},
  url = {https://arxiv.org/abs/2204.11824},
  author = {Blattmann, Andreas and Rombach, Robin and Oktay, Kaan and Ommer, Björn},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Retrieval-Augmented Diffusion Models},
  publisher = {arXiv},
  year = {2022},  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```

---

## References
[original_paper](https://arxiv.org/abs/2006.11239): "Denoising Diffusion Probabilistic Models" by Jonathan Ho, Ajay Jain, and Pieter Abbeel.

---

