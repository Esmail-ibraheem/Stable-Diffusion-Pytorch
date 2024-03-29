import torch 
import torch.nn as nn 
from torch.nn import functional as F 
import numpy as np
import math 
 
class DenoisingDiffusionProbabilisticModelSampler:
    def __init__(self, generator: torch.Generator, number_training_steps = 1000, beta_start: float = 0.00085, beta_end: float = 0.0120) -> None:
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, number_training_steps, dtype=torch.float32) ** 2 
        self.alphas = 1.0 - self.betas
        self.alphas_cumlativeproduct = torch.cumprod(self.alphas, d_model = 0)
        self.one = torch.tensor(1.0)
        self.generator = generator
        self.number_training_timesteps = number_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, number_training_steps)[::-1].copy())
    
    def set_inference_steps(self, number_inference_steps = 50):
        self.number_inference_steps = number_inference_steps
        ratio = self.number_training_timesteps // self.number_inference_steps
        timesteps = (np.arange(0, number_inference_steps) * ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
    
    def _get_previous_timesteps(self, timestep: int) -> int:
        previous_step = timestep - self.number_training_timesteps // self.number_inference_steps
        return previous_step
    
    def _get_variance(self, timestep: int) -> torch.Tensor:
        previous_step = self._get_previous_timesteps(timestep)
        alpha_product_t = self.alphas_cumlativeproduct[timestep]
        alpha_product_t_previous = self.alphas_cumlativeproduct[previous_step] if previous_step >= 0 else self.one 
        current_beta_t = 1 - alpha_product_t / alpha_product_t_previous
        variance = (1 - alpha_product_t_previous) / (1 - alpha_product_t) * current_beta_t
        variance = torch.clamp(variance, 1e-20)
        return variance 
    
    def set_strength(self, strength = 1):
        """
            Set how much noise to add to the input image. 
            More noise (strength ~ 1) means that the output will be further from the input image.
            Less noise (strength ~ 0) means that the output will be closer to the input image.
        """
        startstep = self.number_inference_steps - int(self.number_inference_steps * strength)
        self.timesteps = self.timesteps[startstep:]
        self.startstep = startstep
    
    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        t = timestep
        previous_step = self._get_previous_timesteps(t)
        alpha_product_t = self.alphas_cumlativeproduct[t]
        alpha_product_t_previous = self.alphas_cumlativeproduct[previous_step] if previous_step >= 0 else self.one
        current_alpha_product_t = alpha_product_t / alpha_product_t_previous
        beta_product_t = 1 - alpha_product_t
        beta_product_t_previous = 1 - alpha_product_t_previous
        current_beta_product_t = 1 - current_alpha_product_t
        predict_original_sample = (latents - beta_product_t ** (0.5) * model_output) / alpha_product_t
        predict_original_sample_coeff = (alpha_product_t_previous ** (0.5) * current_beta_product_t) / beta_product_t
        current_sample_coeff = current_alpha_product_t ** (0.5) * beta_product_t_previous / beta_product_t
        predict_previous_sample = predict_original_sample_coeff * predict_original_sample + current_sample_coeff * latents
        variance = 0 
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            variance = (self._get_variance ** 0.5) * noise 
        predict_previous_sample = predict_previous_sample + variance
        return predict_previous_sample
     
