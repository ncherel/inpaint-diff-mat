import torch
import numpy as np


class Diffusion():
    def __init__(self):
        self.T = 1000
        self.betas = np.linspace(1e-4, 0.02, self.T, dtype=np.float32)
        self.alphas = 1 - self.betas
        self.alphas_bar = np.cumprod(self.alphas)

        self.betas = torch.from_numpy(self.betas).to("cuda")
        self.alphas = torch.from_numpy(self.alphas).to("cuda")
        self.alphas_bar = torch.from_numpy(self.alphas_bar).to("cuda")

    def forward(self, x, T, return_noise=False):
        noise = torch.randn_like(x)
        noisy = x * torch.sqrt(self.alphas_bar[T])[:,None,None,None] + torch.sqrt(1 - self.alphas_bar[T])[:,None,None,None] * noise
        if return_noise:
            return noisy, noise
        return noisy
