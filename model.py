import torch
import torch.nn as nn
from torch.nn.functional import interpolate, max_pool2d, unfold, conv2d

import math

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:,None] * embeddings[None]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class UNet(nn.Module):
    def __init__(self, n_levels=2, n_channels=32):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, 3, padding=1),
            nn.ReLU()
        )

        if n_levels > 1:
            self.inner = UNet(n_levels-1, n_channels)
            self.att = None
        else:
            self.inner = nn.Sequential(
                nn.Conv2d(n_channels, n_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(n_channels, n_channels, 3, padding=1),
                nn.ReLU()
            )
            self.att = None

        self.post = nn.Sequential(
            nn.Conv2d(2 * n_channels, n_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.pre(x)
        x = max_pool2d(x1, 2)
        x = self.inner(x)
        if self.att is not None:
            x = self.att(x, x, x)
        x = interpolate(x, scale_factor=2)
        x = self.post(torch.cat([x, x1], dim=1))
        return x


class Model(nn.Module):
    def __init__(self, n_channels=32):
        self.time_dim = 16
        self.d = n_channels
        self.output_dim = 12

        super().__init__()

        # Time embedding
        self.time_encoder = TimeEmbedding(self.time_dim)

        # Feature extractor
        self.pre_features = nn.Sequential(
            # nn.Conv2d(3 + self.time_dim, self.d, 3, padding=1),
            nn.Conv2d(4*3*2 + 1 + self.time_dim, self.d, 3, padding=1),
            nn.ReLU()
        )

        self.time_encoder2 = nn.Sequential(nn.Linear(self.time_dim, self.time_dim), nn.GELU(), nn.Linear(self.time_dim, self.time_dim))

        self.conv_features = UNet(n_levels=3, n_channels=self.d)

        self.post_features = nn.Sequential(
            nn.Conv2d(self.d, self.output_dim, 3, padding=1)
        )

    def forward(self, x, y, mask, t):
        # Encode t and reshape it to image size
        t_embedding = self.time_encoder(t.to("cuda"))
        t_embedding = self.time_encoder2(t_embedding)
        t_embedding = t_embedding.reshape(x.shape[0], -1, 1, 1)
        t_embedding = t_embedding.repeat(1, 1, x.shape[2], x.shape[3])

        a = torch.cat([x, y, mask, t_embedding], dim=1)

        a = self.pre_features(a)
        a = self.conv_features(a)
        a = self.post_features(a)

        return a
