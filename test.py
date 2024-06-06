import os
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data import SingleImageTestDataset
from diffusion import Diffusion
from model import Model


parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')
parser.add_argument("--image", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--n", default=1, type=int)

args = parser.parse_args()

folder = os.path.dirname(args.checkpoint)

device = "cuda"

MACRO_FOLDER = "/data/Data_Deschaintre18/train_split"
test_dataset = SingleImageTestDataset(os.path.join(MACRO_FOLDER, args.image))
test_dataloader = iter(DataLoader(test_dataset, batch_size=1))
test_images, test_masks = next(test_dataloader)

diffusion = Diffusion()

model = Model(32)
model.load_state_dict(torch.load(args.checkpoint))
model = model.to(device)

test_images = test_images.to(device)
test_masks = test_masks.to(device)[:,:1]

x = test_images * (1 - test_masks)

with torch.no_grad():
    t_end = 0
    for i in range(args.n):
        noise = torch.randn_like(test_images)
        y = noise

        for t in range(diffusion.T-1, t_end, -1):
            # Apply model on x, y, t
            y0 = model(x, y, test_masks, torch.tensor([t]))

            # Clipping during inference
            y0 = torch.clip(y0, -1.0, 1.0)

            y0 = test_masks * y0 + (1 - test_masks) * test_images

            # Update mean and variance
            mean = diffusion.betas[t] * torch.sqrt(diffusion.alphas_bar[t-1]) / (1 - diffusion.alphas_bar[t]) * y0 + (1 - diffusion.alphas_bar[t-1]) * torch.sqrt(diffusion.alphas[t]) / (1 - diffusion.alphas_bar[t]) * y
            var = diffusion.betas[t] * (1 - diffusion.alphas_bar[t-1]) / (1 - diffusion.alphas_bar[t])

            y = mean + torch.sqrt(var) * torch.randn_like(y0)

        for i, s in enumerate(["normal", "diffuse", "roughness", "specular"]):
            save_image((1+y0[:,3*i:3*(i+1)])/2, f"{folder}/{s}.png")
