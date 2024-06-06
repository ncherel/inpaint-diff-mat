import os
import argparse

import torch
import numpy as np

from torch.utils.data import DataLoader

from torch.optim import Adam, lr_scheduler

from data import MaskDataset, SingleImageTrainDataset
from diffusion import Diffusion
from model import Model

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')
parser.add_argument("--image", required=True)
parser.add_argument("--steps", type=int, default=15000)

args = parser.parse_args()

writer = SummaryWriter(f"runs/{args.image}")

device = "cuda"
batch_size = 16

MACRO_FOLDER = "/data/Data_Deschaintre18/train_split"
train_dataset = SingleImageTrainDataset(os.path.join(MACRO_FOLDER, args.image))
dataloader = DataLoader(train_dataset, batch_size=batch_size)

mask_dataset = MaskDataset()

np.random.seed(42)
mask_dataloader = iter(DataLoader(mask_dataset, batch_size=batch_size))

diffusion = Diffusion()

model = Model(32)
model = model.to(device)

print(f"Number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

optimizer = Adam(model.parameters(), lr=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 10000, gamma=0.1)

training_iters = args.steps

for i, (images, test_masks) in zip(range(training_iters), dataloader):
    # Get the masks
    masks = next(mask_dataloader)

    images = images.to(device)

    # Copy test_masks to device and make sure that they have a single channel
    test_masks = test_masks.to(device)[:,:1]
    masks = masks.to(device)

    # Random timestep (one for all images)
    t = torch.randint(1, 1000, size=(images.shape[0],))

    images = (1 - test_masks) * images
    masks = 1 - (1 - masks) * (1 - test_masks)

    # Masked images
    x = images * (1 - masks)
    y = diffusion.forward(images, t)

    optimizer.zero_grad()

    # Apply model on x, y, t
    denoised = model(x, y, masks, t)

    # Denoising loss - simple loss without balancing
    loss = torch.nn.functional.mse_loss(masks * (1 - test_masks) * images, masks * (1 - test_masks) * denoised)
    loss.backward()

    optimizer.step()
    scheduler.step()

    if i % 50 == 0:
        writer.add_scalar("Loss/train", loss.item(), i)

    if i % 5000 == 0:
        torch.save(model.state_dict(), f"{writer.log_dir}/model_{i:04d}.pth")

torch.save(model.state_dict(), f"{writer.log_dir}/model_last.pth")
writer.close()
