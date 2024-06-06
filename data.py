import os
import math
from os.path import join

import torch
import numpy as np

from PIL import Image, ImageDraw
from torch.utils.data import IterableDataset, Dataset
from torchvision.transforms import ToTensor, Normalize, Compose


def get_mean_std(pil_image):
    np_img = np.array(pil_image) / 255.0
    std = np.std(np_img, axis=(0,1))
    return np.mean(np_img, axis=(0,1)), std if np.linalg.norm(std) > 1e-4 else np.ones(std.shape)


class SingleImageTrainDataset(IterableDataset):
    def __init__(self, image_path):
        super().__init__()
        self.img = Image.open(join(image_path, "diffuse.png")).convert("RGB")
        self.normal = Image.open(join(image_path, "normal.png")).convert("RGB")
        self.roughness = Image.open(join(image_path, "roughness.png")).convert("RGB")
        self.specular = Image.open(join(image_path, "specular.png")).convert("RGB")
        self.mask = Image.open(join(image_path, "mask.png"))

        self.transforms_mean_std = {
            "normal":  Compose([ToTensor(), Normalize(*get_mean_std(self.normal))]),
            "diffuse":  Compose([ToTensor(), Normalize(*get_mean_std(self.img))]),
            "roughness":  Compose([ToTensor(), Normalize(*get_mean_std(self.roughness))]),
            "specular":  Compose([ToTensor(), Normalize(*get_mean_std(self.specular))])
        }

        self.transforms = Compose([
            ToTensor(),
            Normalize(0.5, 0.5)  # Change range [0,1] -> [-1, 1]
        ])
        self.mask_transform = Compose([
            ToTensor()
        ])

    def __iter__(self):
        def next_crop():
            # Draw new coordinates for our image until not overlapping the center crop (256x256)
            w, h = self.img.size

            while True:
                x, y = np.random.randint(0, w - 256), np.random.randint(0, h - 256)
                box = (x, y, x + 256, y + 256)

                yield torch.cat([self.transforms_mean_std["normal"](self.normal.crop(box)),
                                 self.transforms_mean_std["diffuse"](self.img.crop(box)),
                                 self.transforms_mean_std["roughness"](self.roughness.crop(box)),
                                 self.transforms_mean_std["specular"](self.specular.crop(box))], dim=0), self.mask_transform(self.mask.crop(box))




        return iter(next_crop())


class SingleImageTestDataset(Dataset):
    def __init__(self, image_path):
        super().__init__()
        self.img = Image.open(join(image_path, "diffuse.png")).convert("RGB")
        self.normal = Image.open(join(image_path, "normal.png")).convert("RGB")
        self.roughness = Image.open(join(image_path, "roughness.png")).convert("RGB")
        self.specular = Image.open(join(image_path, "specular.png")).convert("RGB")
        self.mask = Image.open(join(image_path, "mask.png"))
        self.transforms = Compose([
            ToTensor(),
            Normalize(0.5, 0.5)  # Change range [0,1] -> [-1, 1]
        ])
        self.mask_transform = Compose([
            ToTensor()
        ])

        self.transforms_mean_std = {
            "normal":  Compose([ToTensor(), Normalize(*get_mean_std(self.normal))]),
            "diffuse":  Compose([ToTensor(), Normalize(*get_mean_std(self.img))]),
            "roughness":  Compose([ToTensor(), Normalize(*get_mean_std(self.roughness))]),
            "specular":  Compose([ToTensor(), Normalize(*get_mean_std(self.specular))])
        }

        print(self.transforms_mean_std)


    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return  torch.cat([self.transforms_mean_std["normal"](self.normal),
                           self.transforms_mean_std["diffuse"](self.img),
                           self.transforms_mean_std["roughness"](self.roughness),
                           self.transforms_mean_std["specular"](self.specular)], dim=0), self.mask_transform(self.mask)


class MaskDataset(IterableDataset):
    def __init__(self):
        super().__init__()

    def __iter__(self):
        return iter(mask_generator(256, 256))


def mask_generator(H, W):
    min_num_vertex = 4
    max_num_vertex = 12
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 15
    min_width = 12
    max_width = 40

    while True:
        # Code from ContextualAttention
        average_radius = math.sqrt(H*H+W*W) / 8
        mask = Image.new('L', (W, H), 0)

        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                              v[1] - width//2,
                              v[0] + width//2,
                              v[1] + width//2),
                             fill=1)

        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.float32)
        mask = np.reshape(mask, (1, H, W))
        yield mask
