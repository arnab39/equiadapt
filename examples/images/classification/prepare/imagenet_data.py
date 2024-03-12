import random

import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import ImageOps
from torch import nn

DEFAULT_CROP_RATIO = 224 / 256


class GaussianBlur(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            gblur = transforms.GaussianBlur(5, sigma=sigma)
            return gblur(img)
        else:
            return img


class Solarization(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class CustomRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)


class Transform:
    def __init__(self, mode="train"):
        # these transformations are essential for reproducing the zero-shot performance
        if mode == "train":
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        224,
                        interpolation=transforms.functional.InterpolationMode.BILINEAR,
                    ),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        elif mode == "val":
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        256,
                        interpolation=transforms.functional.InterpolationMode.BILINEAR,
                    ),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __call__(self, x):
        return self.transform(x)


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.data_path = hyperparams.data_path
        self.hyperparams = hyperparams

        self.setup()

    def setup(self, stage=None):
        self.loaders = self.get_imagenet_pytorch_dataloaders(
            data_dir=self.data_path,
            batch_size=self.hyperparams.batch_size,
            num_workers=self.hyperparams.num_workers,
        )

    def train_dataloader(self):
        return self.loaders["train"]

    def val_dataloader(self):
        return self.loaders["val"]

    def test_dataloader(self):
        return self.loaders["val"]

    def get_imagenet_pytorch_dataloaders(
        self, data_dir=None, batch_size=None, num_workers=None
    ):
        paths = {
            "train": data_dir + "/train",
            "val": data_dir + "/val",
        }

        loaders = {}

        for name in ["train", "val"]:
            dataset = torchvision.datasets.ImageFolder(paths[name], Transform(name))
            drop_last = True if name == "train" else False
            shuffle = True if name == "train" else False
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=shuffle,
                drop_last=drop_last,
            )
            loaders[name] = loader

        return loaders

    def get_imagenet_pytorch_dataloaders_distributed(
        self, data_dir=None, batch_size=None, num_workers=None, world_size=None
    ):
        paths = {"train": data_dir + "/train", "val": data_dir + "/val"}

        loaders = {}

        for name in ["train", "val"]:
            dataset = torchvision.datasets.ImageFolder(paths[name], Transform())
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            assert batch_size % world_size == 0
            per_device_batch_size = batch_size // world_size
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=per_device_batch_size,
                num_workers=num_workers,
                pin_memory=True,
                sampler=sampler,
            )
            loaders[name] = loader

        return loaders, sampler
