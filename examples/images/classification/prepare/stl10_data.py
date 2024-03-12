import os
import random

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import STL10


class CustomRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)


class STL10DataModule(pl.LightningDataModule):
    def __init__(self, hyperparams, download=False):
        super().__init__()
        self.data_path = hyperparams.data_path
        self.hyperparams = hyperparams
        if hyperparams.augment == 1:
            self.train_transform = transforms.Compose(
                [
                    transforms.Pad(4),
                    transforms.RandomCrop(96),
                    transforms.Resize(224),
                    transforms.RandomRotation(5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        elif hyperparams.augment == 2:
            # all augmentations
            self.train_transform = transforms.Compose(
                [
                    transforms.Pad(4),
                    transforms.RandomCrop(96),
                    transforms.Resize(224),
                    CustomRotationTransform([0, 45, 90, 135, 180, 225, 270, 315]),
                    # transforms.RandomRotation(180),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            # no rotation augmentations
            self.train_transform = transforms.Compose(
                [
                    transforms.Pad(4),
                    transforms.RandomCrop(96),
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        self.test_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        os.makedirs(self.data_path, exist_ok=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = STL10(
                self.data_path,
                split="train",
                transform=self.train_transform,
                download=True,
            )
            self.valid_dataset = STL10(
                self.data_path,
                split="test",
                transform=self.test_transform,
                download=True,
            )
        if stage == "test":
            self.test_dataset = STL10(
                self.data_path,
                split="test",
                transform=self.test_transform,
                download=True,
            )
            print("Test dataset size: ", len(self.test_dataset))

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            self.hyperparams.batch_size,
            shuffle=True,
            num_workers=self.hyperparams.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            self.hyperparams.batch_size,
            shuffle=False,
            num_workers=self.hyperparams.num_workers,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            self.hyperparams.batch_size,
            shuffle=False,
            num_workers=self.hyperparams.num_workers,
        )
        return test_loader
