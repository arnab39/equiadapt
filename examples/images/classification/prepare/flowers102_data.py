
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import Flowers102
import os

import random

class Flowers102DataModule(pl.LightningDataModule):
    def __init__(self, hyperparams, download=False):
        super().__init__()
        self.data_path = hyperparams.data_path
        self.hyperparams = hyperparams
        self.train_transform = transforms.Compose(
            [
                transforms.RandomRotation(30),
                transforms.RandomCrop(224, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        os.makedirs(self.data_path, exist_ok=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = Flowers102(self.data_path, split='train', transform=self.train_transform, download=True)
            self.valid_dataset = Flowers102(self.data_path, split='val', transform=self.test_transform, download=True)
        if stage == "test":
            self.test_dataset = Flowers102(self.data_path, split='test', transform=self.test_transform, download=True)
            print('Test dataset size: ', len(self.test_dataset))

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
