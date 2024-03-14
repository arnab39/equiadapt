from .cifar_data import CIFAR10DataModule, CIFAR100DataModule
from .imagenet_data import ImageNetDataModule
from .rotated_mnist_data import RotatedMNISTDataModule
from .stl10_data import STL10DataModule

__all__ = [
    "CIFAR10DataModule",
    "CIFAR100DataModule",
    "ImageNetDataModule",
    "RotatedMNISTDataModule",
    "STL10DataModule",
]
