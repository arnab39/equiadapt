from typing import List

import torch
import torchvision
from torch import nn


class ConvNetwork(nn.Module):
    """
    This class represents a convolutional neural network.

    The network consists of a sequence of convolutional layers, each followed by batch normalization and a GELU activation function. The number of output channels of the convolutional layers increases after every third layer. The network ends with a fully connected layer.

    Methods:
        __init__: Initializes the ConvNetwork instance.
        forward: Performs a forward pass through the network.
    """

    def __init__(
        self,
        in_shape: tuple,
        out_channels: int,
        kernel_size: int,
        num_layers: int = 2,
        out_vector_size: int = 128,
    ):
        """
        Initializes the ConvNetwork instance.

        Args:
            in_shape (tuple): The shape of the input data. It should be a tuple of the form (in_channels, height, width).
            out_channels (int): The number of output channels of the first convolutional layer.
            kernel_size (int): The size of the kernel of the convolutional layers.
            num_layers (int, optional): The number of convolutional layers. Defaults to 2.
            out_vector_size (int, optional): The size of the output vector of the network. Defaults to 128.
        """
        super().__init__()

        in_channels = in_shape[0]
        layers: List[nn.Module] = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, 2))
            elif i % 3 == 2:
                layers.append(
                    nn.Conv2d(out_channels, 2 * out_channels, kernel_size, 2, 1)
                )
                out_channels *= 2
            else:
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, 2))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.GELU())

        self.enc_network = nn.Sequential(*layers)
        out_shape = self.enc_network(torch.zeros(1, *in_shape)).shape

        # self.scalar_fc = nn.Linear(out_shape[1] * out_shape[2] * out_shape[3], 1)
        out_dim = out_shape[1] * out_shape[2] * out_shape[3]
        self.final_fc = nn.Sequential(
            nn.BatchNorm1d(out_dim),
            nn.Dropout1d(0.5),
            nn.ReLU(),
            nn.Linear(out_dim, out_vector_size),
        )
        self.out_vector_size = out_vector_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): The input data. It should have the shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: The output of the network. It has the shape (batch_size, out_vector_size).
        """
        batch_size = x.shape[0]
        out = self.enc_network(x)
        out = out.reshape(batch_size, -1)
        return self.final_fc(out)


class ResNet18Network(nn.Module):
    """
    This class represents a neural network based on the ResNet-18 architecture.

    The network uses a pre-trained ResNet-18 model without its weights. The final fully connected layer of the ResNet-18 model is replaced with a new fully connected layer.

    Attributes:
        resnet18 (torchvision.models.ResNet): The ResNet-18 model.
        out_vector_size (int): The size of the output vector of the network.
    """

    def __init__(
        self,
        in_shape: tuple,
        out_channels: int,
        kernel_size: int,
        num_layers: int = 2,
        out_vector_size: int = 128,
    ):
        """
        Initializes the ResNet18Network instance.

        Args:
            in_shape (tuple): The shape of the input data. It should be a tuple of the form (in_channels, height, width).
            out_channels (int): The number of output channels of the first convolutional layer.
            kernel_size (int): The size of the kernel of the convolutional layers.
            num_layers (int, optional): The number of convolutional layers. Defaults to 2.
            out_vector_size (int, optional): The size of the output vector of the network. Defaults to 128.
        """
        super().__init__()
        self.resnet18 = torchvision.models.resnet18(weights=None)
        self.resnet18.fc = nn.Sequential(
            nn.Linear(512, out_vector_size),
        )

        self.out_vector_size = out_vector_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): The input data. It should have the shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: The output of the network. It has the shape (batch_size, 1).
        """
        return self.resnet18(x)
