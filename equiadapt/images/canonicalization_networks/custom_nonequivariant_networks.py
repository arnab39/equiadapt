from typing import List

import torch, torchvision
from torch import nn


class ConvNetwork(nn.Module):
    def __init__(self, 
                 in_shape: tuple, 
                 out_channels: int, 
                 kernel_size: int, 
                 num_layers: int = 2, 
                 out_vector_size: int = 128):
        super().__init__()

        in_channels = in_shape[0]
        layers: List[nn.Module] = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, 2))
            elif i % 3 == 2:
                layers.append(nn.Conv2d(out_channels, 2 * out_channels, kernel_size, 2, 1))
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
                                    nn.Linear(out_dim, out_vector_size)
                                    )
        self.out_vector_size = out_vector_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, in_channels, height, width)
        :return: (batch_size, 1)
        """
        batch_size = x.shape[0]
        out = self.enc_network(x)
        out = out.reshape(batch_size, -1)
        return self.final_fc(out)
    
class ResNet18Network(nn.Module):
    def __init__(self, 
                 in_shape: tuple, 
                 out_channels: int, 
                 kernel_size: int, 
                 num_layers: int = 2, 
                 out_vector_size: int = 128):
        super().__init__()
        self.resnet18 = torchvision.models.resnet18(weights=None)
        self.resnet18.fc = nn.Sequential(
            nn.Linear(512, out_vector_size),
        )

        self.out_vector_size = out_vector_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, in_channels, height, width)
        :return: (batch_size, 1)
        """
        return self.resnet18(x)