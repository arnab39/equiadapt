# Description: Contains the implementation of non-equivariant canonicalization networks.

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from equiadapt.pointcloud.canonicalization_networks.utils import get_graph_feature

class PointNet_small(nn.Module):
    def __init__(self, hyperparams: DictConfig):
        super().__init__()
        # Fewer convolutional layers and reduced channel sizes
        self.conv1 = nn.Conv1d(3, 45, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(45, 90, kernel_size=1, bias=False)  # Merging two layers into one
        self.bn1 = nn.BatchNorm1d(45)
        self.bn2 = nn.BatchNorm1d(90)
        self.linear1 = nn.Linear(90, int(512 * 0.7), bias=False)  # Reduced dimension
        self.bn3 = nn.BatchNorm1d(int(512 * 0.7))
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(int(512 * 0.7), hyperparams.output_dim)
        self.out_vector_size = hyperparams.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn3(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x
    
    
class DGCNN_small(nn.Module):
    def __init__(self, hyperparams: DictConfig):
        super().__init__()
        self.n_knn = hyperparams.n_knn
        # Fewer convolutional layers and reduced feature dimensions
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 45, kernel_size=1, bias=False),
            nn.BatchNorm2d(45),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(45 * 2, 90, kernel_size=1, bias=False),  # Skipping one layer and increasing channel size here
            nn.BatchNorm2d(90),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.linear1 = nn.Linear(90, int(512 * 0.7), bias=False)
        self.bn1 = nn.BatchNorm1d(int(512 * 0.7))
        self.dp1 = nn.Dropout(p=hyperparams.dropout)
        self.linear2 = nn.Linear(int(512 * 0.7), hyperparams.output_dim)
        self.out_vector_size = hyperparams.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        x = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv2(x)
        x = x.max(dim=-1, keepdim=False)[0]

        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = self.linear2(x)
        return x
