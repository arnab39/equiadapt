import torch
import torch.nn as nn
from equiadapt.pointcloud.canonicalization_networks.vector_neuron_layers import VNLinearLeakyReLU, VNLinear, VNMaxPool, VNBatchNorm

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)

def get_graph_feature_cross(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)

    idx_base = torch.arange(0, batch_size).type_as(idx).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)

    feature = torch.cat((feature - x, x, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature

class VNSmall(torch.nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
        self.n_knn = hyperparams.n_knn
        self.pooling = hyperparams.pooling
        self.conv_pos = VNLinearLeakyReLU(3, 64 // 3, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(64 // 3, 64 // 3, dim=4, negative_slope=0.0)
        self.bn1 = VNBatchNorm(64 // 3, dim=4)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 12 // 3, dim=4, negative_slope=0.0)
        self.dropout = nn.Dropout(p=0.5)

        if self.pooling == "max":
            self.pool = VNMaxPool(64 // 3)
        elif self.pooling == "mean":
            self.pool = mean_pool
        else:
            raise ValueError(f"Pooling type {self.pooling} not supported")

        # Wild idea -- Just use a linear layer to predict the output
        # self.conv = VNLinear(3, 12 // 3)


    def forward(self, point_cloud):

        point_cloud = point_cloud.unsqueeze(1)
        feat = get_graph_feature_cross(point_cloud, k=self.n_knn)
        point_cloud = self.conv_pos(feat)
        point_cloud = self.pool(point_cloud)
        
        out = self.bn1(self.conv1(point_cloud))
        out = self.conv2(out)
        out = self.dropout(out)

        # out = self.pool(self.conv(feat))
        # out = self.dropout(out)

        return out.mean(dim=-1)[:, :3]

