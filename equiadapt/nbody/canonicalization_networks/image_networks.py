import torch.nn.functional as F
import kornia as K
from torch import nn
import torch
from canonical_network.models.equivariant_layers import (
    RotationEquivariantConvLift,
    RotoReflectionEquivariantConvLift,
    RotationEquivariantConv,
    RotoReflectionEquivariantConv,
)
from canonical_network.models.set_base_models import SequentialMultiple
import numpy as np


class CanonizationNetwork(nn.Module):
    def __init__(
        self,
        in_shape,
        out_channels,
        kernel_size,
        group_type="rotation",
        num_rotations=4,
        num_layers=1,
        device="cuda",
    ):
        super().__init__()
        if group_type == "rotation":
            layer_list = [
                RotationEquivariantConvLift(
                    in_shape[0], out_channels, kernel_size, num_rotations, device=device
                )
            ]
            for i in range(num_layers - 1):
                layer_list.append(nn.ReLU())
                layer_list.append(
                    RotationEquivariantConv(
                        out_channels, out_channels, 1, num_rotations, device=device
                    )
                )
            self.eqv_network = nn.Sequential(*layer_list)
        elif group_type == "roto-reflection":
            layer_list = [
                RotoReflectionEquivariantConvLift(
                    in_shape[0], out_channels, kernel_size, num_rotations, device=device
                )
            ]
            for i in range(num_layers - 1):
                layer_list.append(nn.ReLU())
                layer_list.append(
                    RotoReflectionEquivariantConv(
                        out_channels, out_channels, 1, num_rotations, device=device
                    )
                )
            self.eqv_network = nn.Sequential(*layer_list)
        else:
            raise ValueError("group_type must be rotation or roto-reflection for now.")
        out_shape = self.eqv_network(torch.zeros(1, *in_shape).to(device)).shape
        print("Canonization feature map shape:", out_shape)

    def forward(self, x):
        """
        x shape: (batch_size, in_channels, height, width)
        :return: (batch_size, group_size)
        """
        feature_map = self.eqv_network(x)
        feature_fibres = torch.mean(feature_map, dim=(1, 3, 4))
        return feature_fibres


class EquivariantCanonizationNetwork(nn.Module):
    def __init__(
        self,
        base_encoder,
        in_shape,
        num_classes,
        canonization_out_channels,
        canonization_num_layers,
        canonization_kernel_size,
        canonization_beta=1e4,
        group_type="rotation",
        num_rotations=4,
        device="cuda",
        batch_size=128,
    ):
        super().__init__()
        self.canonization_network = CanonizationNetwork(
            in_shape,
            canonization_out_channels,
            canonization_kernel_size,
            group_type,
            num_rotations,
            canonization_num_layers,
            device,
        )
        print(self.canonization_network)
        self.base_encoder = base_encoder
        out_shape = self.base_encoder(torch.zeros(batch_size, *in_shape)).shape
        print("Base encoder feature map shape:", out_shape)
        print(self.base_encoder)
        if len(out_shape) == 4:
            self.predictor = nn.Linear(
                out_shape[1] * out_shape[2] * out_shape[3], num_classes
            )
        elif len(out_shape) == 2:
            self.predictor = nn.Linear(out_shape[1], num_classes)
        else:
            raise ValueError("Base encoder output shape must be 2 or 4 dimensional.")
        self.num_rotations = num_rotations
        self.group_type = group_type
        self.beta = canonization_beta
        self.num_group = (
            num_rotations if group_type == "rotation" else 2 * num_rotations
        )

    def fibres_to_group(self, fibre_activations):
        device = fibre_activations.device
        # fibre_activations_one_hot = torch.nn.functional.softmax(self.beta * fibre_activations, dim=-1)
        fibre_activations_one_hot = torch.nn.functional.one_hot(
            torch.argmax(fibre_activations, dim=-1), self.num_group
        ).float()
        fibre_activations_soft = torch.nn.functional.softmax(
            self.beta * fibre_activations, dim=-1
        )
        angles = torch.linspace(0.0, 360.0, self.num_rotations + 1)[
            : self.num_rotations
        ].to(device)
        angles = (
            torch.cat([angles, angles], dim=0)
            if self.group_type == "roto-reflection"
            else angles
        )
        if self.training:
            # angle = torch.sum(fibre_activations_one_hot * angles, dim=-1)
            angle = torch.sum(
                (
                    fibre_activations_one_hot
                    + fibre_activations_soft
                    - fibre_activations_soft.detach()
                )
                * angles,
                dim=-1,
            )
        else:
            angle = torch.sum(fibre_activations_one_hot * angles, dim=-1)
        if self.group_type == "roto-reflection":
            reflect_one_hot = torch.cat(
                [torch.zeros(self.num_rotations), torch.ones(self.num_rotations)], dim=0
            ).to(device)
            if self.training:
                reflect_indicator = torch.sum(
                    (
                        fibre_activations_one_hot
                        + fibre_activations_soft
                        - fibre_activations_soft.detach()
                    )
                    * reflect_one_hot,
                    dim=-1,
                )
            else:
                reflect_indicator = torch.sum(
                    fibre_activations_one_hot * reflect_one_hot, dim=-1
                )
            return angle, reflect_indicator
        else:
            return angle

    def inverse_action(self, x, fibres_activations):
        """
        x shape: (batch_size, in_channels, height, width)
        fibres_activations shape: (batch_size, group_size)
        :return: (batch_size, in_channels, height, width)
        """
        if self.group_type == "rotation":
            angles = self.fibres_to_group(fibres_activations)
            group = [angles]
            x = K.geometry.rotate(x, -angles)
        elif self.group_type == "roto-reflection":
            angles, reflect_indicator = self.fibres_to_group(fibres_activations)
            group = [angles, reflect_indicator]
            x_reflected = K.geometry.hflip(x)
            reflect_indicator = reflect_indicator[:, None, None, None]
            x = (1 - reflect_indicator) * x + reflect_indicator * x_reflected
            x = K.geometry.rotate(x, -angles)
        return x, group

    def forward(self, x):
        """
        x shape: (batch_size, in_channels, height, width)
        :return: (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        x_canonized, group = self.get_canonized_images(x)
        reps = self.base_encoder(x_canonized)
        reps = reps.reshape(batch_size, -1)
        return self.predictor(reps)

    def get_canonized_images(self, x):
        fibres_activations = self.canonization_network(x)
        # x = transforms.Pad(4)(x)
        x_canonized, group = self.inverse_action(x, fibres_activations)
        return x_canonized, group


class VanillaNetwork(nn.Module):
    def __init__(self, encoder, in_shape, num_classes, batch_size=128, device="cuda"):
        super().__init__()
        self.encoder = encoder.to(device)
        print(self.encoder)
        out_shape = self.encoder(torch.zeros(batch_size, *in_shape).to(device)).shape
        print("feature map shape:", out_shape)

        if len(out_shape) == 4:
            self.predictor = nn.Linear(
                out_shape[1] * out_shape[2] * out_shape[3], num_classes
            )
        elif len(out_shape) == 2:
            self.predictor = nn.Linear(out_shape[1], num_classes)
        else:
            raise ValueError("Base encoder output shape must be 2 or 4 dimensional.")

    def forward(self, x):
        reps = self.encoder(x)
        reps = reps.view(x.shape[0], -1)
        return self.predictor(reps)


class PCACanonizationNetwork(nn.Module):
    def __init__(self, encoder, in_shape, num_classes, batch_size=128):
        super().__init__()
        self.encoder = encoder
        out_shape = self.encoder(torch.zeros(batch_size, *in_shape)).shape
        print("feature map shape:", out_shape)
        print(self.encoder)
        if len(out_shape) == 4:
            self.predictor = nn.Linear(
                out_shape[1] * out_shape[2] * out_shape[3], num_classes
            )
        elif len(out_shape) == 2:
            self.predictor = nn.Linear(out_shape[1], num_classes)
        else:
            raise ValueError("Base encoder output shape must be 2 or 4 dimensional.")

    def get_angles(self, images):
        images = images.reshape(images.shape[0], -1)
        device = images.device
        xs = np.linspace(-14, 14, num=28)
        ys = np.linspace(14, -14, num=28)
        x, y = np.meshgrid(
            xs,
            ys,
        )
        x, y = torch.tensor(x).float(), torch.tensor(y).float()
        angle_list = []
        for i in range(images.shape[0]):
            image = images[i]
            x_selected = x.flatten()[image > 0.5]
            y_selected = y.flatten()[image > 0.5]
            data = torch.cat([x_selected[:, None], y_selected[:, None]], dim=1)
            data = data - data.mean(dim=0)
            u, s, v = torch.svd(data)
            vect = v[:, torch.argmax(s)]
            angle_list.append(torch.atan2(vect[1], vect[0]) * 180 / np.pi)

        return torch.stack(angle_list).to(device)

    def get_canonized_images(self, x):
        angles = self.get_angles(x).detach()
        x_canonized = K.geometry.rotate(x, -angles)
        return x_canonized, angles

    def forward(self, x):
        batch_size = x.shape[0]
        x_canonized, angles = self.get_canonized_images(x)
        reps = self.encoder(x_canonized)
        reps = reps.view(batch_size, -1)
        return self.predictor(reps)


class OptimizationCanonizationNetwork(nn.Module):
    def __init__(self, encoder, in_shape, num_classes, hyperparams=None):
        super().__init__()
        self.energy = CustomDeepSets(hyperparams)
        self.lr = hyperparams.rot_opt_lr
        self.iters = hyperparams.num_optimization_iters
        self.implicit = True if hyperparams.implicit else False
        self.encoder = encoder
        out_shape = self.encoder(torch.zeros(1, *in_shape)).shape
        print("feature map shape:", out_shape)
        print(self.encoder)
        if len(out_shape) == 4:
            self.predictor = nn.Linear(
                out_shape[1] * out_shape[2] * out_shape[3], num_classes
            )
        elif len(out_shape) == 2:
            self.predictor = nn.Linear(out_shape[1], num_classes)
        else:
            raise ValueError("Base encoder output shape must be 2 or 4 dimensional.")

        # n_angles = 4
        # angles = 2 * torch.pi * torch.arange(n_angles) / n_angles
        # self.register_buffer('initial_rotation', self.generate_rotations(angles))

    @torch.enable_grad()
    def min_energy(self, points):
        batch_size = points.shape[0]
        rotation_angle = torch.zeros(batch_size, device=points.device).requires_grad_(
            True
        )
        # rotations = self.initial_rotation.clone().requires_grad_(True).unsqueeze(0).expand(batch_size, -1, -1, -1)
        # rotated_points = self.apply_rotations_to_points(points, rotations)
        # n_rotations = rotations.size(1)
        # energy = self.energy(rotated_points.flatten(0, 1)).view(-1, n_rotations)
        # _, indices = energy.min(dim=1, keepdim=True)
        # indices = torch.ones(points.size(0), 1, device=points.device).long()
        # best_rotation = rotations.gather(1, indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2))
        # return best_rotation.squeeze(1)

        m = torch.zeros_like(rotation_angle)
        for i in range(self.iters):
            if self.implicit:
                rotation_angle = rotation_angle.detach()
                rotation_angle.requires_grad_(True)
            rotation = self.generate_rotations(rotation_angle)

            rotated_coord = torch.bmm(points[:, :, :2], rotation)
            rotated = torch.cat([rotated_coord, points[:, :, 2:]], dim=-1)
            energy = self.energy(rotated)
            print(i, "e", energy[0])
            (g,) = torch.autograd.grad(
                energy.sum(),
                rotation_angle,
                only_inputs=True,
                create_graph=(i == self.iters - 1) if self.implicit else True,
            )
            if m is None:
                m = g
            else:
                m = 0.5 * g
            rotation_angle = rotation_angle - self.lr * m
            print(i, "  g", g[0])
            print(i, "    a", rotation_angle[0])
        return self.generate_rotations(rotation_angle)

    def apply_rotations_to_points(self, points, rotation):
        coords, data = points[:, :, :2], points[:, :, 2:]
        new_coords = torch.einsum("nsc, nrcd -> nrsd", coords, rotation)
        # new shape is (batch, rotations, set, dimensions)
        return torch.cat(
            [new_coords, data.unsqueeze(1).expand(-1, rotation.size(1), -1, -1)], dim=3
        )

    def generate_rotations(self, angles):
        rotations = torch.stack(
            [
                torch.cos(angles),
                torch.sin(angles),
                -torch.sin(angles),
                torch.cos(angles),
            ],
            dim=1,
        )
        rotations = rotations.view(-1, 2, 2)
        return rotations

    def get_canonized_images(self, x, points):
        rotation_matrices = self.min_energy(points)
        # code for affine matrix from rotation matrices
        # https://kornia.readthedocs.io/en/v0.1.2/geometric.html#torchgeometry.get_rotation_matrix2d
        alpha = rotation_matrices[:, 0, 0]
        beta = rotation_matrices[:, 0, 1]
        cx = cy = 28 / 2
        affine_part = torch.stack(
            [(1 - alpha) * cx - beta * cy, beta * cx + (1 - alpha) * cy], dim=1
        )
        affine_matrices = torch.cat(
            [rotation_matrices, affine_part.unsqueeze(-1)], dim=-1
        )

        x_canonized = K.geometry.affine(x, affine_matrices)
        return x_canonized

    def forward(self, images, points):
        # breakpoint()
        batch_size = points.shape[0]
        images_canonized = self.get_canonized_images(images, points)
        reps = self.encoder(images_canonized)
        reps = reps.view(batch_size, -1)
        return self.predictor(reps)

    def gram_schmidt(self, vectors):
        v1 = vectors[:, 0]
        v1 = v1 / torch.norm(v1, dim=1, keepdim=True)
        v2 = vectors[:, 1] - torch.sum(vectors[:, 1] * v1, dim=1, keepdim=True) * v1
        v2 = v2 / torch.norm(v2, dim=1, keepdim=True)
        return torch.stack([v1, v2], dim=1)


class BasicConvEncoder(nn.Module):
    def __init__(self, in_shape, out_channels, num_layers=6):
        super().__init__()
        encoder_layers = []
        for i in range(num_layers):
            if i == 0:
                encoder_layers.append(nn.Conv2d(in_shape[0], out_channels, 3, 1))
            elif i % 3 == 2:
                encoder_layers.append(
                    nn.Conv2d(out_channels, 2 * out_channels, 5, 2, 1)
                )
                out_channels *= 2
            else:
                encoder_layers.append(nn.Conv2d(out_channels, out_channels, 3, 1))
            encoder_layers.append(nn.BatchNorm2d(out_channels))
            encoder_layers.append(nn.ReLU())
            if i % 3 == 2:
                encoder_layers.append(nn.Dropout2d(0.4))

        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        return x


class RotationEquivariantConvEncoder(nn.Module):
    def __init__(
        self, in_shape, out_channels, num_layers=6, num_rotations=4, device="cuda"
    ):
        super().__init__()
        encoder_layers = []
        for i in range(num_layers):
            if i == 0:
                encoder_layers.append(
                    RotationEquivariantConvLift(
                        in_shape[0], out_channels, 3, num_rotations, 1, device=device
                    )
                )
            elif i % 3 == 2:
                encoder_layers.append(
                    RotationEquivariantConv(
                        out_channels,
                        out_channels,
                        5,
                        num_rotations,
                        2,
                        1,
                        device=device,
                    )
                )
                # out_channels *= 2
            else:
                encoder_layers.append(
                    RotationEquivariantConv(
                        out_channels, out_channels, 3, num_rotations, 1, device=device
                    )
                )

            encoder_layers.append(nn.BatchNorm3d(out_channels))
            encoder_layers.append(nn.ReLU())
            if i % 3 == 2:
                encoder_layers.append(nn.Dropout3d(0.4))

        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        return x.mean(dim=2)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class CustomSetLayer(nn.Module):
    def __init__(self, in_dim, out_dim, pooling="sum"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pooling = pooling

        self.identity_linear = nn.Linear(in_dim, out_dim)
        self.pooling_linear = nn.Linear(in_dim, out_dim)

    def forward(self, points):
        """
        :param points: shape (batch_size, num_points, in_dim)
        :return:
            output: shape (batch_size, num_points, out_dim)
        """
        identity = self.identity_linear(points)
        if self.pooling == "sum":
            pooled_set = points.sum(dim=1)
        elif self.pooling == "mean":
            pooled_set = points.mean(dim=1)
        elif self.pooling == "max":
            pooled_set = points.max(dim=1)[0]
        else:
            raise NotImplementedError
        pooling = self.pooling_linear(pooled_set)

        output = F.relu(identity + pooling[:, None, :])
        if self.in_dim == self.out_dim:
            output = output + points

        return output


class CustomDeepSets(nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
        self.start_layer = nn.Linear(
            3, hyperparams.hidden_dim
        )  # 3 for (x, y, pixel value)
        self.set_layers = SequentialMultiple(
            *[
                CustomSetLayer(
                    hyperparams.hidden_dim,
                    hyperparams.hidden_dim,
                    hyperparams.layer_pooling,
                )
                for i in range(hyperparams.num_layers - 1)
            ]
        )
        self.output_layer = SequentialMultiple(nn.Linear(hyperparams.hidden_dim, 1))
        self.final_pooling = hyperparams.final_pooling

    def forward(self, points):
        """
        :param points: shape (batch_size, num_points, 3)   # 3 for (x, y, pixel value)
        :return:
            output: shape (batch_size, 1)
        """
        embeddings = self.start_layer(points)
        x = self.set_layers(embeddings)
        if self.final_pooling == "sum":
            x = x.sum(dim=1)
        elif self.final_pooling == "mean":
            x = x.mean(dim=1)
        elif self.final_pooling == "max":
            x = x.max(dim=1)[0]
        else:
            raise NotImplementedError
        output = self.output_layer(x)
        return output
