import math

import torch
from omegaconf import DictConfig
from torchvision import transforms


def get_inference_method(
    canonicalizer: torch.nn.Module,
    prediction_network: torch.nn.Module,
    num_classes: int,
    inference_hyperparams: DictConfig,
    in_shape: tuple = (3, 32, 32),
):
    if inference_hyperparams.method == "vanilla":
        return VanillaInference(canonicalizer, prediction_network, num_classes)
    elif inference_hyperparams.method == "group":
        return GroupInference(
            canonicalizer,
            prediction_network,
            num_classes,
            inference_hyperparams,
            in_shape,
        )
    else:
        raise ValueError(f"{inference_hyperparams.method} is not implemented for now.")


class VanillaInference:
    def __init__(
        self,
        canonicalizer: torch.nn.Module,
        prediction_network: torch.nn.Module,
        num_classes: int,
    ) -> None:
        self.canonicalizer = canonicalizer
        self.prediction_network = prediction_network
        self.num_classes = num_classes

    def forward(self, x):
        # canonicalize the input data
        # For the vanilla model, the canonicalization is the identity transformation
        x_canonicalized = self.canonicalizer(x)

        # Forward pass through the prediction network as you'll normally do
        logits = self.prediction_network(x_canonicalized)
        return logits

    def get_inference_metrics(self, x: torch.Tensor, y: torch.Tensor):
        # Forward pass through the prediction network
        logits = self.forward(x)
        preds = logits.argmax(dim=-1)

        # Calculate the accuracy
        acc = (preds == y).float().mean()
        metrics = {"test/acc": acc}

        # Calculate accuracy per class
        acc_per_class = [
            (preds[y == i] == y[y == i]).float().mean() for i in range(self.num_classes)
        ]

        # check if the accuracy per class is nan
        acc_per_class = [0.0 if math.isnan(acc) else acc for acc in acc_per_class]

        # Update metrics with accuracy per class
        metrics.update(
            {
                f"test/acc_class_{i}": max(acc, 0.0)
                for i, acc in enumerate(acc_per_class)
            }
        )

        return metrics


class GroupInference(VanillaInference):
    def __init__(
        self,
        canonicalizer: torch.nn.Module,
        prediction_network: torch.nn.Module,
        num_classes: int,
        inference_hyperparams: DictConfig,
        in_shape: tuple = (3, 32, 32),
    ):

        super().__init__(canonicalizer, prediction_network, num_classes)
        self.group_type = inference_hyperparams.group_type
        self.num_rotations = inference_hyperparams.num_rotations
        self.num_group_elements = (
            self.num_rotations
            if self.group_type == "rotation"
            else 2 * self.num_rotations
        )
        self.pad = transforms.Pad(math.ceil(in_shape[-2] * 0.4), padding_mode="edge")
        self.crop = transforms.CenterCrop((in_shape[-2], in_shape[-1]))

    def get_group_element_wise_logits(self, x: torch.Tensor):
        logits_dict = {}
        degrees = torch.linspace(0, 360, self.num_rotations + 1)[:-1]
        for rot, degree in enumerate(degrees):

            x_pad = self.pad(x)
            x_rot = transforms.functional.rotate(x_pad, degree.item())
            x_rot = self.crop(x_rot)

            logits_dict[rot] = self.forward(x_rot)

        if self.group_type == "roto-reflection":
            # Rotate the reflected images and get the logits
            for rot, degree in enumerate(degrees):

                x_pad = self.pad(x)
                x_reflect = transforms.functional.hflip(x_pad)
                x_rotoreflect = transforms.functional.rotate(x_reflect, degree.item())
                x_rotoreflect = self.crop(x_rotoreflect)

                logits_dict[rot + len(degrees)] = self.forward(x_rotoreflect)

        return logits_dict

    def get_inference_metrics(self, x: torch.Tensor, y: torch.Tensor):

        logits_dict = self.get_group_element_wise_logits(x)

        # Use list comprehension to calculate accuracy for each group element
        acc_per_group_element = torch.tensor(
            [
                (logits.argmax(dim=-1) == y).float().mean()
                for logits in logits_dict.values()
            ]
        )

        metrics = {"test/group_acc": torch.mean(acc_per_group_element)}
        metrics.update(
            {
                f"test/acc_group_element_{i}": acc_per_group_element[i]
                for i in range(self.num_group_elements)
            }
        )

        preds = logits_dict[0].argmax(dim=-1)

        # Calculate the accuracy
        acc = (preds == y).float().mean()
        metrics.update({"test/acc": acc})

        # Calculate accuracy per class
        acc_per_class = [
            (preds[y == i] == y[y == i]).float().mean() for i in range(self.num_classes)
        ]

        # check if the accuracy per class is nan
        acc_per_class = [0.0 if math.isnan(acc) else acc for acc in acc_per_class]

        # Update metrics with accuracy per class
        metrics.update(
            {f"test/acc_class_{i}": acc for i, acc in enumerate(acc_per_class)}
        )

        return metrics
