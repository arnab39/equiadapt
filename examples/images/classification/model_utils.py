import torch
import torch.nn as nn
import torchvision


class PredictionNetwork(nn.Module):
    def __init__(self, encoder: torch.nn.Module, feature_dim: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.predictor = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reps = self.encoder(x)
        reps = reps.view(x.shape[0], -1)
        return self.predictor(reps)


def get_dataset_specific_info(dataset_name: str) -> tuple:
    dataset_info = {
        "rotated_mnist": (nn.CrossEntropyLoss(), (1, 28, 28), 10),
        "cifar10": (nn.CrossEntropyLoss(), (3, 224, 224), 10),
        "cifar100": (nn.CrossEntropyLoss(), (3, 224, 224), 100),
        "stl10": (nn.CrossEntropyLoss(), (3, 224, 224), 10),
        "flowers102": (nn.CrossEntropyLoss(), (3, 224, 224), 102),
        "celeba": (nn.BCEWithLogitsLoss(), (3, 224, 224), 40),
        "ImageNet": (nn.CrossEntropyLoss(), (3, 224, 224), 1000),
    }

    if dataset_name not in dataset_info:
        raise ValueError("Dataset not implemented for now.")

    return dataset_info[dataset_name]


def get_prediction_network(
    architecture: str = "resnet50",
    dataset_name: str = "cifar10",
    use_pretrained: bool = False,
    freeze_encoder: bool = False,
    input_shape: tuple = (3, 32, 32),
    num_classes: int = 10,
) -> torch.nn.Module:
    weights = "DEFAULT" if use_pretrained else None
    model_dict = {
        "resnet50": torchvision.models.resnet50,
        "vit": torchvision.models.vit_b_16,
    }

    if architecture not in model_dict:
        raise ValueError(
            f"{architecture} is not implemented as prediction network for now."
        )

    encoder = model_dict[architecture](weights=weights)

    if architecture == "resnet50" and dataset_name in (
        "cifar10",
        "cifar100",
        "rotated_mnist",
    ):
        if input_shape[-2:] == [32, 32] or dataset_name == "rotated_mnist":
            encoder.conv1 = nn.Conv2d(
                input_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            encoder.maxpool = nn.Identity()

    if freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False

    if dataset_name != "ImageNet":
        if architecture == "resnet50":
            feature_dim = encoder.fc.in_features
            encoder.fc = nn.Identity()
        elif architecture == "vit":
            feature_dim = encoder.heads.head.in_features
            encoder.heads.head = nn.Identity()
        prediction_network = PredictionNetwork(encoder, feature_dim, num_classes)
    else:
        prediction_network = encoder

    return prediction_network
