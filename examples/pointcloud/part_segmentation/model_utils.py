import torch
from omegaconf import DictConfig

from examples.pointcloud.common.networks import DGCNN_partseg


def get_prediction_network(
    architecture: str,
    hyperparams: DictConfig,
) -> torch.nn.Module:
    """
    The function returns the prediction network based on the architecture type
    """
    model_dict = {"dgcnn": DGCNN_partseg}

    if architecture not in model_dict:
        raise ValueError(
            f"{architecture} is not implemented as prediction network for now."
        )

    prediction_network = model_dict[architecture](hyperparams.network_hyperparams)

    return prediction_network
