import torch
from omegaconf import DictConfig

from examples.pointcloud.common.networks import DGCNN, PointNet


def get_prediction_network(
    architecture: str,
    hyperparams: DictConfig,
) -> torch.nn.Module:
    """
    The function returns the prediction network based on the architecture type
    """
    model_dict = {
        "pointnet": PointNet,
        "dgcnn": DGCNN,
    }

    if architecture not in model_dict:
        raise ValueError(
            f"{architecture} is not implemented as prediction network for now."
        )

    prediction_network = model_dict[architecture](hyperparams.network_hyperparams)
    
    # first train a model with identity canonicalization (or look online for pointnet pretrained model)
    # load and remove last later as in image experiments
    # look for Mamba3D weights on huggingface
    # model is fixed only train the canonicalizer
    # first time, need to run prior calculation with samll batch size
    # second time, train canonicalizer
    # shapenet benchmark from papers with code (same thing for segmentation)

    return prediction_network
