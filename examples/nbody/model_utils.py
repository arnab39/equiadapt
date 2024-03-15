from typing import Any, List

import torch
from torch import nn

from equiadapt.nbody.canonicalization_networks.custom_equivariant_networks import (
    VNDeepSets,
)
from examples.nbody.networks.euclideangraph_base_models import GNN, Transformer


def get_canonicalization_network(hyperparams: Any) -> nn.Module:
    """
    Returns the canonicalization network based on the given hyperparameters.

    Args:
        hyperparams: The hyperparameters for the network.

    Returns:
        The canonicalization network.

    Raises:
        ValueError: If the specified architecture is not implemented.
    """
    architecture = hyperparams.architecture
    model_dict = {
        "vndeepsets": lambda: VNDeepSets(hyperparams.network_hyperparams),
    }

    return model_dict[architecture]()


def get_prediction_network(hyperparams: Any) -> nn.Module:
    """
    Returns the prediction network based on the given hyperparameters.

    Args:
        hyperparams: The hyperparameters for the network.

    Returns:
        The prediction network.

    Raises:
        ValueError: If the specified architecture is not implemented.
    """
    architecture = hyperparams.architecture
    model_dict = {
        "GNN": lambda: GNN(hyperparams.network_hyperparams),
        "vndeepsets": lambda: VNDeepSets(hyperparams.network_hyperparams),
        "Transformer": lambda: Transformer(hyperparams.network_hyperparams),
    }
    if architecture not in model_dict:
        raise ValueError(
            f"{architecture} is not implemented as a prediction network for now."
        )

    return model_dict[architecture]()


def get_edges(batch_size: int, n_nodes: int) -> List[torch.LongTensor]:
    """
    Returns the edges of the graph.

    Args:
        batch_size: The number of graphs in the batch.
        n_nodes: The number of nodes in each graph.

    Returns:
        The edges of the graph as a tuple of two LongTensors.

    Raises:
        ValueError: If the batch size is less than 1.
    """
    edges = [
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
        [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3],
    ]
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(torch.add(edges[0], n_nodes * i))
            cols.append(torch.add(edges[1], n_nodes * i))
        edges = [torch.cat(rows), torch.cat(cols)]
    else:
        raise ValueError("Batch size must be greater than or equal to 1.")
    return edges
