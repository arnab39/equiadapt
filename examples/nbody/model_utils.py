from equiadapt.nbody.canonicalization_networks.euclideangraph_base_models import (
    EGNN_vel,
    VNDeepSets,
    GNN,
    Transformer,
)
import torch


def get_canonicalization_network(hyperparams):
    architecture = hyperparams.architecture
    model_dict = {
        "vndeepsets": lambda: VNDeepSets(hyperparams),
    }

    return model_dict[architecture]()


def get_prediction_network(hyperparams):
    architecture = hyperparams.architecture
    model_dict = {
        "GNN": lambda: GNN(hyperparams),
        "EGNN": lambda: EGNN_vel(hyperparams),
        "vndeepsets": lambda: VNDeepSets(hyperparams),
        "Transformer": lambda: Transformer(hyperparams),
    }
    if architecture not in model_dict:
        raise ValueError(
            f"{architecture} is not implemented as prediction network for now."
        )

    return model_dict[architecture]()


def get_edges(batch_size, n_nodes):
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
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges
