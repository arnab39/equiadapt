from equiadapt.nbody.canonicalization_networks.euclideangraph_base_models import EGNN_vel, VNDeepSets, GNN, Transformer
from collections import namedtuple
from omegaconf import OmegaConf
import torch

def get_canonicalization_network(hyperparams):
    architecture = hyperparams.canon_model_type
    model_hyperparams = {
            "num_layers": hyperparams.canon_num_layers,
            "hidden_dim": hyperparams.canon_hidden_dim,
            "layer_pooling": hyperparams.canon_layer_pooling,
            "final_pooling": hyperparams.canon_final_pooling,
            "out_dim": 4,
            "batch_size": hyperparams.batch_size,
            "nonlinearity": hyperparams.canon_nonlinearity,
            "canon_feature": hyperparams.canon_feature,
            "canon_translation": hyperparams.canon_translation,
            "angular_feature": hyperparams.canon_angular_feature,
            "dropout": hyperparams.canon_dropout,
        }
    model_hyperparams = OmegaConf.create(dict(model_hyperparams))
    
    model_dict = {
    #"EGNN": lambda: EGNN_vel(hyperparams),
    "vndeepsets": lambda: VNDeepSets(model_hyperparams),
    }

    if architecture not in model_dict:
        raise ValueError(f'{architecture} is not implemented as prediction network for now.')
    
    return model_dict[architecture]()


def get_prediction_network(hyperparams):
    architecture = hyperparams.pred_model_type
    model_hyperparams = {
            "num_layers": hyperparams.num_layers,
            "hidden_dim": hyperparams.hidden_dim,
            "input_dim": hyperparams.input_dim,
            "in_node_nf": hyperparams.in_node_nf,
            "in_edge_nf": hyperparams.in_edge_nf,
        }
    model_hyperparams = OmegaConf.create(dict(model_hyperparams))
    model_dict = {
            "GNN": lambda: GNN(model_hyperparams),
            "EGNN": lambda: EGNN_vel(model_hyperparams),
            "vndeepsets": lambda: VNDeepSets(model_hyperparams),
            "Transformer": lambda: Transformer(model_hyperparams),
        }

    if architecture not in model_dict:
        raise ValueError(f'{architecture} is not implemented as prediction network for now.')
    
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