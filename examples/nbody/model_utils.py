from equiadapt.nbody.canonicalization_networks.euclideangraph_base_models import EGNN_vel, VNDeepSets, GNN, Transformer
from collections import namedtuple

def get_canonicalization_network(hyperparams):
    architecture = hyperparams.canon_model_type
    model_dict = {
    "EGNN": lambda: EGNN_vel(hyperparams),
    "vndeepsets": lambda: VNDeepSets(hyperparams),
    }

    if architecture not in model_dict:
        raise ValueError(f'{architecture} is not implemented as prediction network for now.')
    
    return model_dict[architecture]()

def get_prediction_network(hyperparams):
    architecture = hyperparams.pred_model_type
    model_dict = {
            "GNN": lambda: GNN(hyperparams),
            "EGNN": lambda: EGNN_vel(hyperparams),
            "vndeepsets": lambda: VNDeepSets(hyperparams),
            "Transformer": lambda: Transformer(hyperparams),
        }

    if architecture not in model_dict:
        raise ValueError(f'{architecture} is not implemented as prediction network for now.')
    
    return model_dict[architecture]()