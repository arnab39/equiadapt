import torch 
from omegaconf import DictConfig

from equiadapt.common.basecanonicalization import IdentityCanonicalization
from equiadapt.images.canonicalization.discrete_group import GroupEquivariantImageCanonicalization, OptimizedGroupEquivariantImageCanonicalization
from equiadapt.images.canonicalization.continuous_group import SteerableImageCanonicalization, OptimizedSteerableImageCanonicalization
from equiadapt.images.canonicalization_networks import ESCNNEquivariantNetwork, ConvNetwork, CustomEquivariantNetwork, ESCNNSteerableNetwork

def get_canonicalization_network(
    canonicalization_type: str,
    canonicalization_hyperparams: DictConfig,
    in_shape: tuple
):
    """
    The function returns the canonicalization network based on the canonicalization type

    Args:
        canonicalization_type (str): defines the type of canonicalization network
        options are 1) group_equivariant 2) steerable 3) opt_group_equivariant 4) opt_steerable
    """
    if canonicalization_type == 'identity':
        return torch.nn.Identity()
    
    canonicalization_network_dict = {
        'group_equivariant': {
            'escnn': ESCNNEquivariantNetwork,
            'custom': CustomEquivariantNetwork,
        },
        'steerable': {
            'escnn': ESCNNSteerableNetwork,
        },
        'opt_group_equivariant':{
            'cnn': ConvNetwork,
        },
        'opt_steerable': {
            'cnn': ConvNetwork,
        }
    }
    
    if canonicalization_type not in canonicalization_network_dict:
        raise ValueError(f'{canonicalization_type} is not implemented')   
    if canonicalization_hyperparams.network_type not in canonicalization_network_dict[canonicalization_type]:
        raise ValueError(f'{canonicalization_hyperparams.network_type} is not implemented for {canonicalization_type}')
    
    canonicalization_network = \
    canonicalization_network_dict[canonicalization_type][
        canonicalization_hyperparams.network_type
        ](
           in_shape = in_shape, 
           **canonicalization_hyperparams.network_hyperparams
        )
    
    return canonicalization_network


def get_canonicalizer(
    canonicalization_type: str,
    canonicalization_network: torch.nn.Module,
    canonicalization_hyperparams: DictConfig,
    in_shape: tuple
):
    """
    The function returns the canonicalization network based on the canonicalization type

    Args:
        canonicalization_type (str): defines the type of canonicalization network
        options are 1) group_equivariant 2) steerable 3) opt_group_equivariant 4) opt_steerable
    """
    if canonicalization_type == 'identity':
        return IdentityCanonicalization(canonicalization_network)
    
    canonicalizer_dict = {
        'group_equivariant': GroupEquivariantImageCanonicalization,
        'steerable': SteerableImageCanonicalization,
        'opt_group_equivariant': OptimizedGroupEquivariantImageCanonicalization,
        'opt_steerable': OptimizedSteerableImageCanonicalization
    }
    
    if canonicalization_type not in canonicalizer_dict:
        raise ValueError(f'{canonicalization_type} needs a canonicalization network implementation.')
    
    canonicalizer = canonicalizer_dict[canonicalization_type](
        canonicalization_network=canonicalization_network,
        canonicalization_hyperparams=canonicalization_hyperparams,
        in_shape=in_shape
    )
    
    return canonicalizer