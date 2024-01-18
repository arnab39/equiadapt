import torch

class BaseWrapper(torch.nn.Module):
    def __init__(self, canonicalization_network, **kwargs):
        super().__init__()
        self.canonicalization_network = canonicalization_network
        self.canonicalization_info_dict = {}
       
    def forward(self, x, **kwargs):
        # if the child class has a forward method, then it will be called
        # else the forward method of this class will throw an error
        canonicalized_x, canonicalization_info_dict = self.canonize(x, **kwargs)
        
        self.canonicalization_info_dict = canonicalization_info_dict
        
        return canonicalized_x
    
    def invert
        
    
    def canonize(self, x, kwargs):
        return self.canonicalization_network(x, kwargs)
    
    def add_prior_regularizer(self, loss):
        return loss + self.get_prior_regularization_loss()
    
    def get_prior_regularization_loss(self):
        pass
    

# Idea for the user interface:

# 1. The user creates a canonicalization network or uses our provided networks
#    and a wrap it using equiadapt wrappers.
# example:  canonicalization_network = ESCNNEquivariantNetwork(in_shape, out_channels, kernel_size, group_type='rotation', num_rotations=4, num_layers=3)
#           canonicalizer = GroupEquivariantImageCanonicalization(canonicalization_network, beta=1.0)
#           wrapper = ImageEquiWrapper(canonicalizer)
# 
# 2. The user uses this wrapper with their code
#    example: model = ResNet18()
#             x_canonized = wrapper(x)
#             model_out = model(x_canonized)
#             loss = criterion(model_out, y)
#             loss = wrapper.add_prior_regularizer(loss)
#             loss.backward()
# 3. The user creates a loss function and a wrapper for it.
    
    