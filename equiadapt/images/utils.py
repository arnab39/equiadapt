import torch

def roll_by_gather(feature_map, shifts):
    device = shifts.device
    # assumes 2D array
    batch, channel, group, x_dim, y_dim = feature_map.shape
    arange1 = torch.arange(group).view((1, 1, group, 1, 1)).repeat((batch, channel, 1, x_dim, y_dim)).to(device)
    arange2 = (arange1 - shifts[:, None, None,None,None].long()) % group
    return torch.gather(feature_map, 2, arange2)