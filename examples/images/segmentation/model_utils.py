import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = 0.8
GAMMA = 2


class FocalLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE

        return focal_loss


class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice
    
def get_dataset_specific_info(dataset_name, prediction_architecture_name):
    dataset_info = {
        'coco': {
            'sam': ([FocalLoss(), DiceLoss()], (3, 1024, 1024), 91),
            'maskrcnn': (None, (3, 512, 512), 91), # task-specific loss is evaluated on its own for MaskRCNN
        }
    }

    if dataset_name not in dataset_info:
        raise ValueError('Dataset not implemented for now.')
    if prediction_architecture_name not in dataset_info[dataset_name]:
        raise ValueError('Segmentation architecture not implemented for now.')

    return dataset_info[dataset_name][prediction_architecture_name]

def get_prediction_network(
    architecture: str = 'sam',
    architecture_type: str = 'vit_h',
    dataset_name: str = 'coco',
    use_pretrained: bool = False,
    freeze_encoder: bool = False,
    num_classes: int = 91
):
    weights = 'DEFAULT' if use_pretrained else None
    model_dict = {
        'sam': SAMModel(architecture_type),
        'maskrcnn': MaskRCNNModel(architecture_type, num_classes)
    }

    if architecture not in model_dict:
        raise ValueError(f'{architecture} is not implemented as prediction network for now.')

    prediction_network = model_dict[architecture](weights=weights)
    
    if freeze_encoder:
        for param in prediction_network.parameters():
            param.requires_grad = False

    if dataset_name != 'coco' and architecture == 'maskrcnn':
        for param in prediction_network.roi_heads.mask_predictor.parameters():
            param.requires_grad = True

    return prediction_network