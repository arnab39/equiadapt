import torch
import torch.nn as nn
import torch.nn.functional as F


from segment_anything import sam_model_registry
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

ALPHA = 0.8
GAMMA = 2

class MaskRCNNModel(nn.Module):
    def __init__(self, architecture_type, num_classes, weights='DEFAULT'):
        super().__init__()
        
        assert architecture_type in ['resnet50_fpn_v2'], NotImplementedError('Only `maskrcnn_resnet50_fpn_v2` is supported for now.')
        if architecture_type == 'resnet50_fpn_v2':
            self.model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')

    def forward(self, images, targets):
        pred_masks = []
        ious = []
        outputs = []
        if self.training:
            loss_dict = self.model(images, targets)
            return loss_dict, None, None, None
        else:
            for _, (image, target) in enumerate(zip(images, targets)):
                output = self.model([image], [target])

                # if the model doesn't predict any labels return the target
                if len(output[0]['labels']) == 0:
                    output[0]['labels'] = target['labels']
                    output[0]['boxes'] = target['boxes']
                    output[0]['masks'] = target['masks']
                    output[0]['scores'] = torch.ones(len(target['masks']))
                    ious.append(torch.ones(len(target['masks']), dtype=torch.float32))
                    pred_masks.append(torch.ones(len(target['masks']), image.shape[-2], image.shape[-1], dtype=torch.float32, device=self.hyperparams.device))

                else:
                    masks = output[0]['masks']
                    iou_predictions = output[0]['scores']
                    pred_masks.append(masks.squeeze(1))
                    ious.append(iou_predictions)

                    output[0]['masks'] = torch.as_tensor(output[0]['masks'].squeeze(1) > 0.5, dtype=torch.uint8).squeeze(1)
                    output[0]['scores'] = torch.as_tensor(output[0]['scores'], dtype=torch.float32)
                    output[0]['labels'] = torch.as_tensor(output[0]['labels'], dtype=torch.int64)
                    output[0]['boxes'] = torch.as_tensor(output[0]['boxes'], dtype=torch.float32)
                outputs.append(output[0])

            return None, pred_masks, ious, outputs

class SAMModel(nn.Module):

    def __init__(self, 
                architecture_type: str,
                sam_pretrained_ckpt_path: str):
        super().__init__()
        assert sam_pretrained_ckpt_path is not None, ValueError('SAM requires a pretrained checkpoint path.')
        self.model = sam_model_registry[architecture_type](checkpoint=sam_pretrained_ckpt_path)

    def forward(self, images, targets):
        if type(images) == list:
            images = torch.stack(images)
        _, _, H, W = images.shape
        image_embeddings = self.model.image_encoder(images)
        pred_masks = []
        ious = []
        outputs = []
        for _, embedding, target in zip(images, image_embeddings, targets):
            bbox = target['boxes']

            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=bbox,
                masks=None,
            )

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(1)) # bbox_length x H x W
            ious.append(iou_predictions) # bbox_length x 1

            output = dict(
                masks = torch.as_tensor(masks.squeeze(1) > 0.5, dtype=torch.uint8),
                scores = torch.as_tensor(iou_predictions.squeeze(1), dtype=torch.float32),
                labels = torch.as_tensor(target['labels'], dtype=torch.int64),
                boxes = torch.as_tensor(target['boxes'], dtype=torch.float32)
            )
            outputs.append(output)

        return None, pred_masks, ious, outputs
    
class FocalLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.name = 'focal_loss'

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
        self.name = 'dice_loss'

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
            'maskrcnn': (None, (3, 512, 512), 91),
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
    num_classes: int = 91,
    sam_pretrained_ckpt_path=None
):
    weights = 'DEFAULT' if use_pretrained else None
    model_dict = {
        'sam': SAMModel(architecture_type, sam_pretrained_ckpt_path),
        'maskrcnn': MaskRCNNModel(architecture_type, num_classes, weights)
    }

    if architecture not in model_dict:
        raise ValueError(f'{architecture} is not implemented as prediction network for now.')

    prediction_network = model_dict[architecture](weights=weights)
    
    if freeze_encoder:
        for param in prediction_network.parameters():
            param.requires_grad = False

    if dataset_name != 'coco' and architecture in ['maskrcnn']:
        # maskrcnn mask predictor needs to be trained on the new number of classes
        for param in prediction_network.roi_heads.mask_predictor.parameters():
            param.requires_grad = True

    return prediction_network

def calc_iou(pred_mask, gt_mask):
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    batch_iou = intersection / union
    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou