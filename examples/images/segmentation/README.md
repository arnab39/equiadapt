# Commands to run instance segmentation experiments

## For COCO
### For instance segmentation (with prior regularization)
``` 
python train.py canonicalization=group_equivariant canonicalization.network_type=equivariant_wrn dataset.img_size=512 prediction.prediction_network_architecture=maskrcnn prediction.prediction_network_architecture_type=resnet50_fpn_v2
```
### For instance segmentation (without prior regularization)
```
python train.py canonicalization=group_equivariant canonicalization.network_type=equivariant_wrn dataset.img_size=512 prediction.prediction_network_architecture=maskrcnn prediction.prediction_network_architecture_type=resnet50_fpn_v2 experiment.training.loss.prior_weight=0 
```

**Note**: You can also run the `train.py` as follows from root directory of the project: 
```
python examples/images/segmentation/train.py canonicalization=group_equivariant
```
