# Commands to run instance segmentation experiments

## For COCO
### For instance segmentation (without prior regularization)
```
python train.py canonicalization=group_equivariant experiment.training.loss.prior_weight=0 
```
### For instance segmentation (with prior regularization)
``` 
python train.py canonicalization=group_equivariant  
```

**Note**: You can also run the `train.py` as follows from root directory of the project: 
```
python examples/images/segmentation/train.py canonicalization=group_equivariant
```
