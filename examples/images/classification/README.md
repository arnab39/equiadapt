# Commands to run image classification experiments

## For Rotated MNIST
### For image classification (without prior regularization)
```
python train.py canonicalization=group_equivariant experiment.training.loss.prior_weight=0
```
### For image classification (with prior regularization)
``` 
python train.py canonicalization=group_equivariant  
```

**Note**: You can also run the `train.py` as follows from root directory of the project: 
```
python examples/images/classification/train.py canonicalization=group_equivariant
```