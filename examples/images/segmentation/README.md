# Commands to run instance segmentation experiments

### For instance segmentation (with prior regularization)
```
python train.py canonicalization=group_equivariant canonicalization.network_type=equivariant_wrn \
dataset.dataset_name=coco dataset.img_size=512 \
prediction.prediction_network_architecture=maskrcnn prediction.prediction_network_architecture_type=resnet50_fpn_v2
```
### For instance segmentation (without prior regularization)
```
python train.py canonicalization=group_equivariant canonicalization.network_type=equivariant_wrn \
dataset.dataset_name=coco dataset.img_size=512 \
prediction.prediction_network_architecture=maskrcnn prediction.prediction_network_architecture_type=resnet50_fpn_v2 \
experiment.training.loss.prior_weight=0
```

**Note**: You can also run the `train.py` as follows from the root directory of the project:
```
python examples/images/segmentation/train.py canonicalization=group_equivariant
```

## Important Hyperparameters
We use `hydra` and `OmegaConf` to setup experiments and parse configs. All the config files are available in [`/configs`](configs), along with the meaning of the hyperparameters in each yaml file. Below, we highlight some important details:
- Choose canonicalization type from [`here`](configs/canonicalization) and set with `canonicalizaton=group_equivariant`
- Canonicalization network architecture and relevant hyperparameters are detailed within canonicalization configs
- Dataset settings can be found [`here`](configs/dataset) and set with `dataset.dataset_name=coco`
- Experiment settings can be found [`here`](configs/experiment) and set with `experiment.inference.num_rotations=8`
- Prediction architecture settings can be found [`here`](configs/prediction) and set with `prediction.prediction_network_architecture=maskrcnn`
- Wandb logging settings can can be found [`here`](configs/wandb) and set with `wandb.use_wandb=1`
