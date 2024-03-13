# Commands to run pointcloud classification experiments on ModelNet

### For pointcloud classification (with prior regularization)
```
python train.py canonicalization=group_equivariant
```
### For pointcloud classification (without prior regularization)
```
python train.py canonicalization=group_equivariant experiment.training.loss.prior_weight=0
```

**Note**: You can also run the `train.py` as follows from the root directory of the project:
```
python examples/pointcloud/classification/train.py canonicalization=group_equivariant
```

## Important Hyperparameters
We use `hydra` and `OmegaConf` to setup experiments and parse configs. All the config files are available in [`/configs`](configs), along with the meaning of the hyperparameters in each yaml file. Below, we highlight some important details:
- Choose canonicalization type from [`here`](configs/canonicalization) and set with `canonicalizaton=group_equivariant`. If you select it to be `canonicalizaton=identity` you will run just the prediction network which is useful for the baseline.
- Canonicalization network architecture and relevant hyperparameters are detailed within canonicalization configs
- Experiment settings can be found [`here`](configs/experiment). You can select the rotation type of the train, validation and test data. During inference we always want to test with so3. You can change the it with `experiment.training.rotation_type=none` and `experiment.validation.rotation_type=none` if you don't want any rotation.
- Prediction architecture settings (whether dgcnn or pointnet) can be found [`here`](configs/prediction) and set with `prediction.prediction_network_architecture=dgcnn`
- Wandb logging settings can can be found [`here`](configs/wandb) and set with `wandb.use_wandb=1`. You have to change the entity to your wandb team.
