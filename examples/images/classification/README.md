# Commands to run image classification experiments

### For image classification (with prior regularization)
```
python train.py canonicalization=group_equivariant
```
### For image classification (without prior regularization)
```
python train.py canonicalization=group_equivariant experiment.training.loss.prior_weight=0
```

**Note**: You can also run the `train.py` as follows from the root directory of the project:
```
python examples/images/classification/train.py canonicalization=group_equivariant
```

### For testing checkpoints
```
python train.py experiment.run_mode=test dataset.dataset_name=stl10 \
checkpoint.checkpoint_path=/path/of/checkpoint/dir checkpoint.checkpoint_name=<name-of-checkpoint>

```

**Note**:
The final checkpoint that will be loaded during evaluation as follows, hence ensure that the combination of `checkpoint.checkpoint_path` and `checkpoint.checkpoint_name` is correct:
```
    model = ImageClassifierPipeline.load_from_checkpoint(
            checkpoint_path=hyperparams.checkpoint.checkpoint_path + "/" + \
                hyperparams.checkpoint.checkpoint_name + ".ckpt",
            hyperparams=hyperparams
        )

```

## Important Hyperparameters
We use `hydra` and `OmegaConf` to setup experiments and parse configs. All the config files are available in [`/configs`](configs), along with the meaning of the hyperparameters in each yaml file. Below, we highlight some important details:
- Choose canonicalization type from [`here`](configs/canonicalization) and set with `canonicalizaton=group_equivariant`
- Canonicalization network architecture and relevant hyperparameters are detailed within canonicalization configs
- Dataset settings can be found [`here`](configs/dataset) and set with `dataset.dataset_name=cifar10`
- Experiment settings can be found [`here`](configs/experiment) and set with `experiment.inference.num_rotations=8`
- Prediction architecture settings can be found [`here`](configs/prediction) and set with `prediction.prediction_network_architecture=vit`
- Wandb logging settings can can be found [`here`](configs/wandb) and set with `wandb.use_wandb=1`. You have to change the entity to your wandb team.
