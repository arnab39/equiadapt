# Commands to run N-body experiments

### For default settings (vndeepsets canonicalizer and GNN prediction)
```
python train.py canonicalization=vndeepsets
```

**Note**: You can also run the `train.py` as follows from the root directory of the project:
```
python examples/nbdoy/train.py canonicalization=vndeepsets
```


## Important Hyperparameters
We use `hydra` and `OmegaConf` to setup experiments and parse configs. All the config files are available in [`/configs`](configs), along with the meaning of the hyperparameters in each yaml file. Below, we highlight some important details:
- Choose canonicalization type from [`here`](configs/canonicalization) and set with `canonicalizaton=vndeepsets`
- Canonicalization network architecture and relevant hyperparameters are detailed within canonicalization configs
- Dataset settings can be found [`here`](configs/dataset) and set with `dataset.batch_size=100`
- Experiment settings can be found [`here`](configs/experiment) and set with `experiment.learning_rate=1e-3`
- Prediction architecture settings can be found [`here`](configs/prediction) and set with `prediction.architecture=GNN`
- Wandb logging settings can can be found [`here`](configs/wandb) and set with `wandb.use_wandb=1`. You have to change the entity to your wandb team.
