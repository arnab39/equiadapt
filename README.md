# EquivariantAdaptation
Library to make any existing neural network architecture equivariant

# Setup instructions
### Setup Conda environment

To create a conda environment with the necessary packages:

```
conda env create -f conda_env.yaml
conda activate equiadapt
pip install -e .
```

#### For Python 3.10

Currently, everything works in Python 3.8.
But to use Python 3.10, you need to remove `py3nj` from the `escnn` package requirements and install `escnn` from GitHub manually.

```
git clone https://github.com/QUVA-Lab/escnn.git
cd escnn (and go to setup.py and remove py3nj from the requirements)
pip install -e .
```

### Setup Hydra
- Create a `.env` file in the root of the project with the following content:
  ```
    export HYDRA_JOBS="/path/to/your/hydra/jobs/directory"
    export WANDB_DIR="/path/to/your/wandb/jobs/directory"
    export WANDB_CACHE_DIR="/path/to/your/wandb/cache/directory"
    export DATA_PATH="/path/to/your/data/directory"
    export CHECKPOINT_PATH="/path/to/your/checkpoint/directory"
  ```


# Running Instructions
For image classification: [here](/examples/images/classification/README.md)
For image segmentation: [here](/examples/images/segmentation/README.md)


# Related papers

For more insights on this library refer to our original paper on the idea: [Equivariance with Learned Canonicalization Function (ICML 2023)](https://proceedings.mlr.press/v202/kaba23a.html) and how to extend it to make any existing large pre-trained model equivariant: [Equivariant Adaptation of Large Pretrained Models (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9d5856318032ef3630cb580f4e24f823-Abstract-Conference.html).

To learn more about this from a blog, check out: [How to make your foundation model equivariant](https://mila.quebec/en/article/how-to-make-your-foundation-model-equivariant/)

# Citation
If you find this library or the associated papers useful, please cite:
```
@inproceedings{kaba2023equivariance,
  title={Equivariance with learned canonicalization functions},
  author={Kaba, S{\'e}kou-Oumar and Mondal, Arnab Kumar and Zhang, Yan and Bengio, Yoshua and Ravanbakhsh, Siamak},
  booktitle={International Conference on Machine Learning},
  pages={15546--15566},
  year={2023},
  organization={PMLR}
}
```

```
@article{mondal2024equivariant,
  title={Equivariant Adaptation of Large Pretrained Models},
  author={Mondal, Arnab Kumar and Panigrahi, Siba Smarak and Kaba, Oumar and Mudumba, Sai Rajeswar and Ravanbakhsh, Siamak},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

# Contact

For questions related to this code, you can mail us at:
```arnab.mondal@mila.quebec```
```siba-smarak.panigrahi@mila.quebec```
```kabaseko@mila.quebec```

# Contributing

You can check out the [contributor's guide](CONTRIBUTING.md).

This project uses `pre-commit`_, you can install it before making any
changes::

    pip install pre-commit
    cd equiadapt
    pre-commit install

It is a good idea to update the hooks to the latest version::

    pre-commit autoupdate
