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
- Run `source .env` on terminal

# Running Instructions
For image classification: [here](/examples/images/classification/README.md)


# Related papers

For more insights on this library refer to our original paper on the idea: [Equivariance with Learned Canonicalization Function](https://proceedings.mlr.press/v202/kaba23a.html) and how to extend it to make any existing large pre-trained model equivariant: [Equivariant Adaptation of Large Pretrained Models](https://arxiv.org/abs/2310.01647).

To learn more about this from a blog check out: [How to make your foundation model equivariant](https://mila.quebec/en/article/how-to-make-your-foundation-model-equivariant/)

# Contact

For question related to this code you can mail us at: 
```arnab.mondal@mila.quebec```
```siba-smarak.panigrahi@mila.quebec```
```kabaseko@mila.quebec```