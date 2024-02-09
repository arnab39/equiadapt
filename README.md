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
  ```  
- Run `source .env` on terminal

# Running Instructions
### For image classification (without prior regularization)
```
python examples/images/classification/train.py canonicalization_type=group_equivariant
```
### For image classification (with prior regularization)
``` 
python examples/images/classification/train.py canonicalization_type=group_equivariant experiment.training.loss.prior_weight=0 
```

**Note**: You can also run the `train.py` as follows: 
```
cd examples/images/classification  
python train.py canonicalization_type=group_equivariant
```
