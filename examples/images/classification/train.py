import os
import torch
import yaml
import wandb
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from train_utils import get_model_data_and_callbacks, get_trainer
from common import ConfigDict

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_config", type=str, default='./configs/group_equivariant.yaml', help="pass the config path of the model you want to use")
    parser.add_argument("--run_mode", type=str, default='train', help="pass the run mode")
    parser.add_argument("--use_wandb", type=bool, default=False, help="pass True if you want to use wandb for logging")
    args = parser.parse_args()
    return args

def update_config(config, hyperparams):
    for key, value in config.items():
        if key in hyperparams:
            config[key] = hyperparams[key]
        elif isinstance(value, dict):
            update_config(config[key], hyperparams)
        else:
            continue


def train_images():
    args = get_args()
    
    # Load hyperparameters from YAML file
    with open(args.model_config, 'r') as f:
        hyperparams = yaml.safe_load(f)
    
    hyperparams['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    hyperparams['dataset']['data_path'] = hyperparams['dataset']['data_path'] + "/" + hyperparams['dataset']['dataset_name']
    hyperparams['checkpoint']['checkpoint_path'] = hyperparams['checkpoint']['checkpoint_path'] + "/" + \
                                hyperparams['dataset']['dataset_name'] + "/" + hyperparams['canonicalization_type'] \
                                + "/" + hyperparams['prediction']['prediction_network_architecture']

    # Override hyperparameters with command line arguments
    update_config(hyperparams, vars(args))
    
    # set system environment variables for wandb
    if hyperparams['wandb']['use_wandb']:
        print("Using wandb for logging...")
        os.environ["WANDB_MODE"] = "online"
    else:
        print("Wandb disabled for logging...")
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DIR"] = hyperparams['wandb']['wandb_dir']
    os.environ["WANDB_CACHE_DIR"] = hyperparams['wandb']['wandb_cache_dir']   
    
    # initialize wandb
    wandb.init(config=hyperparams, entity=hyperparams['wandb']['wandb_entity'], project=hyperparams['wandb']['wandb_project'], dir=hyperparams['wandb']['wandb_dir'])
    wandb_logger = WandbLogger(project=hyperparams['wandb']['wandb_project'], log_model="all")
    
    # Change to ConfigDict to access hyperparameters as attributes
    hyperparams = ConfigDict(hyperparams)
    

    # set seed
    pl.seed_everything(hyperparams.experiment.seed)
    
    # get model, callbacks, and image data
    model, image_data, callbacks = get_model_data_and_callbacks(hyperparams)
        
    if hyperparams.canonicalization_type in ("group_equivariant", "opt_equivariant", "steerable"):
        wandb.watch(model.canonicalizer.canonicalization_network, log='all')

    # get trainer
    trainer = get_trainer(hyperparams, callbacks, wandb_logger)

    if hyperparams.experiment.run_mode == "train":
        trainer.fit(model, datamodule=image_data)
        
    elif hyperparams.experiment.run_mode == "auto_tune":
        trainer.tune(model, datamodule=image_data)

    trainer.test(model, datamodule=image_data)

def main():
    train_images()


if __name__ == "__main__":
    main()