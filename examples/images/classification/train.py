import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from examples.images.classification.train_utils import get_model_data_and_callbacks, get_trainer
import wandb
from argparse import ArgumentParser
import os
import torch
import time
from tqdm import tqdm
import yaml

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_config_name", type=str, default='group_equivariant', help="pass the config name of the model you want to use")
    args = parser.parse_args()
    return args


def train_images():
    args = get_args()
    
    model_config = './configs/' + args.model_config_name + '.yaml'
    hyperparams['canonicalization_type'] = args.model_config_name
    
    # Load hyperparameters from YAML file
    with open(model_config, 'r') as f:
        hyperparams = yaml.safe_load(f)
    
    hyperparams['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    hyperparams['data']['data_path'] = hyperparams['data']['data_path'] + "/" + hyperparams['dataset_name']
    hyperparams['checkpoint']['checkpoint_path'] = hyperparams['checkpoint']['checkpoint_path'] + "/" + \
                                hyperparams['dataset_name'] + "/" + hyperparams['canonicalization_type'] \
                                + "/" + hyperparams['prediction']['prediction_network_architecture']

    # set system environment variables for wandb
    os.environ["WANDB_MODE"] = "online" if hyperparams['use_wandb'] else "disabled"
    os.environ["WANDB_CACHE_DIR"] = hyperparams['wandb_cache_dir']
    os.environ["WANDB_DIR"] = hyperparams['wandb_dir']
    
    # initialize wandb
    wandb.init(config=hyperparams, entity=hyperparams['wandb']['wandb_entity'], project=hyperparams['wandb']['wandb_project'], dir=hyperparams['wandb']['wandb_dir'])
    wandb_logger = WandbLogger(project=hyperparams['wandb']['wandb_project'], log_model="all")
    hyperparams = wandb.config

    # set seed
    pl.seed_everything(hyperparams.experiment.seed)
    
    # get model, callbacks, and image data
    model, image_data, callbacks = get_model_data_and_callbacks(hyperparams)
        
    if hyperparams.canonicalization_type in ("group_equivariant", "opt_equivariant", "steerable"):
        wandb.watch(model.canonicalization_network, log='all')

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