import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
import wandb
import os
from omegaconf import DictConfig, OmegaConf
from examples.nbody.model import NBodyPipeline
import torch

from examples.nbody.prepare.nbody_data import NBodyDataModule


CANON_MODEL_HYPERPARAMETERS = {
    "architecture": "vndeepsets",
    "num_layers": 4,
    "hidden_dim": 16,
    "layer_pooling": "sum",
    "final_pooling": "mean",
    "out_dim": 4,
    "batch_size": 100,
    "nonlinearity": "relu",
    "canon_feature": "p",
    "canon_translation": False,
    "angular_feature": "pv",
    "dropout": 0,
}

PRED_MODEL_HYPERPARAMETERS = {
    "architecture": "GNN",
    "num_layers": 4,
    "hidden_dim": 32,
    "input_dim": 6,
    "in_node_nf": 1,
    "in_edge_nf": 2,
}

HYPERPARAMS = {
    "model": "nbody",
    "learning_rate": 1e-3,
    "weight_decay": 1e-12,
    "patience": 1000,
    "batch_size": 100, 
    "dryrun": False, 
    "use_wandb": False, 
    "checkpoint": False, 
    "num_epochs": 1000, 
    "num_workers":11, 
    "auto_tune":False, 
    "seed": 0,
    "canon_hyperparams": CANON_MODEL_HYPERPARAMETERS,
    "pred_hyperparams": PRED_MODEL_HYPERPARAMETERS,
}

def train_nbody():
    hyperparams = HYPERPARAMS

    if not hyperparams["use_wandb"]:
        print('Wandb disable for logging.')
        os.environ["WANDB_MODE"] = "disabled"
    else:
        print('Using wandb for logging.')
        os.environ["WANDB_MODE"] = "online"
        
    wandb.login()
    wandb.init(config=hyperparams, entity="symmetry_group", project="canonical_network-nbody-transformer")
    wandb_logger = WandbLogger(project="canonical_network-nbody-transformer")

    hyperparams = wandb.config
    pl.seed_everything(hyperparams.seed)
    nbody_data = NBodyDataModule(hyperparams)

    checkpoint_callback = ModelCheckpoint(dirpath=".", filename= hyperparams.model + "_" + wandb.run.name + "_{epoch}_{valid/loss:.3f}", monitor="valid/loss", mode="min")
    early_stop_metric_callback = EarlyStopping(monitor="valid/loss", min_delta=0.0, patience=600, verbose=True, mode="min")
    # early_stop_lr_callback = EarlyStopping(monitor="lr", min_delta=0.0, patience=10000, verbose=True, mode="min", stopping_threshold=1.1e-6)
    callbacks = [checkpoint_callback, early_stop_metric_callback] if hyperparams.checkpoint else [early_stop_metric_callback]

    model = NBodyPipeline(OmegaConf.create(dict(hyperparams)))

    if hyperparams.auto_tune:
        trainer = pl.Trainer(fast_dev_run=hyperparams.dryrun, max_epochs=hyperparams.num_epochs, accelerator="auto", auto_scale_batch_size=True, auto_lr_find=True, logger=wandb_logger, callbacks=callbacks, deterministic=False)
        trainer.tune(model, datamodule=nbody_data, enable_checkpointing=hyperparams.checkpoint)
    elif hyperparams.dryrun:
        trainer = pl.Trainer(fast_dev_run=False, max_epochs=2, accelerator="auto", limit_train_batches=10, limit_val_batches=10, logger=wandb_logger, callbacks=callbacks, deterministic=False, enable_checkpointing=hyperparams.checkpoint, log_every_n_steps=30)
    else:
        trainer = pl.Trainer(fast_dev_run=hyperparams.dryrun, max_epochs=hyperparams.num_epochs, accelerator="auto", logger=wandb_logger, callbacks=callbacks, deterministic=False, enable_checkpointing=hyperparams.checkpoint, log_every_n_steps=30)
    
    
    trainer.fit(model, datamodule=nbody_data)


def main():
    train_nbody()

if __name__ == "__main__":
    train_nbody()