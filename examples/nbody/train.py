# type: ignore
import os

import pytorch_lightning as pl
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from examples.nbody.model import NBodyPipeline
from examples.nbody.prepare.nbody_data import NBodyDataModule

HYPERPARAMS = {
    "model": "NBodyPipeline",
    "canon_model_type": "vndeepsets",
    "pred_model_type": "Transformer",
    "batch_size": 100,
    "dryrun": False,
    "use_wandb": False,
    "checkpoint": False,
    "num_epochs": 1000,
    "num_workers": 0,
    "auto_tune": False,
    "seed": 0,
    "learning_rate": 1e-3,  # 1e-3
    "weight_decay": 1e-12,
    "patience": 1000,
}

CANON_HYPERPARAMS = {
    "architecture": "vndeepsets",
    "num_layers": 4,
    "hidden_dim": 16,
    "layer_pooling": "mean",
    "final_pooling": "mean",
    "out_dim": 4,
    "batch_size": 100,
    "nonlinearity": "relu",
    "canon_feature": "p",
    "canon_translation": False,
    "angular_feature": "pv",
    "dropout": 0.5,
}

PRED_HYPERPARAMS = {
    "architecture": "GNN",
    "num_layers": 4,
    "hidden_dim": 32,
    "input_dim": 6,
    "in_node_nf": 1,
    "in_edge_nf": 2,
    "nheads": 8,
    "ff_hidden": 32,
}

HYPERPARAMS["canon_hyperparams"] = CANON_HYPERPARAMS
HYPERPARAMS["pred_hyperparams"] = PRED_HYPERPARAMS


def train_nbody() -> None:
    hyperparams = HYPERPARAMS  # type: ignore

    if not hyperparams["use_wandb"]:
        print("Wandb disable for logging.")
        os.environ["WANDB_MODE"] = "disabled"
    else:
        print("Using wandb for logging.")
        os.environ["WANDB_MODE"] = "online"

    wandb.login()
    wandb.init(config=hyperparams, entity="symmetry_group", project="equiadapt")
    wandb_logger = WandbLogger(project="equiadapt")

    hyperparams = wandb.config
    pl.seed_everything(hyperparams.seed)
    nbody_data = NBodyDataModule(hyperparams)

    checkpoint_callback = ModelCheckpoint(
        dirpath=".",
        filename=hyperparams.model + "_" + wandb.run.name + "_{epoch}_{valid/loss:.3f}",
        monitor="valid/loss",
        mode="min",
    )
    early_stop_metric_callback = EarlyStopping(
        monitor="valid/loss", min_delta=0.0, patience=600, verbose=True, mode="min"
    )
    # early_stop_lr_callback = EarlyStopping(monitor="lr", min_delta=0.0, patience=10000, verbose=True, mode="min", stopping_threshold=1.1e-6)
    callbacks = (
        [checkpoint_callback, early_stop_metric_callback]
        if hyperparams.checkpoint
        else [early_stop_metric_callback]
    )

    model = NBodyPipeline(OmegaConf.create(dict(hyperparams)))

    if hyperparams.auto_tune:
        trainer = pl.Trainer(
            fast_dev_run=hyperparams.dryrun,
            max_epochs=hyperparams.num_epochs,
            accelerator="auto",
            auto_scale_batch_size=True,
            auto_lr_find=True,
            logger=wandb_logger,
            callbacks=callbacks,
            deterministic=False,
        )
        trainer.tune(
            model, datamodule=nbody_data, enable_checkpointing=hyperparams.checkpoint
        )
    elif hyperparams.dryrun:
        trainer = pl.Trainer(
            fast_dev_run=False,
            max_epochs=2,
            accelerator="auto",
            limit_train_batches=10,
            limit_val_batches=10,
            logger=wandb_logger,
            callbacks=callbacks,
            deterministic=False,
            enable_checkpointing=hyperparams.checkpoint,
            log_every_n_steps=30,
        )
    else:
        trainer = pl.Trainer(
            fast_dev_run=hyperparams.dryrun,
            max_epochs=hyperparams.num_epochs,
            accelerator="auto",
            logger=wandb_logger,
            callbacks=callbacks,
            deterministic=False,
            enable_checkpointing=hyperparams.checkpoint,
            log_every_n_steps=30,
        )

    trainer.fit(model, datamodule=nbody_data)


def main() -> None:
    train_nbody()


if __name__ == "__main__":
    train_nbody()
