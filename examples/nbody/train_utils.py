from typing import Optional

import dotenv
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn

from examples.nbody.model import NBodyPipeline
from examples.nbody.prepare.nbody_data import NBodyDataModule


def get_model_data_and_callbacks(hyperparams: DictConfig) -> tuple:
    hyperparams.canonicalization.network_hyperparams["batch_size"] = (
        hyperparams.dataset.batch_size
    )
    model = get_model_pipeline(hyperparams)
    data = get_nbody_data(hyperparams.dataset)
    callbacks = get_callbacks(hyperparams)

    return model, data, callbacks


def get_model_pipeline(hyperparams: DictConfig) -> pl.LightningModule:
    return NBodyPipeline(hyperparams)


def get_callbacks(hyperparams: DictConfig) -> list:
    checkpoint_callback = ModelCheckpoint(
        dirpath=".",
        filename=hyperparams.checkpoint.checkpoint_name,
        monitor="valid/loss",
        mode="min",
    )
    early_stop_metric_callback = EarlyStopping(
        monitor="valid/loss", min_delta=0.0, patience=600, verbose=True, mode="min"
    )

    return [checkpoint_callback, early_stop_metric_callback]


def get_nbody_data(dataset_hyperparams: DictConfig) -> NBodyDataModule:
    return NBodyDataModule(dataset_hyperparams)


def get_trainer(
    hyperparams: DictConfig, callbacks: list, wandb_logger: pl.loggers.WandbLogger
) -> pl.Trainer:
    if hyperparams.experiment.run_mode == "dryrun":
        trainer = pl.Trainer(
            fast_dev_run=False,
            max_epochs=2,
            accelerator="auto",
            limit_train_batches=10,
            limit_val_batches=10,
            logger=wandb_logger,
            callbacks=callbacks,
            deterministic=False,
            enable_checkpointing=hyperparams.experiment.checkpoint,
            log_every_n_steps=30,
        )
    else:
        trainer = pl.Trainer(
            max_epochs=hyperparams.experiment.num_epochs,
            accelerator="auto",
            logger=wandb_logger,
            callbacks=callbacks,
            deterministic=False,
            enable_checkpointing=hyperparams.experiment.checkpoint,
            log_every_n_steps=30,
        )

    return trainer


def load_envs(env_file: Optional[str] = None) -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.

    It is possible to define all the system specific variables in the `env_file`.

    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)
