from typing import Optional

import dotenv
import pytorch_lightning as pl
from model import PointcloudClassificationPipeline
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def get_model_pipeline(hyperparams: DictConfig) -> pl.LightningModule:

    if hyperparams.experiment.run_mode == "test":
        model = PointcloudClassificationPipeline.load_from_checkpoint(
            checkpoint_path=hyperparams.checkpoint.checkpoint_path
            + "/"
            + hyperparams.checkpoint.checkpoint_name
            + ".ckpt",
            hyperparams=hyperparams,
        )
        model.freeze()
        model.eval()
    else:
        model = PointcloudClassificationPipeline(hyperparams)

    return model


def get_trainer(
    hyperparams: DictConfig, callbacks: list, wandb_logger: pl.loggers.WandbLogger
) -> pl.Trainer:
    if hyperparams.experiment.run_mode == "dryrun":
        trainer = pl.Trainer(
            fast_dev_run=5,
            max_epochs=hyperparams.experiment.training.num_epochs,
            accelerator="auto",
            limit_train_batches=5,
            limit_val_batches=5,
            logger=wandb_logger,
            callbacks=callbacks,
            deterministic=hyperparams.experiment.deterministic,
        )
    else:
        trainer = pl.Trainer(
            max_epochs=hyperparams.experiment.training.num_epochs,
            accelerator="auto",
            logger=wandb_logger,
            callbacks=callbacks,
            deterministic=hyperparams.experiment.deterministic,
            num_nodes=hyperparams.experiment.num_nodes,
            devices=hyperparams.experiment.num_gpus,
            strategy="ddp",
        )

    return trainer


def get_callbacks(hyperparams: DictConfig) -> list:

    checkpoint_callback = ModelCheckpoint(
        dirpath=hyperparams.checkpoint.checkpoint_path,
        filename=hyperparams.checkpoint.checkpoint_name,
        monitor="val/iou",
        mode="max",
        save_on_train_epoch_end=False,
    )
    early_stop_metric_callback = EarlyStopping(
        monitor="val/iou",
        min_delta=hyperparams.experiment.training.min_delta,
        patience=hyperparams.experiment.training.patience,
        verbose=True,
        mode="max",
    )

    return [checkpoint_callback, early_stop_metric_callback]


def get_recursive_hyperparams_identifier(hyperparams: DictConfig) -> str:
    # get the identifier for the canonicalization network hyperparameters
    # recursively go through the dictionary and get the values and concatenate them
    identifier = ""
    for key, value in hyperparams.items():
        if isinstance(value, DictConfig):
            identifier += f"_{get_recursive_hyperparams_identifier(value)}_"  # type: ignore
        else:
            identifier += f"_{key}_{value}_"  # type: ignore
    return identifier


def get_checkpoint_name(hyperparams: DictConfig) -> str:

    return (
        f"{get_recursive_hyperparams_identifier(hyperparams.canonicalization)}".lstrip(
            "_"
        )
        + f"__epochs_{hyperparams.experiment.training.num_epochs}_"
        + f"__seed_{hyperparams.experiment.seed}"
    )


def load_envs(env_file: Optional[str] = None) -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.

    It is possible to define all the system specific variables in the `env_file`.

    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)
