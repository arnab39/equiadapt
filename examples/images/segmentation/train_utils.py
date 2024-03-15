from typing import Optional

import dotenv
import pytorch_lightning as pl
from model import ImageSegmentationPipeline
from omegaconf import DictConfig
from prepare import COCODataModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def get_model_data_and_callbacks(hyperparams: DictConfig) -> tuple:

    # get image data
    image_data = get_image_data(hyperparams.dataset)

    # checkpoint callbacks
    callbacks = get_callbacks(hyperparams)

    # get model pipeline
    model = get_model_pipeline(hyperparams)

    return model, image_data, callbacks


def get_model_pipeline(hyperparams: DictConfig) -> pl.LightningModule:

    if hyperparams.experiment.run_mode == "test":
        model = ImageSegmentationPipeline.load_from_checkpoint(
            checkpoint_path=hyperparams.checkpoint.checkpoint_path
            + "/"
            + hyperparams.checkpoint.checkpoint_name
            + ".ckpt",
            hyperparams=hyperparams,
            strict=hyperparams.checkpoint.strict_loading,
        )
        model.freeze()
        model.eval()
    else:
        model = ImageSegmentationPipeline(hyperparams)

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
            strategy=(
                "ddp"
                if not hyperparams.experiment.training.loss.task_weight
                else "ddp_find_unused_parameters_true"
            ),
            # since when you do a forward pass through the (large) prediction network (such as Segment-Anything Model)
            # there might be some unused parameters in the prediction network, so we need to set the strategy to
            # ddp_find_unused_parameters_true
        )

    return trainer


def get_callbacks(hyperparams: DictConfig) -> list:

    checkpoint_callback = ModelCheckpoint(
        dirpath=hyperparams.checkpoint.checkpoint_path,
        filename=hyperparams.checkpoint.checkpoint_name,
        monitor="val/map",
        mode="max",
        save_on_train_epoch_end=False,
    )
    early_stop_metric_callback = EarlyStopping(
        monitor="val/map",
        min_delta=hyperparams.experiment.training.min_delta,
        patience=hyperparams.experiment.training.patience,
        verbose=True,
        mode="max",
    )

    return [checkpoint_callback, early_stop_metric_callback]


def get_image_data(dataset_hyperparams: DictConfig) -> pl.LightningDataModule:

    dataset_classes = {"coco": COCODataModule}

    if dataset_hyperparams.dataset_name not in dataset_classes:
        raise ValueError(f"{dataset_hyperparams.dataset_name} not implemented")

    return dataset_classes[dataset_hyperparams.dataset_name](dataset_hyperparams)


def load_envs(env_file: Optional[str] = None) -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.

    It is possible to define all the system specific variables in the `env_file`.

    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)
