import os

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from train_utils import get_model_data_and_callbacks, get_trainer, load_envs


def train_images(hyperparams: DictConfig) -> None:

    if hyperparams["experiment"]["run_mode"] == "test":
        assert (
            len(hyperparams["checkpoint"]["checkpoint_name"]) > 0
        ), "checkpoint_name must be provided for test mode"

        existing_ckpt_path = (
            hyperparams["checkpoint"]["checkpoint_path"]
            + "/"
            + hyperparams["checkpoint"]["checkpoint_name"]
            + ".ckpt"
        )
        existing_ckpt = torch.load(existing_ckpt_path)
        conf = OmegaConf.create(existing_ckpt["hyper_parameters"]["hyperparams"])

        hyperparams["canonicalization_type"] = conf["canonicalization_type"]
        hyperparams["canonicalization"] = conf["canonicalization"]
        if hyperparams["checkpoint"]["strict_loading"]:
            hyperparams["prediction"] = conf["prediction"]

    else:
        hyperparams["canonicalization_type"] = hyperparams["canonicalization"][
            "canonicalization_type"
        ]
        hyperparams["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        hyperparams["dataset"]["data_path"] = (
            hyperparams["dataset"]["data_path"]
            + "/"
            + hyperparams["dataset"]["dataset_name"]
        )
        hyperparams["checkpoint"]["checkpoint_path"] = (
            hyperparams["checkpoint"]["checkpoint_path"]
            + "/"
            + hyperparams["dataset"]["dataset_name"]
            + "/"
            + hyperparams["canonicalization_type"]
            + "/"
            + hyperparams["prediction"]["prediction_network_architecture"]
        )

    # set system environment variables for wandb
    if hyperparams["wandb"]["use_wandb"]:
        print("Using wandb for logging...")
        os.environ["WANDB_MODE"] = "online"
    else:
        print("Wandb disabled for logging...")
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DIR"] = hyperparams["wandb"]["wandb_dir"]
    os.environ["WANDB_CACHE_DIR"] = hyperparams["wandb"]["wandb_cache_dir"]

    # initialize wandb
    wandb_run = wandb.init(
        config=OmegaConf.to_container(hyperparams, resolve=True),
        entity=hyperparams["wandb"]["wandb_entity"],
        project=hyperparams["wandb"]["wandb_project"],
        dir=hyperparams["wandb"]["wandb_dir"],
    )
    wandb_logger = WandbLogger(
        project=hyperparams["wandb"]["wandb_project"], log_model="all"
    )

    if not hyperparams["experiment"]["run_mode"] == "test":
        hyperparams["checkpoint"]["checkpoint_name"] = (
            wandb_run.id
            + "_"
            + wandb_run.name
            + "_"
            + wandb_run.sweep_id
            + "_"
            + wandb_run.group
        )

    # set seed
    pl.seed_everything(hyperparams.experiment.seed)

    # get model, callbacks, and image data
    model, image_data, callbacks = get_model_data_and_callbacks(hyperparams)

    if hyperparams.canonicalization_type in (
        "group_equivariant",
        "opt_equivariant",
        "steerable",
    ):
        wandb.watch(model.canonicalizer.canonicalization_network, log="all")

    # get trainer
    trainer = get_trainer(hyperparams, callbacks, wandb_logger)

    if hyperparams.experiment.run_mode in ["train", "dryrun"]:
        trainer.fit(model, datamodule=image_data)

    elif hyperparams.experiment.run_mode == "auto_tune":
        trainer.tune(model, datamodule=image_data)

    trainer.test(model, datamodule=image_data)


# load the variables from .env file
load_envs()


@hydra.main(config_path=str("./configs/"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    train_images(cfg)


if __name__ == "__main__":
    main()
