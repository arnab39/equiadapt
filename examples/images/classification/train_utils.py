from typing import Dict, Optional

import dotenv
import pytorch_lightning as pl
from model import ImageClassifierPipeline
from omegaconf import DictConfig
from prepare import (
    CelebADataModule,
    CIFAR10DataModule,
    CIFAR100DataModule,
    Flowers102DataModule,
    ImageNetDataModule,
    RotatedMNISTDataModule,
    STL10DataModule,
)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def get_model_data_and_callbacks(hyperparams : DictConfig):
    
     # get image data
    image_data = get_image_data(hyperparams.dataset)
    
    # checkpoint name
    hyperparams.checkpoint.checkpoint_name = get_checkpoint_name(hyperparams)
    
    # checkpoint callbacks
    callbacks = get_callbacks(hyperparams)

    # get model pipeline 
    model = get_model_pipeline(hyperparams)
    
    return  model, image_data, callbacks

def get_model_pipeline(hyperparams: DictConfig):

    if hyperparams.experiment.run_mode == "test":
        model = ImageClassifierPipeline.load_from_checkpoint(
            checkpoint_path=hyperparams.checkpoint.checkpoint_path + "/" + \
                hyperparams.checkpoint.checkpoint_name + ".ckpt",
            hyperparams=hyperparams
        )
        model.freeze()
        model.eval()
    else:
        model = ImageClassifierPipeline(hyperparams)
        
    return model

def get_trainer(
    hyperparams: DictConfig,
    callbacks: list,
    wandb_logger: pl.loggers.WandbLogger
):
    if hyperparams.experiment.run_mode == "dryrun":
        trainer = pl.Trainer(
            fast_dev_run=5, max_epochs=hyperparams.experiment.training.num_epochs, accelerator="auto", 
            limit_train_batches=5, limit_val_batches=5, logger=wandb_logger, 
            callbacks=callbacks, deterministic=hyperparams.experiment.deterministic
        )
    else:
        trainer = pl.Trainer(
            max_epochs=hyperparams.experiment.training.num_epochs, accelerator="auto", 
            logger=wandb_logger, callbacks=callbacks, deterministic=hyperparams.experiment.deterministic,
            num_nodes=hyperparams.experiment.num_nodes, devices=hyperparams.experiment.num_gpus, 
            strategy='ddp'
        )

    return trainer
    
    
def get_callbacks(hyperparams: DictConfig):
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=hyperparams.checkpoint.checkpoint_path,
        filename=hyperparams.checkpoint.checkpoint_name,
        monitor="val/acc",
        mode="max",
        save_on_train_epoch_end=False,
    )
    early_stop_metric_callback = EarlyStopping(monitor="val/acc", 
                    min_delta=hyperparams.experiment.training.min_delta, 
                    patience=hyperparams.experiment.training.patience, 
                    verbose=True, 
                    mode="max")
    
    return [checkpoint_callback, early_stop_metric_callback]

def get_recursive_hyperparams_identifier(hyperparams: DictConfig):
    # get the identifier for the canonicalization network hyperparameters
    # recursively go through the dictionary and get the values and concatenate them
    identifier = ""
    for key, value in hyperparams.items():
        if isinstance(value, DictConfig):
            identifier += f"{get_recursive_hyperparams_identifier(value)}"
        # special manipulation for the keys (to avoid exceeding OS limit for file names)    
        elif key not in ["canonicalization_type", "beta", "input_crop_ratio"]:
            if key == "network_type":
                identifier += f"_net_type_{value}_"
            elif key == "out_vector_size":
                identifier += f"_out_vec_{value}_"
            elif key in ["kernel_size", "resize_shape", "group_type", "artifact_err_wt"]:
                identifier += f"_{key.split('_')[0]}_{value}_"
            elif key in ["num_layers", "out_channels", "num_rotations"]:
                identifier += f"_{key.split('_')[-1]}_{value}_"
            else:
                identifier += f"_{key}_{value}_"
    return identifier
    
def get_checkpoint_name(hyperparams : DictConfig):
    return f"{get_recursive_hyperparams_identifier(hyperparams.canonicalization)}".lstrip("_") + \
                        f"_loss_wts_{int(hyperparams.experiment.training.loss.task_weight)}_{int(hyperparams.experiment.training.loss.prior_weight)}_{int(hyperparams.experiment.training.loss.group_contrast_weight)}" + \
                        f"_lrs_{hyperparams.experiment.training.prediction_lr}_{hyperparams.experiment.training.canonicalization_lr}" + \
                        f"_seed_{hyperparams.experiment.seed}"
                        

def get_image_data(dataset_hyperparams: DictConfig):
    
    dataset_classes = {
        "rotated_mnist": RotatedMNISTDataModule,
        "cifar10": CIFAR10DataModule,
        "cifar100": CIFAR100DataModule,
        "stl10": STL10DataModule,
        "celeba": CelebADataModule,
        "flowers102": Flowers102DataModule,
        "imagenet": ImageNetDataModule
    }
    
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