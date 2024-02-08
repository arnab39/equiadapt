from typing import Dict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from model import ImageClassifierPipeline
from common import ConfigDict

from prepare import RotatedMNISTDataModule, CIFAR10DataModule, CIFAR100DataModule, STL10DataModule, Flowers102DataModule, CelebADataModule, ImageNetDataModule
    
def get_model_data_and_callbacks(hyperparams : ConfigDict):
    
     # get image data
    image_data = get_image_data(hyperparams.dataset)
    
    # checkpoint name
    hyperparams.checkpoint.checkpoint_name = get_checkpoint_name(hyperparams)
    
    # checkpoint callbacks
    callbacks = get_callbacks(hyperparams)

    # get model pipeline 
    model = get_model_pipeline(hyperparams)
    
    return  model, image_data, callbacks

def get_model_pipeline(hyperparams: ConfigDict):

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
    hyperparams: ConfigDict,
    callbacks: list,
    wandb_logger: pl.loggers.WandbLogger
):
    if hyperparams.experiment.run_mode == "auto_tune":
        trainer = pl.Trainer(
            max_epochs=hyperparams.experiment.num_epochs, accelerator="auto", 
            auto_scale_batch_size=True, auto_lr_find=True, logger=wandb_logger, 
            callbacks=callbacks, deterministic=hyperparams.experiment.deterministic,
            num_nodes=hyperparams.experiment.num_nodes, devices=hyperparams.experiment.num_gpus, 
            strategy='ddp'
        )
        
    elif hyperparams.experiment.run_mode == "dryrun":
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
    
    
def get_callbacks(hyperparams: ConfigDict):
    
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

def get_recursive_hyperparams_identifier(hyperparams: Dict):
    # get the identifier for the canonicalization network hyperparameters
    # recursively go through the dictionary and get the values and concatenate them
    identifier = ""
    for key, value in hyperparams.items():
        if isinstance(value, dict):
            identifier += f"_{get_recursive_hyperparams_identifier(value)}_"
        else:
            identifier += f"_{key}_{value}_"
    return identifier
    
def get_checkpoint_name(hyperparams : ConfigDict):
    
    return f"{get_recursive_hyperparams_identifier(hyperparams.canonicalization.to_dict())}" + \
                          f"__seed_{hyperparams.experiment.seed}"
                        

def get_image_data(dataset_hyperparams: ConfigDict):
    
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