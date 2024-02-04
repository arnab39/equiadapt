import pytorch_lightning as pl
from typing import Union, Dict
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from examples.images.classification.model import ImageClassifierPipeline

from examples.images.classification.prepare import RotatedMNISTDataModule, CIFAR10DataModule, CIFAR100DataModule, STL10DataModule, Flowers102DataModule, CelebADataModule, ImageNetDataModule


def get_model_data_and_callbacks(hyperparams : Union[Dict, wandb.Config]):
    
     # get image data
    image_data = get_image_data(hyperparams['dataset'])
    
    # checkpoint name
    hyperparams['checkpoint']['checkpoint_name'] = get_checkpoint_name(hyperparams)
    
    # checkpoint callbacks
    callbacks = get_callbacks(hyperparams)

    # get model pipeline 
    model = get_model_pipeline(hyperparams)
    
    return  model, image_data, callbacks

def get_model_pipeline(hyperparams: Union[Dict, wandb.Config]):

    if hyperparams.experiment.run_mode == "test":
        model = ImageClassifierPipeline.load_from_checkpoint(
            checkpoint_path=hyperparams.checkpoint.checkpoint_path + "/" + hyperparams.checkpoint.checkpoint_name + ".ckpt",
            hyperparams=hyperparams
        )
        model.freeze()
        model.eval()
    else:
        model = ImageClassifierPipeline(hyperparams)
        
    return model

def get_trainer(
    hyperparams:Union[Dict, wandb.Config],
    callbacks: list,
    wandb_logger: pl.loggers.WandbLogger
):
    if hyperparams.experiment.run_mode == "auto_tune":
        trainer = pl.Trainer(
            max_epochs=hyperparams.experiment.num_epochs, accelerator="auto", 
            auto_scale_batch_size=True, auto_lr_find=True, logger=wandb_logger, 
            callbacks=callbacks, deterministic=hyperparams.experiment.deterministic
        )
        
    elif hyperparams.experiment.run_mode == "dryrun":
        trainer = pl.Trainer(
            fast_dev_run=5, max_epochs=hyperparams.experiment.num_epochs, accelerator="auto", 
            limit_train_batches=5, limit_val_batches=5, logger=wandb_logger, 
            callbacks=callbacks, deterministic=hyperparams.experiment.deterministic
        )
    else:
        trainer = pl.Trainer(
            max_epochs=hyperparams.experiment.num_epochs, accelerator="auto", logger=wandb_logger, 
            callbacks=callbacks, deterministic=hyperparams.deterministic,
            num_nodes=hyperparams.tra, gpus=1, strategy='ddp'
        )

    return trainer
    
    
def get_callbacks(hyperparams:Union[Dict, wandb.Config]):
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=hyperparams.checkpoint.checkpoint_path,
        filename=hyperparams.checkpoint.checkpoint_name,
        monitor="val/acc",
        mode="max"
    )
    early_stop_metric_callback = EarlyStopping(monitor="val/acc", 
                    min_delta=hyperparams.experiment.min_delta, 
                    patience=hyperparams.experiment.patience, 
                    verbose=True, 
                    mode="max")
    
    return [checkpoint_callback, early_stop_metric_callback]

def get_recursive_hyperparams_identifier(hyperparams: Dict):
        # get the identifier for the canonicalization network hyperparameters
        # recursively go through the dictionary and get the values and concatenate them
        identifier = ""
        for key, value in hyperparams.items():
            if isinstance(value, dict):
                identifier += f"{key}_{get_recursive_hyperparams_identifier(value)}"
            else:
                identifier += f"{key}_{value}"
    
def get_checkpoint_name(hyperparams : Union[Dict, wandb.Config]):
    
    return f"{get_recursive_hyperparams_identifier(hyperparams['canonicalization'])}" + \
                          f"_seed_{hyperparams['experiment'].seed}" + \
                          f"_lr_{hyperparams['experiment'].lr}"
                        

def get_image_data(dataset_hyperparams:Union[Dict, wandb.Config]):
    
    dataset_classes = {
        "rotated_mnist": RotatedMNISTDataModule(dataset_hyperparams),
        "cifar10": CIFAR10DataModule(dataset_hyperparams),
        "cifar100": CIFAR100DataModule(dataset_hyperparams),
        "stl10": STL10DataModule(dataset_hyperparams),
        "celeba": CelebADataModule(dataset_hyperparams),
        "flowers102": Flowers102DataModule(dataset_hyperparams),
        "imagenet": ImageNetDataModule(dataset_hyperparams)
    }
    
    if dataset_hyperparams.dataset.dataset_name not in dataset_classes:
        raise ValueError(f"{dataset_hyperparams.dataset_name} not implemented")
    
    return dataset_classes[dataset_hyperparams.dataset_name]