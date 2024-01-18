import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
from argparse import ArgumentParser
import os
import torch
import time
from tqdm import tqdm

from examples.images.prepare import RotatedMNISTDataModule, CIFAR10DataModule, CIFAR100DataModule, STL10DataModule, Flowers102DataModule, CelebADataModule, ImageNetDataModule
from canonical_network.models.classification_model import LitClassifier

def get_hyperparams():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="vanilla", choices=["equivariant", "vanilla", "steerable", "opt_equivariant"], help="model to train")
    parser.add_argument("--opt_type", type=str, default="cosine", choices=["cosine", "hinge"], help="optimization loss to avoid collapse")
    parser.add_argument("--base_encoder", type=str, default="cnn", help="base encoder to use for the model")
    
    parser.add_argument("--use_wandb", type=int, default=1, help="use wandb")
    parser.add_argument("--run_mode", type=str, default='train', help="different run modes 1)dryrun 2)train 3)test 4)auto_tune")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    parser.add_argument("--wandb_project", type=str, default="canonical_network", help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="symmetry_group", help="wandb entity name")
    parser.add_argument("--wandb_cache_dir", type=str, default="/home/mila/s/siba-smarak.panigrahi/scratch/wandb_artifacts")
    parser.add_argument("--wandb_dir", type=str, default="/home/mila/s/siba-smarak.panigrahi/scratch/wandb")

    parser.add_argument("--dataset", type=str, default="rotated_mnist", help="dataset to train on")
    parser.add_argument("--data_path", type=str, default="/home/mila/s/siba-smarak.panigrahi/scratch/data", help="path to data")

    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="patience for early stopping")
    parser.add_argument("--min_delta", type=float, default=0.0, help="min_delta for early stopping")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")

    parser.add_argument("--canonization_kernel_size", type=int, default=3, help="kernel size for canonization layer")
    parser.add_argument("--canonization_out_channels", type=int, default=16, help="number of equivariant output channels for the canonization network")
    parser.add_argument("--canonization_num_layers", type=int, default=3, help="number of equivariant output channels for the canonization network")
    parser.add_argument("--canonization_beta", type=float, default=1.0, help="sharpness of the canonization network output")
    parser.add_argument("--canonization_name", type=str, default="ours", choices=["ours","escnn","steerable","cnn"], help="name of the canonization network")
    parser.add_argument("--num_freq", type=int, default=1, help="number of frequencies for the steerable canonization network")
    parser.add_argument("--group_type", type=str, default="rotation", help="group type for equivariance 1) rotation 2) roto-reflection")
    parser.add_argument("--num_rotations", type=int, default=4, help="order of the group")
    parser.add_argument("--num_channels", type=int, default=20, help="num_channels for equivariant cnn base encoder")
    
    parser.add_argument("--checkpoint_path", type=str, default="/home/mila/s/siba-smarak.panigrahi/scratch/non_equivariance", help="path to checkpoint")
    parser.add_argument("--deterministic", type=bool, default=False, help="deterministic training")
    parser.add_argument("--save_canonized_images", type=int, default=0, help="save canonized images")
    parser.add_argument("--check_invariance", type=int, default=0, help="check if the network is invariant")
    parser.add_argument("--freeze_steerable_regression_model", type=int, default=0)

    parser.add_argument("--hinge_threshold", type=float, default=10.0, help="threshold for the hinge loss in opt_equivariant model")
    parser.add_argument("--prior_weight", type=float, default=100.0, help="weight for the prior loss for combined training")
    parser.add_argument("--prior_train_epochs", type=int, default=20, help="number of epochs to train the prior fitting objective")
    parser.add_argument("--version", type=str, default='v0', help="version of the training")

    parser.add_argument("--eval_mode", type=str, default="o2_classification", help="eval mode 1) default 2) so2_classification 3) o2_classification")
    parser.add_argument("--eval_num_rotations", type=int, default=4, help="order of group to evaluate")
    parser.add_argument("--use_pretrained", type=int, default=0, help="use pretrained model")

    parser.add_argument("--augment", type=int, default=1, help="augment data")
    parser.add_argument("--freeze_pretrained_encoder", type=int, default=0, help="freeze pretrained encoder")
    args = parser.parse_args()
    return args


def train_images():
    hyperparams = get_hyperparams()
    hyperparams.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hyperparams.data_path = hyperparams.data_path + "/" + hyperparams.dataset
    hyperparams.checkpoint_path = hyperparams.checkpoint_path + "/" + hyperparams.dataset + "/" + hyperparams.model \
                                  + "/" + hyperparams.base_encoder
    hyperparams.wandb_project = hyperparams.wandb_project

    if not hyperparams.use_wandb:
        print('Wandb disable for logging.')
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_CACHE_DIR"] = hyperparams.wandb_cache_dir
        os.environ["WANDB_DIR"] = hyperparams.wandb_dir
    else:
        print('Using wandb for logging.')
        os.environ["WANDB_MODE"] = "online"
        os.environ["WANDB_CACHE_DIR"] = hyperparams.wandb_cache_dir

    wandb.init(config=hyperparams, entity=hyperparams.wandb_entity, project=hyperparams.wandb_project, dir=hyperparams.wandb_dir)
    wandb_logger = WandbLogger(project=hyperparams.wandb_project, log_model="all")
    hyperparams = wandb.config

    pl.seed_everything(hyperparams.seed)

    if hyperparams.dataset == "rotated_mnist":
        image_data = RotatedMNISTDataModule(hyperparams)
    elif hyperparams.dataset == "cifar10":
        image_data = CIFAR10DataModule(hyperparams)
    elif hyperparams.dataset == "cifar100":
        image_data = CIFAR100DataModule(hyperparams)
    elif hyperparams.dataset == "stl10":
        image_data = STL10DataModule(hyperparams)
    elif hyperparams.dataset == "celeba":
        image_data = CelebADataModule(hyperparams)
    elif hyperparams.dataset == "flowers102":
        image_data = Flowers102DataModule(hyperparams)
    elif hyperparams.dataset == "ImageNet":
        image_data = ImageNetDataModule(hyperparams)
    else:
        raise NotImplementedError("Dataset not implemented")

    if hyperparams.model == "vanilla":
        checkpoint_name = f"{hyperparams.model}_seed_{hyperparams.seed}"
    else:
        checkpoint_name = f"{hyperparams.model}_kernel_{hyperparams.canonization_kernel_size}_" \
                          f"num-layer_{hyperparams.canonization_num_layers}_{hyperparams.group_type}_" \
                          f"{hyperparams.num_rotations}_seed_{hyperparams.seed}_version_{hyperparams.version}_"\
                          f"dataset_{hyperparams.dataset}_lr_{hyperparams.lr}_opt_{hyperparams.opt_type}"

    hyperparams.checkpoint_name = checkpoint_name
    checkpoint_callback = ModelCheckpoint(
        dirpath=hyperparams.checkpoint_path,
        filename= checkpoint_name,
        monitor="val/acc",
        mode="max"
    )
    early_stop_metric_callback = EarlyStopping(monitor="val/acc", 
                    min_delta=hyperparams.min_delta, 
                    patience=hyperparams.patience, 
                    verbose=True, 
                    mode="max")

    callbacks = [checkpoint_callback, early_stop_metric_callback]

    if hyperparams.run_mode == "test":
        model = LitClassifier.load_from_checkpoint(
            checkpoint_path=hyperparams.checkpoint_path + "/" + checkpoint_name + ".ckpt",
            hyperparams=hyperparams
        )
        model.freeze()
        model.eval()
    else:
        model = LitClassifier(hyperparams)

    if hyperparams.model == "equivariant":
        wandb.watch(model.network.canonization_network, log='all')

    if hyperparams.run_mode == "auto_tune":
        trainer = pl.Trainer(
            max_epochs=hyperparams.num_epochs, accelerator="auto", 
            auto_scale_batch_size=True, auto_lr_find=True, logger=wandb_logger, 
            callbacks=callbacks, deterministic=hyperparams.deterministic
        )
        trainer.tune(model, datamodule=image_data)
    elif hyperparams.run_mode == "dryrun":
        trainer = pl.Trainer(
            fast_dev_run=2000, max_epochs=hyperparams.num_epochs, accelerator="auto", 
            limit_train_batches=5, limit_val_batches=5, logger=wandb_logger, 
            callbacks=callbacks, deterministic=hyperparams.deterministic
        )
    else:
        trainer = pl.Trainer(
            max_epochs=hyperparams.num_epochs, accelerator="auto", logger=wandb_logger, 
            callbacks=callbacks, deterministic=hyperparams.deterministic,
            num_nodes=1, gpus=1, strategy='ddp'
        )

    if hyperparams.run_mode == "train":
        trainer.fit(model, datamodule=image_data)
    


    trainer.test(model, datamodule=image_data)



def main():
    train_images()


if __name__ == "__main__":
    main()