from typing import List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
from model_utils import get_prediction_network
from omegaconf import DictConfig
from pytorch3d.transforms import Rotate, RotateAxisAngle, random_rotations
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from examples.pointcloud.common.utils import (
    get_canonicalization_network,
    get_canonicalizer,
    random_point_dropout,
    random_scale_point_cloud,
    random_shift_point_cloud,
)


class PointcloudClassificationPipeline(pl.LightningModule):
    def __init__(self, hyperparams: DictConfig):
        super().__init__()
        self.hyperparams = hyperparams

        canonicalization_network = get_canonicalization_network(
            hyperparams.canonicalization_type, hyperparams.canonicalization
        )

        self.canonicalizer = get_canonicalizer(
            hyperparams.canonicalization_type,
            canonicalization_network,
            hyperparams.canonicalization,
        )

        self.prediction_network = get_prediction_network(
            hyperparams.prediction.prediction_network_architecture,
            hyperparams.prediction,
        )

        self.save_hyperparameters()

    def maybe_transform_points(
        self, points: torch.Tensor, rotation_type: str
    ) -> torch.Tensor:
        """
        Apply random rotation to the pointcloud

        Args:
            points (torch.Tensor): pointcloud of shape (B, 3, N)
            rotation_type (str): type of rotation to apply. Options are 1) z 2) so3
        """
        if rotation_type == "z":
            trot = RotateAxisAngle(
                angle=torch.rand(points.shape[0]) * 360,
                axis="Z",
                degrees=True,
                device=self.device,
            )
        elif rotation_type == "so3":
            trot = Rotate(R=random_rotations(points.shape[0]), device=self.device)
        elif rotation_type == "none":
            trot = None
        else:
            raise NotImplementedError(f"Unknown rotation type {rotation_type}")
        if trot is not None:
            points = trot.transform_points(points)
        return points

    def augment_points(self, points: torch.Tensor) -> torch.Tensor:
        points = random_point_dropout(points)
        points = random_scale_point_cloud(points)
        points = random_shift_point_cloud(points)
        return points

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        points, targets = batch
        targets = targets.squeeze()

        training_metrics = {}
        loss = 0.0

        points = self.maybe_transform_points(
            points, self.hyperparams.experiment.training.rotation_type
        )

        if self.hyperparams.experiment.training.augment:
            points = self.augment_points(points)

        points = points.transpose(2, 1)

        # Canonicalize the pointcloud
        canonicalized_points = self.canonicalizer(points)

        # calculate the task loss which is the cross-entropy loss for classification
        if self.hyperparams.experiment.training.loss.task_weight:
            # Get the outputs from the prediction network
            logits = self.prediction_network(canonicalized_points)

            # Get the task loss
            task_loss = self.get_loss(logits, targets)

            loss += task_loss * self.hyperparams.experiment.training.loss.task_weight

            training_metrics.update(
                {
                    "train/task_loss": task_loss,
                }
            )

        if (
            self.hyperparams.experiment.training.loss.prior_weight
            and self.hyperparams.canonicalization_type != "identity"
        ):
            prior_loss = self.canonicalizer.get_prior_regularization_loss()
            loss += prior_loss * self.hyperparams.experiment.training.loss.prior_weight
            metric_identity = self.canonicalizer.get_identity_metric()
            training_metrics.update(
                {
                    "train/prior_loss": prior_loss,
                    "train/identity_metric": metric_identity,
                }
            )

        training_metrics.update(
            {
                "train/loss": loss,
            }
        )

        self.log_dict(training_metrics, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def on_validation_epoch_start(self) -> None:
        self.test_pred: List[np.ndarray] = []
        self.test_true: List[np.ndarray] = []

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        points, targets = batch
        targets = targets.squeeze()

        points = self.maybe_transform_points(
            points, self.hyperparams.experiment.validation.rotation_type
        )

        points = points.transpose(2, 1)

        # Canonicalize the pointcloud
        canonicalized_points = self.canonicalizer(points)

        # Get the outputs from the prediction network
        logits = self.prediction_network(canonicalized_points)

        preds = logits.max(dim=1)[1]

        self.test_true.append(targets.cpu().numpy())
        self.test_pred.append(preds.detach().cpu().numpy())

        return preds

    def on_validation_epoch_end(self) -> dict:
        test_true = np.concatenate(self.test_true)
        test_pred = np.concatenate(self.test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        self.log_dict(
            {
                "val/acc": test_acc,
                "val/avg_per_class_acc": avg_per_class_acc,
            },  # type: ignore
            prog_bar=True,
            sync_dist=True,
        )

        return {"val/acc": test_acc, "val/avg_per_class_acc": avg_per_class_acc}

    def on_test_epoch_start(self) -> None:
        self.test_pred = []
        self.test_true = []

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        points, targets = batch
        targets = targets.squeeze()

        points = self.maybe_transform_points(
            points, self.hyperparams.experiment.test.rotation_type
        )

        points = points.transpose(2, 1)

        # Canonicalize the pointcloud
        canonicalized_points = self.canonicalizer(points)

        # Get the outputs from the prediction network
        logits = self.prediction_network(canonicalized_points)

        preds = logits.max(dim=1)[1]

        self.test_true.append(targets.cpu().numpy())
        self.test_pred.append(preds.detach().cpu().numpy())

        return preds

    def on_test_epoch_end(self) -> dict:
        test_true = np.concatenate(self.test_true)
        test_pred = np.concatenate(self.test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        self.log_dict(
            {
                "test/acc": test_acc,
                "test/avg_per_class_acc": avg_per_class_acc,
            },  # type: ignore
            prog_bar=True,
            sync_dist=True,
        )

        return {"test/acc": test_acc, "test/avg_per_class_acc": avg_per_class_acc}

    def get_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        smoothing: bool = False,
        ignore_index: int = 255,
    ) -> torch.Tensor:
        targets = targets.contiguous().view(-1)
        if smoothing:
            eps = 0.2
            num_classes = predictions.size(1)
            one_hot = torch.zeros_like(predictions).scatter(1, targets.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (num_classes - 1)
            log_prb = F.log_softmax(predictions, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(
                predictions, targets, reduction="mean", ignore_index=ignore_index
            )

        return loss

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, dict]:
        # torch.autograd.set_detect_anomaly(True)
        if self.hyperparams.experiment.training.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": self.prediction_network.parameters(),
                        "lr": self.hyperparams.experiment.training.prediction_lr,
                    },
                    {
                        "params": self.canonicalizer.parameters(),
                        "lr": self.hyperparams.experiment.training.canonicalization_lr,
                    },
                ],
                weight_decay=1e-4,
            )
            print("Using Adam optimizer")
            return optimizer
        elif self.hyperparams.experiment.training.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                [
                    {
                        "params": self.prediction_network.parameters(),
                        "lr": self.hyperparams.experiment.training.prediction_lr * 100,
                    },
                    {
                        "params": self.canonicalizer.parameters(),
                        "lr": self.hyperparams.experiment.training.canonicalization_lr
                        * 100,
                    },
                ],
                momentum=0.9,
                weight_decay=1e-4,
            )

            if self.hyperparams.experiment.training.lr_scheduler == "cosine":
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=self.hyperparams.experiment.training.num_epochs,
                    eta_min=1e-3,
                )
            elif self.hyperparams.experiment.training.lr_scheduler == "step":
                scheduler = StepLR(optimizer, step_size=20, gamma=0.7)
            else:
                raise NotImplementedError(
                    f"Unknown learning rate decay schedule {self.hyperparams.experiment.training.lr_scheduler}"
                )

            scheduler_dict = {
                "scheduler": scheduler,
                "interval": "epoch",
            }
            print("Using SGD optimizer with learning rate scheduler")
            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        else:
            raise NotImplementedError
