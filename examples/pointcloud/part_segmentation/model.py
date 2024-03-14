from typing import List, Optional, Tuple, Union

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

class_choices = [
    "airplane",
    "bag",
    "cap",
    "car",
    "chair",
    "earphone",
    "guitar",
    "knife",
    "lamp",
    "laptop",
    "motorbike",
    "mug",
    "pistol",
    "rocket",
    "skateboard",
    "table",
]
seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]


class PointcloudClassificationPipeline(pl.LightningModule):
    def __init__(self, hyperparams: DictConfig) -> None:
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
        trot = None
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

    def get_label_one_hot(self, label: torch.Tensor) -> torch.Tensor:
        label_one_hot = np.zeros((label.shape[0], 16))
        for idx in range(label.shape[0]):
            label_one_hot[idx, label[idx]] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32)).to(
            label.device
        )
        return label_one_hot

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        points, targets, seg = batch
        label_one_hot = self.get_label_one_hot(targets)

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
            seg_pred = self.prediction_network(canonicalized_points, label_one_hot)
            seg_pred = seg_pred.transpose(2, 1).contiguous()

            # Loss
            task_loss = self.get_loss(
                seg_pred.view(-1, seg_pred.size(-1)), seg.view(-1)
            )
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
        self.test_pred_cls: List[np.ndarray] = []
        self.test_true_cls: List[np.ndarray] = []
        self.test_pred_seg: List[np.ndarray] = []
        self.test_true_seg: List[np.ndarray] = []
        self.test_label_seg: List[np.ndarray] = []

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        points, targets, seg = batch

        label_one_hot = self.get_label_one_hot(targets)

        points = self.maybe_transform_points(
            points, self.hyperparams.experiment.validation.rotation_type
        )

        points = points.transpose(2, 1)

        # Canonicalize the pointcloud
        canonicalized_points = self.canonicalizer(points)

        # Get the outputs from the prediction network
        seg_pred = self.prediction_network(canonicalized_points, label_one_hot)
        seg_pred = seg_pred.transpose(2, 1).contiguous()

        pred = seg_pred.max(dim=2)[1]

        seg_np = seg.cpu().numpy()
        pred_np = pred.detach().cpu().numpy()

        self.test_true_cls.append(seg_np.reshape(-1))
        self.test_pred_cls.append(pred_np.reshape(-1))
        self.test_true_seg.append(seg_np)
        self.test_pred_seg.append(pred_np)
        self.test_label_seg.append(targets.detach().cpu().numpy().reshape(-1))

        return pred

    def on_validation_epoch_end(self) -> dict:
        test_true = np.concatenate(self.test_true_cls)
        test_pred = np.concatenate(self.test_pred_cls)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        test_true_seg = np.concatenate(self.test_true_seg, axis=0)
        test_pred_seg = np.concatenate(self.test_pred_seg, axis=0)
        test_label_seg = np.concatenate(self.test_label_seg)
        test_ious = calculate_shape_IoU(
            test_pred_seg, test_true_seg, test_label_seg, class_choice=None
        )
        validation_metrics = {
            "val/acc": test_acc,
            "val/avg_per_class_acc": avg_per_class_acc,
            "val/iou": np.mean(test_ious),
        }
        self.log_dict(validation_metrics, on_epoch=True, prog_bar=True, sync_dist=True)

        return validation_metrics

    def on_test_epoch_start(self) -> None:
        self.test_pred_cls = []
        self.test_true_cls = []
        self.test_pred_seg = []
        self.test_true_seg = []
        self.test_label_seg = []

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        points, targets, seg = batch

        label_one_hot = self.get_label_one_hot(targets)

        points = self.maybe_transform_points(
            points, self.hyperparams.experiment.validation.rotation_type
        )

        points = points.transpose(2, 1)

        # Canonicalize the pointcloud
        canonicalized_points = self.canonicalizer(points)

        # Get the outputs from the prediction network
        seg_pred = self.prediction_network(canonicalized_points, label_one_hot)
        seg_pred = seg_pred.transpose(2, 1).contiguous()

        pred = seg_pred.max(dim=2)[1]

        seg_np = seg.cpu().numpy()
        pred_np = pred.detach().cpu().numpy()

        self.test_true_cls.append(seg_np.reshape(-1))
        self.test_pred_cls.append(pred_np.reshape(-1))
        self.test_true_seg.append(seg_np)
        self.test_pred_seg.append(pred_np)
        self.test_label_seg.append(targets.detach().cpu().numpy().reshape(-1))

        return pred

    def on_test_epoch_end(self) -> dict:
        test_true = np.concatenate(self.test_true_cls)
        test_pred = np.concatenate(self.test_pred_cls)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        test_true_seg = np.concatenate(self.test_true_seg, axis=0)
        test_pred_seg = np.concatenate(self.test_pred_seg, axis=0)
        test_label_seg = np.concatenate(self.test_label_seg)
        test_ious = calculate_shape_IoU(
            test_pred_seg, test_true_seg, test_label_seg, class_choice=None
        )
        test_metrics = {
            "test/acc": test_acc,
            "test/avg_per_class_acc": avg_per_class_acc,
            "test/iou": np.mean(test_ious),
        }
        self.log_dict(test_metrics, on_epoch=True, prog_bar=True, sync_dist=True)

        return test_metrics

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


def calculate_shape_IoU(
    pred_np: np.ndarray,
    seg_np: np.ndarray,
    label: np.ndarray,
    class_choice: Optional[str] = None,
    visual: bool = False,
) -> List[float]:
    if not visual:
        label = label.squeeze()
    shape_ious = []
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            start_index = index_start[label[shape_idx]]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
        part_ious = []
        for part in parts:
            I = np.sum(
                np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part)
            )
            U = np.sum(
                np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part)
            )
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious
