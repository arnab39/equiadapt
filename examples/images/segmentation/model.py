import pytorch_lightning as pl
import torch
from inference_utils import get_inference_method
from model_utils import calc_iou, get_dataset_specific_info, get_prediction_network
from omegaconf import DictConfig
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from examples.images.common.utils import get_canonicalization_network, get_canonicalizer


# define the LightningModule
class ImageSegmentationPipeline(pl.LightningModule):
    def __init__(self, hyperparams: DictConfig):
        super().__init__()

        self.loss, self.image_shape, self.num_classes = get_dataset_specific_info(
            hyperparams.dataset.dataset_name,
            hyperparams.prediction.prediction_network_architecture,
        )

        self.prediction_network = get_prediction_network(
            architecture=hyperparams.prediction.prediction_network_architecture,
            architecture_type=hyperparams.prediction.prediction_network_architecture_type,
            dataset_name=hyperparams.dataset.dataset_name,
            use_pretrained=hyperparams.prediction.use_pretrained,
            freeze_encoder=hyperparams.prediction.freeze_encoder,
            num_classes=self.num_classes,
            pretrained_ckpt_path=hyperparams.prediction.pretrained_ckpt_path,
        )

        canonicalization_network = get_canonicalization_network(
            hyperparams.canonicalization_type,
            hyperparams.canonicalization,
            self.image_shape,
        )

        self.canonicalizer = get_canonicalizer(
            hyperparams.canonicalization_type,
            canonicalization_network,
            hyperparams.canonicalization,
            self.image_shape,
        )

        self.hyperparams = hyperparams

        self.inference_method = get_inference_method(
            self.canonicalizer,
            self.prediction_network,
            hyperparams.experiment.inference,
            self.image_shape,
        )

        self.max_epochs = hyperparams.experiment.training.num_epochs

        self.save_hyperparameters()

    def apply_loss(
        self,
        loss_dict: dict,
        pred_masks: torch.Tensor,
        targets_canonicalized: dict,
        iou_predictions: torch.Tensor = None,
    ):
        assert (
            self.loss or loss_dict
        ), "Either pass a loss function or a dictionary of pre-computed losses for segmentation task loss"

        if loss_dict:
            # for maskrcnn model, the loss_dict will contain the losses
            return sum(loss_dict.values())

        num_masks = sum(len(pred_mask) for pred_mask in pred_masks)

        loss_focal = torch.tensor(0.0, device=self.hyperparams.device)
        loss_dice = torch.tensor(0.0, device=self.hyperparams.device)
        loss_iou = torch.tensor(0.0, device=self.hyperparams.device)

        for pred_mask, target, iou_prediction in zip(
            pred_masks, targets_canonicalized, iou_predictions
        ):

            # if gt_masks is larger then select the first len(pred_masks) masks
            gt_mask = target["masks"][: len(pred_mask), :, :]

            for loss_func in self.loss:
                assert hasattr(
                    loss_func, "forward"
                ), "The loss function must have a forward method"
                if loss_func.name == "focal_loss":
                    loss_focal += loss_func(pred_mask, gt_mask.float(), num_masks)
                elif loss_func.name == "dice_loss":
                    loss_dice += loss_func(pred_mask, gt_mask, num_masks)
                else:
                    raise ValueError(f"Loss function {loss_func.name} is not supported")

            if iou_predictions:
                batch_iou = calc_iou(pred_mask, gt_mask)
                loss_iou += (
                    torch.nn.functional.mse_loss(
                        iou_prediction, batch_iou, reduction="sum"
                    )
                    / num_masks
                )

        return 20.0 * loss_focal + loss_dice + loss_iou

    def training_step(self, batch: torch.Tensor):
        x, targets = batch
        x = torch.stack(x)
        batch_size, num_channels, height, width = x.shape

        # assert that the input is in the right shape
        assert (num_channels, height, width) == self.image_shape

        training_metrics = {}
        loss = 0.0

        # canonicalize the input data
        # For the vanilla model, the canonicalization is the identity transformation
        x_canonicalized, targets_canonicalized = self.canonicalizer(x, targets)

        # add group contrast loss while using optmization based canonicalization method
        if "opt" in self.hyperparams.canonicalization_type:
            group_contrast_loss = self.canonicalizer.get_optimization_specific_loss()
            loss += (
                group_contrast_loss
                * self.hyperparams.experiment.training.loss.group_contrast_weight
            )
            training_metrics.update(
                {"train/optimization_specific_loss": group_contrast_loss}
            )

        # calculate the task loss
        # if finetuning is not required, set the weight for task loss to 0
        # it will avoid unnecessary forward pass through the prediction network
        if self.hyperparams.experiment.training.loss.task_weight:

            # Forward pass through the prediction network as you'll normally do
            # Finetuning maskrcnn model will return the losses which can be used to fine tune the model
            # Meanwhile, Segment-Anything (SAM) can return boxes, ious, masks predictions
            # For uniformity, we will ensure the prediction network returns both losses and predictions irrespective of the model
            loss_dict, pred_masks, iou_predictions, _ = self.prediction_network(
                x_canonicalized, targets_canonicalized
            )

            # no requirement to invert canonicalization for the loss calculation
            # since we will compute the loss w.r.t canonicalized targets (to align with the loss computation in maskrcnn)
            task_loss = self.apply_loss(
                loss_dict, pred_masks, targets_canonicalized, iou_predictions
            )
            loss += self.hyperparams.experiment.training.loss.task_weight * task_loss

            training_metrics.update(
                {
                    "train/task_loss": task_loss,
                }
            )

        # Add prior regularization loss if the prior weight is non-zero
        if self.hyperparams.experiment.training.loss.prior_weight:
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

        # Log the training metrics
        self.log_dict(training_metrics, prog_bar=True)

        assert not torch.isnan(loss), "Loss is NaN"
        return {"loss": loss}

    def validation_step(self, batch: torch.Tensor):
        x, targets = batch
        x = torch.stack(x)
        batch_size, num_channels, height, width = x.shape

        # assert that the input is in the right shape
        assert (num_channels, height, width) == self.image_shape

        validation_metrics = {}

        # canonicalize the input data
        # For the vanilla model, the canonicalization is the identity transformation
        x_canonicalized, targets_canonicalized = self.canonicalizer(x, targets)

        # Forward pass through the prediction network as you'll normally do
        # Finetuning maskrcnn model will return the losses which can be used to fine tune the model
        # Meanwhile, Segment-Anything (SAM) can return boxes, ious, masks predictions
        # For uniformity, we will ensure the prediction network returns both losses and predictions irrespective of the model
        _, _, _, outputs = self.prediction_network(
            x_canonicalized, targets_canonicalized
        )

        _map = MeanAveragePrecision(iou_type="segm")
        targets = [
            dict(boxes=target["boxes"], labels=target["labels"], masks=target["masks"])
            for target in targets
        ]
        outputs = [
            dict(
                boxes=output["boxes"],
                labels=output["labels"],
                scores=output["scores"],
                masks=output["masks"],
            )
            for output in outputs
        ]
        _map.update(outputs, targets)
        _map_dict = _map.compute()

        validation_metrics.update(
            {
                "val/map": _map_dict["map"],
                "val/map_small": _map_dict["map_small"],
                "val/map_medium": _map_dict["map_medium"],
                "val/map_large": _map_dict["map_large"],
                "val/map_50": _map_dict["map_50"],
                "val/map_75": _map_dict["map_75"],
                "val/mar_1": _map_dict["mar_1"],
                "val/mar_10": _map_dict["mar_10"],
                "val/mar_100": _map_dict["mar_100"],
                "val/mar_small": _map_dict["mar_small"],
                "val/mar_medium": _map_dict["mar_medium"],
                "val/mar_large": _map_dict["mar_large"],
            }
        )

        # Log the identity metric if the prior weight is non-zero
        if self.hyperparams.experiment.training.loss.prior_weight:
            metric_identity = self.canonicalizer.get_identity_metric()
            validation_metrics.update({"val/identity_metric": metric_identity})

        self.log_dict(
            {key: value.to(self.device) for key, value in validation_metrics.items()},
            prog_bar=True,
            sync_dist=True,
        )

        return {"map": _map_dict["map"]}

    def test_step(self, batch: torch.Tensor):
        images, targets = batch
        images = torch.stack(images)
        batch_size, num_channels, height, width = images.shape

        # assert that the input is in the right shape
        assert (num_channels, height, width) == self.image_shape

        test_metrics = self.inference_method.get_inference_metrics(images, targets)

        # Log the test metrics
        self.log_dict(
            {key: value.to(self.device) for key, value in test_metrics.items()},
            prog_bar=True,
            sync_dist=True,
        )

        return test_metrics

    def configure_optimizers(self):
        # using SGD optimizer and MultiStepLR scheduler
        optimizer = torch.optim.SGD(
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
            momentum=0.9,
            weight_decay=5e-4,
        )

        scheduler_dict = {
            "scheduler": MultiStepLR(
                optimizer,
                milestones=self.hyperparams.experiment.training.milestones,
                gamma=0.1,
            ),
            "interval": "epoch",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
