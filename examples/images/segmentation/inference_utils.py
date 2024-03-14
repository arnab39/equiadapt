import copy
import math
from typing import Tuple, Union

import torch
from omegaconf import DictConfig
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import transforms

from equiadapt.images.utils import flip_boxes, flip_masks, rotate_boxes, rotate_masks


class VanillaInference:
    def __init__(
        self, canonicalizer: torch.nn.Module, prediction_network: torch.nn.Module
    ) -> None:
        self.canonicalizer = canonicalizer
        self.prediction_network = prediction_network

    def forward(
        self, x: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # canonicalize the input data
        # For the vanilla model, the canonicalization is the identity transformation
        x_canonicalized, targets_canonicalized = self.canonicalizer(x, targets)

        # Forward pass through the prediction network as you'll normally do
        # Finetuning maskrcnn model will return the losses which can be used to fine tune the model
        # Meanwhile, Segment-Anything (SAM) can return boxes, ious, masks predictions
        # For uniformity, we will ensure the prediction network returns both losses and predictions irrespective of the model
        return self.prediction_network(x_canonicalized, targets_canonicalized)

    def get_inference_metrics(self, x: torch.Tensor, targets: torch.Tensor) -> dict:
        # Forward pass through the prediction network
        _, _, _, outputs = self.forward(x, targets)

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

        metrics = {"test/map": _map_dict["map"]}

        return metrics


class GroupInference(VanillaInference):
    def __init__(
        self,
        canonicalizer: torch.nn.Module,
        prediction_network: torch.nn.Module,
        inference_hyperparams: DictConfig,
        in_shape: tuple = (3, 32, 32),
    ):

        super().__init__(canonicalizer, prediction_network)
        self.group_type = inference_hyperparams.group_type
        self.num_rotations = inference_hyperparams.num_rotations
        self.num_group_elements = (
            self.num_rotations
            if self.group_type == "rotation"
            else 2 * self.num_rotations
        )
        self.pad = transforms.Pad(math.ceil(in_shape[-2] * 0.4), padding_mode="edge")
        self.crop = transforms.CenterCrop((in_shape[-2], in_shape[-1]))

    def get_group_element_wise_maps(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> dict:
        map_dict = dict()
        image_width = images[0].shape[1]

        degrees = torch.linspace(0, 360, self.num_rotations + 1)[:-1]
        for rot, degree in enumerate(degrees):

            targets_transformed = copy.deepcopy(targets)

            # apply group element on images
            images_pad = self.pad(images)
            images_rot = transforms.functional.rotate(images_pad, degree.item())
            images_rot = self.crop(images_rot)

            # apply group element on bounding boxes and masks
            for t in range(len(targets_transformed)):
                targets_transformed[t]["boxes"] = rotate_boxes(
                    targets_transformed[t]["boxes"], -degree, image_width
                )
                targets_transformed[t]["masks"] = rotate_masks(
                    targets_transformed[t]["masks"], degree.item()
                )

            # get predictions for the transformed images
            _, _, _, outputs = self.forward(images_rot, targets_transformed)

            Map = MeanAveragePrecision(iou_type="segm")
            targets = [
                dict(
                    boxes=target["boxes"],
                    labels=target["labels"],
                    masks=target["masks"],
                )
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
            Map.update(outputs, targets)

            map_dict[rot] = Map.compute()

        if self.group_type == "roto-reflection":
            # Rotate the reflected images and get the logits
            for rot, degree in enumerate(degrees):

                images_pad = self.pad(images)
                images_reflect = transforms.functional.hflip(images_pad)
                images_rotoreflect = transforms.functional.rotate(
                    images_reflect, degree.item()
                )
                images_rotoreflect = self.crop(images_rotoreflect)

                # apply group element on bounding boxes and masks
                for t in range(len(targets_transformed)):
                    targets_transformed[t]["boxes"] = rotate_boxes(
                        targets_transformed[t]["boxes"], -degree, image_width
                    )
                    targets_transformed[t]["boxes"] = flip_boxes(
                        targets_transformed[t]["boxes"], image_width
                    )

                    targets_transformed[t]["masks"] = rotate_masks(
                        targets_transformed[t]["masks"], degree
                    )
                    targets_transformed[t]["masks"] = flip_masks(
                        targets_transformed[t]["masks"]
                    )

                # get predictions for the transformed images
                _, _, _, outputs = self.forward(images_rotoreflect, targets_transformed)

                Map = MeanAveragePrecision(iou_type="segm")
                targets = [
                    dict(
                        boxes=target["boxes"],
                        labels=target["labels"],
                        masks=target["masks"],
                    )
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
                Map.update(outputs, targets)

                map_dict[rot + len(degrees)] = Map.compute()

        return map_dict

    def get_inference_metrics(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> dict:
        metrics = {}

        map_dict = self.get_group_element_wise_maps(images, targets)

        # Use list comprehension to calculate accuracy for each group element
        for i in range(self.num_group_elements):
            metrics.update(
                {
                    f"test/map_group_element_{i}": max(map_dict[i]["map"], 0.0),
                    f"test/map_small_group_element_{i}": max(
                        map_dict[i]["map_small"], 0.0
                    ),
                    f"test/map_medium_group_element_{i}": max(
                        map_dict[i]["map_medium"], 0.0
                    ),
                    f"test/map_large_group_element_{i}": max(
                        map_dict[i]["map_large"], 0.0
                    ),
                    f"test/map_50_group_element_{i}": max(map_dict[i]["map_50"], 0.0),
                    f"test/map_75_group_element_{i}": max(map_dict[i]["map_75"], 0.0),
                    f"test/mar_1_group_element_{i}": max(map_dict[i]["mar_1"], 0.0),
                    f"test/mar_10_group_element_{i}": max(map_dict[i]["mar_10"], 0.0),
                    f"test/mar_100_group_element_{i}": max(map_dict[i]["mar_100"], 0.0),
                    f"test/mar_small_group_element_{i}": max(
                        map_dict[i]["mar_small"], 0.0
                    ),
                    f"test/mar_medium_group_element_{i}": max(
                        map_dict[i]["mar_medium"], 0.0
                    ),
                    f"test/mar_large_group_element_{i}": max(
                        map_dict[i]["mar_large"], 0.0
                    ),
                }
            )

        map_per_group_element = torch.tensor(
            [map_dict[i]["map"] for i in range(self.num_group_elements)]
        )

        metrics.update({"test/group_map": torch.mean(map_per_group_element)})
        metrics.update(
            {
                f"test/map_group_element_{i}": max(map_per_group_element[i], 0.0)
                for i in range(self.num_group_elements)
            }
        )

        # Calculate the overall map
        metrics.update({"test/map": max(map_dict[0]["map"], 0.0)})

        return metrics


def get_inference_method(
    canonicalizer: torch.nn.Module,
    prediction_network: torch.nn.Module,
    inference_hyperparams: DictConfig,
    in_shape: tuple = (3, 1024, 1024),
) -> Union[VanillaInference, GroupInference]:
    if inference_hyperparams.method == "vanilla":
        return VanillaInference(canonicalizer, prediction_network)
    elif inference_hyperparams.method == "group":
        return GroupInference(
            canonicalizer, prediction_network, inference_hyperparams, in_shape
        )
    else:
        raise ValueError(f"{inference_hyperparams.method} is not implemented for now.")
