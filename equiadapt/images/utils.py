from typing import List, Tuple

import kornia as K
import torch
from torchvision import transforms


def roll_by_gather(feature_map: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """
    Shifts the feature map along the group dimension by the specified shifts.

    Args:
        feature_map (torch.Tensor): The input feature map. It should have the shape (batch, channel, group, x_dim, y_dim).
        shifts (torch.Tensor): The shifts for each feature map in the batch.

    Returns:
        torch.Tensor: The shifted feature map.
    """
    device = shifts.device
    # assumes 2D array
    batch, channel, group, x_dim, y_dim = feature_map.shape
    arange1 = (
        torch.arange(group)
        .view((1, 1, group, 1, 1))
        .repeat((batch, channel, 1, x_dim, y_dim))
        .to(device)
    )
    arange2 = (arange1 - shifts[:, None, None, None, None].long()) % group
    return torch.gather(feature_map, 2, arange2)


def get_action_on_image_features(
    feature_map: torch.Tensor,
    group_info_dict: dict,
    group_element_dict: dict,
    induced_rep_type: str = "regular",
) -> torch.Tensor:
    """
    Applies a group action to the feature map.

    Args:
        feature_map (torch.Tensor): The input feature map.
        group_info_dict (dict): A dictionary containing information about the group.
        group_element_dict (dict): A dictionary containing the group elements.
        induced_rep_type (str, optional): The type of induced representation. Defaults to "regular".

    Returns:
        torch.Tensor: The feature map after the group action has been applied.
    """
    num_rotations = group_info_dict["num_rotations"]
    num_group = group_info_dict["num_group"]
    assert len(feature_map.shape) == 4
    batch_size, C, H, W = feature_map.shape
    if induced_rep_type == "regular":
        assert feature_map.shape[1] % num_group == 0
        angles = group_element_dict["group"]["rotation"]
        x_out = K.geometry.rotate(feature_map, angles)

        if "reflection" in group_element_dict["group"]:
            reflect_indicator = group_element_dict["group"]["reflection"]
            x_out_reflected = K.geometry.hflip(x_out)
            x_out = x_out * reflect_indicator[:, None, None, None] + x_out_reflected * (
                1 - reflect_indicator[:, None, None, None]
            )

        x_out = x_out.reshape(batch_size, C // num_group, num_group, H, W)
        shift = angles / 360.0 * num_rotations
        if "reflection" in group_element_dict["group"]:
            x_out = torch.cat(
                [
                    roll_by_gather(x_out[:, :, :num_rotations], shift),
                    roll_by_gather(x_out[:, :, num_rotations:], -shift),
                ],
                dim=2,
            )
        else:
            x_out = roll_by_gather(x_out, shift)
        x_out = x_out.reshape(batch_size, -1, H, W)
        return x_out
    elif induced_rep_type == "scalar":
        angles = group_element_dict["group"][0]
        x_out = K.geometry.rotate(feature_map, angles)
        if "reflection" in group_element_dict["group"]:
            reflect_indicator = group_element_dict["group"]["reflection"]
            x_out_reflected = K.geometry.hflip(x_out)
            x_out = x_out * reflect_indicator[:, None, None, None] + x_out_reflected * (
                1 - reflect_indicator[:, None, None, None]
            )
        return x_out
    elif induced_rep_type == "vector":
        # TODO: Implement the action for vector representation
        raise NotImplementedError("Action for vector representation is not implemented")
    else:
        raise ValueError("induced_rep_type must be regular, scalar or vector")


def flip_boxes(boxes: torch.Tensor, width: int) -> torch.Tensor:
    """
    Flips bounding boxes horizontally.

    Args:
        boxes (torch.Tensor): The bounding boxes to flip.
        width (int): The width of the image.

    Returns:
        torch.Tensor: The flipped bounding boxes.
    """
    boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
    return boxes


def flip_masks(masks: torch.Tensor) -> torch.Tensor:
    """
    Flips masks horizontally.

    Args:
        masks (torch.Tensor): The masks to flip.

    Returns:
        torch.Tensor: The flipped masks.
    """
    return masks.flip(-1)


def rotate_masks(masks: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Rotates masks by a specified angle.

    Args:
        masks (torch.Tensor): The masks to rotate.
        angle (torch.Tensor): The angle to rotate the masks by.

    Returns:
        torch.Tensor: The rotated masks.
    """
    return transforms.functional.rotate(masks, angle)


def rotate_points(
    origin: List[float], point: torch.Tensor, angle: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rotates a point around an origin by a specified angle.

    Args:
        origin (List[float]): The origin to rotate the point around.
        point (torch.Tensor): The point to rotate.
        angle (torch.Tensor): The angle to rotate the point by.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The rotated point.
    """
    ox, oy = origin
    px, py = point

    qx = ox + torch.cos(angle) * (px - ox) - torch.sin(angle) * (py - oy)
    qy = oy + torch.sin(angle) * (px - ox) + torch.cos(angle) * (py - oy)
    return qx, qy


def rotate_boxes(boxes: torch.Tensor, angle: torch.Tensor, width: int) -> torch.Tensor:
    """
    Rotates bounding boxes by a specified angle.

    Args:
        boxes (torch.Tensor): The bounding boxes to rotate.
        angle (torch.Tensor): The angle to rotate the bounding boxes by.
        width (int): The width of the image.

    Returns:
        torch.Tensor: The rotated bounding boxes.
    """
    # rotate points
    origin: List[float] = [width / 2, width / 2]
    x_min_rot, y_min_rot = rotate_points(origin, boxes[:, :2].T, torch.deg2rad(angle))
    x_max_rot, y_max_rot = rotate_points(origin, boxes[:, 2:].T, torch.deg2rad(angle))

    # rearrange the max and mins to get rotated boxes
    x_min_rot, x_max_rot = torch.min(x_min_rot, x_max_rot), torch.max(
        x_min_rot, x_max_rot
    )
    y_min_rot, y_max_rot = torch.min(y_min_rot, y_max_rot), torch.max(
        y_min_rot, y_max_rot
    )
    rotated_boxes = torch.stack([x_min_rot, y_min_rot, x_max_rot, y_max_rot], dim=-1)

    return rotated_boxes
