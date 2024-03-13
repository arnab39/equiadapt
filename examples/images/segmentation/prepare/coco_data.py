import os

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader, Dataset

import examples.images.segmentation.prepare.vision_transforms as T


class ResizeAndPad:
    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, target):
        masks = [mask.numpy() for mask in target["masks"]]
        bboxes = target["boxes"].numpy()

        # Resize image and masks
        _, og_h, og_w = image.shape
        image = self.transform.apply_image_torch(image.unsqueeze(0)).squeeze(0)
        masks = [torch.tensor(self.transform.apply_image(mask)) for mask in masks]

        # Pad image and masks to form a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)
        masks = [transforms.Pad(padding)(mask) for mask in masks]

        # Adjust bounding boxes
        bboxes = self.transform.apply_boxes(bboxes, (og_h, og_w))
        bboxes = [
            [bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h]
            for bbox in bboxes
        ]

        target["masks"] = torch.stack(masks)
        target["boxes"] = torch.as_tensor(bboxes, dtype=torch.float32)
        return image, target


class COCODataModule(pl.LightningDataModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.hyperparams = hyperparams

    def get_transform(self, train=True):
        tr = []
        tr.append(T.PILToTensor())
        tr.append(T.ConvertImageDtype(torch.float))
        tr.append(ResizeAndPad(self.hyperparams.img_size))
        if train and self.hyperparams.augment == "flip":
            tr.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(tr)

    def collate_fn(self, batch):
        images = [x[0] for x in batch]
        targets = [x[1] for x in batch]
        return images, targets

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = COCODataset(
                root_dir=os.path.join(self.hyperparams.root_dir, "train2017"),
                annotation_file=os.path.join(
                    self.hyperparams.ann_dir, "instances_train2017.json"
                ),
                transform=self.get_transform(train=True),
            )
            self.valid_dataset = COCODataset(
                root_dir=os.path.join(self.hyperparams.root_dir, "val2017"),
                annotation_file=os.path.join(
                    self.hyperparams.ann_dir, "instances_val2017.json"
                ),
                transform=self.get_transform(train=False),
            )
        if stage == "test":
            self.test_dataset = COCODataset(
                root_dir=os.path.join(self.hyperparams.root_dir, "val2017"),
                annotation_file=os.path.join(
                    self.hyperparams.ann_dir, "instances_val2017.json"
                ),
                transform=self.get_transform(train=False),
            )
            print("Test dataset size: ", len(self.test_dataset))

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            self.hyperparams.batch_size,
            shuffle=True,
            num_workers=self.hyperparams.num_workers,
            collate_fn=self.collate_fn,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            self.hyperparams.val_batch_size,
            shuffle=False,
            num_workers=self.hyperparams.num_workers,
            collate_fn=self.collate_fn,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            self.hyperparams.val_batch_size,
            shuffle=False,
            num_workers=self.hyperparams.num_workers,
            collate_fn=self.collate_fn,
        )
        return test_loader


class COCODataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, sam_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

        # Filter out image_ids without any annotations
        self.image_ids = [
            image_id
            for image_id in self.image_ids
            if len(self.coco.getAnnIds(imgIds=image_id)) > 0
        ]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = []
        masks = []
        labels = []
        iscrowds = []
        image_ids = []
        areas = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            # there are degenerate boxes in the dataset, skip them
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            bboxes.append([x, y, x + w, y + h])  # NOTE: origin is left top corner
            labels.append(ann["category_id"])
            mask = self.coco.annToMask(ann)
            masks.append(mask)
            image_ids.append(ann["image_id"])
            areas.append(ann["area"])
            iscrowds.append(ann["iscrowd"])

        target = {}
        target["boxes"] = torch.as_tensor(bboxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["masks"] = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        target["image_id"] = torch.as_tensor(image_ids[0], dtype=torch.int64)
        target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (
            target["boxes"][:, 2] - target["boxes"][:, 0]
        )
        target["iscrowd"] = torch.as_tensor(iscrowds, dtype=torch.int64)

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target
