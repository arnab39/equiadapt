import glob
import os
import warnings

import h5py
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


def download_shapenetpart(root_dir):
    DATA_DIR = root_dir
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, "shapenet_part_seg_hdf5_data")):
        www = "https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip"
        zipfile = os.path.basename(www)
        os.system("wget --no-check-certificate %s; unzip %s" % (www, zipfile))
        os.system(
            "mv %s %s"
            % ("hdf5_data", os.path.join(DATA_DIR, "shapenet_part_seg_hdf5_data"))
        )
        os.system("rm %s" % (zipfile))


def load_data_partseg(root_dir, partition):
    download_shapenetpart(root_dir)
    DATA_DIR = root_dir
    all_data = []
    all_label = []
    all_seg = []
    if partition == "trainval":
        file = glob.glob(
            os.path.join(DATA_DIR, "shapenet_part_seg_hdf5_data", "*train*.h5")
        ) + glob.glob(os.path.join(DATA_DIR, "shapenet_part_seg_hdf5_data", "*val*.h5"))
    else:
        file = glob.glob(
            os.path.join(DATA_DIR, "shapenet_part_seg_hdf5_data", "*%s*.h5" % partition)
        )
    for h5_name in file:
        f = h5py.File(h5_name, "r+")
        data = f["data"][:].astype("float32")
        label = f["label"][:].astype("int64")
        seg = f["pid"][:].astype("int64")
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class ShapeNetPartDataset(Dataset):
    def __init__(self, root_dir, num_points, partition="train", normalize=False):
        self.data, self.label, self.seg = load_data_partseg(root_dir, partition)
        self.cat2id = {
            "airplane": 0,
            "bag": 1,
            "cap": 2,
            "car": 3,
            "chair": 4,
            "earphone": 5,
            "guitar": 6,
            "knife": 7,
            "lamp": 8,
            "laptop": 9,
            "motor": 10,
            "mug": 11,
            "pistol": 12,
            "rocket": 13,
            "skateboard": 14,
            "table": 15,
        }
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition
        self.normalize = normalize
        self.seg_num_all = 50
        self.seg_start_index = 0

    def __getitem__(self, item):
        pointcloud = self.data[item][: self.num_points]
        label = self.label[item]
        seg = self.seg[item][: self.num_points]
        if self.partition == "trainval":
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        if self.normalize:
            pointcloud = pc_normalize(pointcloud)
        return pointcloud, label, seg

    def __len__(self):
        return self.data.shape[0]


class ShapeNetDataModule(pl.LightningDataModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.data_path = hyperparams.data_path
        self.hyperparams = hyperparams

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = ShapeNetPartDataset(
                root_dir=self.data_path,
                num_points=self.hyperparams.num_points,
                partition="trainval",
                normalize=self.hyperparams.normalize,
            )
            self.valid_dataset = ShapeNetPartDataset(
                root_dir=self.data_path,
                num_points=self.hyperparams.num_points,
                partition="test",
                normalize=self.hyperparams.normalize,
            )
        if stage == "test":
            self.test_dataset = ShapeNetPartDataset(
                root_dir=self.data_path,
                num_points=self.hyperparams.num_points,
                partition="test",
                normalize=self.hyperparams.normalize,
            )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.hyperparams.batch_size,
            shuffle=True,
            num_workers=self.hyperparams.num_workers,
            drop_last=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.hyperparams.batch_size,
            shuffle=True,
            num_workers=self.hyperparams.num_workers,
            drop_last=False,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.hyperparams.batch_size,
            shuffle=False,
            num_workers=self.hyperparams.num_workers,
            drop_last=False,
        )
        return test_loader
