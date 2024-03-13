import pathlib
from typing import Any, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

SRC_PATH = pathlib.Path(__file__).parent.parent
DATA_PATH = SRC_PATH / "data"


class NBodyDataset:
    """
    Dataset class for N-Body simulation data.
    """

    def __init__(
        self,
        partition: str = "train",
        max_samples: int = 3000,
        dataset_name: str = "nbody_small",
    ):
        """
        Initialize the NBodyDataset.

        Args:
            partition (str): Partition of the dataset ("train", "val", or "test").
            max_samples (int): Maximum number of samples to load.
            dataset_name (str): Name of the dataset.
        """
        self.partition = partition
        if self.partition == "val":
            self.suffix = "valid"
        else:
            self.suffix = self.partition
        self.dataset_name = dataset_name
        if dataset_name == "nbody":
            self.suffix += "_charged5_initvel1"
        elif dataset_name == "nbody_small" or dataset_name == "nbody_small_out_dist":
            self.suffix += "_charged5_initvel1small"
        else:
            raise Exception("Wrong dataset name %s" % self.dataset_name)

        self.max_samples = int(max_samples)
        self.dataset_name = dataset_name
        self.data, self.edges = self.load()

    def load(
        self,
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        List[torch.Tensor],
    ]:
        """
        Load the N-Body simulation data.

        Returns:
            tuple: A tuple containing the loaded data and edges.
        """
        loc = np.load(DATA_PATH / f"n_body_system/dataset/loc_{self.suffix}.npy")
        vel = np.load(DATA_PATH / f"n_body_system/dataset/vel_{self.suffix}.npy")
        edges = np.load(DATA_PATH / f"n_body_system/dataset/edges_{self.suffix}.npy")
        charges = np.load(
            DATA_PATH / f"n_body_system/dataset/charges_{self.suffix}.npy"
        )

        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, charges)
        return (
            torch.Tensor(loc),
            torch.Tensor(vel),
            torch.Tensor(edge_attr),
            torch.Tensor(charges),
        ), edges

    def preprocess(
        self, loc: np.ndarray, vel: np.ndarray, edges: np.ndarray, charges: np.ndarray
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor
    ]:
        """
        Preprocess the loaded data.

        Args:
            loc (np.ndarray): Array of node locations.
            vel (np.ndarray): Array of node velocities.
            edges (np.ndarray): Array of edges.
            charges (np.ndarray): Array of charges.

        Returns:
            tuple: A tuple containing the preprocessed data.
        """
        # cast to torch and swap n_nodes <--> n_features dimensions
        loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
        n_nodes = loc.size(2)
        loc = loc[
            0 : self.max_samples, :, :, :
        ]  # limit number of samples, max_samples x 49 x 5 x 3
        vel = vel[
            0 : self.max_samples, :, :, :
        ]  # speed when starting the trajectory, max_samples x 49 x 5 x 3
        charges = charges[0 : self.max_samples]  # max_samples x 5 x 1
        edge_attr = []

        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_attr.append(edges[: self.max_samples, i, j])
                    rows.append(i)
                    cols.append(j)

        edges = [rows, cols]
        edge_attr = torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(2)

        return (
            torch.Tensor(loc),
            torch.Tensor(vel),
            torch.Tensor(edge_attr),
            edges,
            torch.Tensor(charges),
        )

    def set_max_samples(self, max_samples: int) -> None:
        """
        Set the maximum number of samples to load.

        Args:
            max_samples (int): Maximum number of samples.
        """
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()

    def get_n_nodes(self) -> int:
        """
        Get the number of nodes in the dataset.

        Returns:
            int: Number of nodes.
        """
        return self.data[0].size(1)

    def __getitem__(
        self, i: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get an item from the dataset.

        Args:
            i (int): Index of the item.

        Returns:
            tuple: A tuple containing the item data.
        """
        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr, charges = loc[i], vel[i], edge_attr[i], charges[i]

        if self.dataset_name == "nbody":
            frame_0, frame_T = 6, 8
        elif self.dataset_name == "nbody_small":
            frame_0, frame_T = 30, 40
        elif self.dataset_name == "nbody_small_out_dist":
            frame_0, frame_T = 20, 30
        else:
            raise Exception("Wrong dataset partition %s" % self.dataset_name)

        return loc[frame_0], vel[frame_0], edge_attr, charges, loc[frame_T]

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data[0])

    def get_edges(self, batch_size: int, n_nodes: int) -> List[torch.Tensor]:
        """
        Get the edges of the dataset.

        Args:
            batch_size (int): Batch size.
            n_nodes (int): Number of nodes.

        Returns:
            list: A list containing the edges.
        """
        edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges


class NBodyDataModule(pl.LightningDataModule):
    """
    Data module for N-Body simulation data.
    """

    def __init__(self, hyperparams: Any) -> None:
        """
        Initialize the NBodyDataModule.

        Args:
            hyperparams (dict): Hyperparameters for the data module.
        """
        super().__init__()
        self.hyperparams = hyperparams

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup the data module.

        Args:
            stage (str): Stage of the data module ("fit" or "test").
        """
        if stage == "fit" or stage is None:
            self.train_dataset: NBodyDataset = NBodyDataset(partition="train")
            self.valid_dataset: NBodyDataset = NBodyDataset(partition="val")
        if stage == "test":
            self.test_dataset: NBodyDataset = NBodyDataset(partition="test")

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Get the train dataloader.

        Returns:
            torch.utils.data.DataLoader: Train dataloader.
        """
        train_loader: torch.utils.data.DataLoader = DataLoader(
            self.train_dataset,
            batch_size=self.hyperparams.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=11,
        )
        return train_loader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Get the validation dataloader.

        Returns:
            torch.utils.data.DataLoader: Validation dataloader.
        """
        train_loader: torch.utils.data.DataLoader = DataLoader(
            self.valid_dataset,
            batch_size=self.hyperparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=11,
        )
        return train_loader
