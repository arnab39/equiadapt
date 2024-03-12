import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import pathlib

SRC_PATH = pathlib.Path(__file__).parent.parent
DATA_PATH = SRC_PATH / "data"

class NBodyDataset():
    def __init__(self, partition='train', max_samples=3000, dataset_name="nbody_small"):
        self.partition = partition
        if self.partition == 'val':
            self.sufix = 'valid'
        else:
            self.sufix = self.partition
        self.dataset_name = dataset_name
        if dataset_name == "nbody":
            self.sufix += "_charged5_initvel1"
        elif dataset_name == "nbody_small" or dataset_name == "nbody_small_out_dist":
            self.sufix += "_charged5_initvel1small"
        else:
            raise Exception("Wrong dataset name %s" % self.dataset_name)

        self.max_samples = int(max_samples)
        self.dataset_name = dataset_name
        self.data, self.edges = self.load()

    def load(self):
        loc = np.load(DATA_PATH / f'n_body_system/dataset/loc_{self.sufix}.npy')
        vel = np.load(DATA_PATH / f'n_body_system/dataset/vel_{self.sufix}.npy')
        edges = np.load(DATA_PATH / f'n_body_system/dataset/edges_{self.sufix}.npy')
        charges = np.load(DATA_PATH / f'n_body_system/dataset/charges_{self.sufix}.npy')

        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, charges)
        return (loc, vel, edge_attr, charges), edges


    def preprocess(self, loc, vel, edges, charges):
        # cast to torch and swap n_nodes <--> n_features dimensions
        loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
        n_nodes = loc.size(2)
        loc = loc[0:self.max_samples, :, :, :]  # limit number of samples, max_samples x 49 x 5 x 3
        vel = vel[0:self.max_samples, :, :, :]  # speed when starting the trajectory, max_samples x 49 x 5 x 3
        charges = charges[0:self.max_samples] # max_samples x 5 x 1
        edge_attr = []

        # edges is currently 10000 x 5 x 5.
        # i believe M = edges[i,:,:] is a symmetric matrix where M[j][k] = charges[i][j] * charges[i][k]
        #Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                # if i != j, append charge(node_i)*charge(node_j) for all samples
                # and save the combination of row and col
                if i != j:
                    # edge_attr.append(edges[:, i, j]) #COMMENTED THIS OUT! SEEMED TO USE UNEEDED ROWS
                    # could we instead just use edge_attr.append(edges[:self.max_samples, i, j])? --> I think so. would save memory too
                    edge_attr.append(edges[:self.max_samples, i, j])
                    rows.append(i)
                    cols.append(j)
        # Once loop is over,
        # edge_attr = list all charge products between distinct nodes (20 x max_samples) (ie. product for all edges)
        # where edge_attr[i] = product of charges for node rows[i] and cols[i]
        edges = [rows, cols]
        edge_attr = torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(2) # swap n_nodes <--> batch_size and add nf dimension -> 10000 x 20 x 1

        return torch.Tensor(loc), torch.Tensor(vel), torch.Tensor(edge_attr), edges, torch.Tensor(charges)

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()

    def get_n_nodes(self):
        return self.data[0].size(1)

    def __getitem__(self, i):
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

    def __len__(self):
        return len(self.data[0])

    def get_edges(self, batch_size, n_nodes):
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
    def __init__(
        self, hyperparams
    ):
        super().__init__()
        self.hyperparams = hyperparams

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = NBodyDataset(partition="train")
            self.valid_dataset = NBodyDataset(partition="val")
        if stage == "test":
            self.test_dataset = NBodyDataset(partition="test")

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.hyperparams.batch_size, shuffle=True, drop_last=True, num_workers=11)
        return train_loader

    def val_dataloader(self):
        train_loader = DataLoader(self.valid_dataset, batch_size=self.hyperparams.batch_size, shuffle=False, drop_last=False, num_workers=11)
        return train_loader
