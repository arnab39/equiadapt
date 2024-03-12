import torch
import pytorch_lightning as pl
from examples.nbody.model_utils import (
    get_canonicalization_network,
    get_prediction_network,
    get_edges,
)
import torch.nn as nn
from omegaconf import DictConfig
from equiadapt.nbody.canonicalization.continuous_group import ContinuousGroupNBody


class NBodyPipeline(pl.LightningModule):
    def __init__(self, hyperparams: DictConfig):
        super().__init__()
        self.hyperparams = hyperparams
        self.prediction_network = get_prediction_network(hyperparams.pred_hyperparams)
        canonicalization_network = get_canonicalization_network(
            hyperparams.canon_hyperparams
        )

        self.canonicalizer = ContinuousGroupNBody(canonicalization_network, hyperparams)

        self.learning_rate = (
            hyperparams.learning_rate if hasattr(hyperparams, "learning_rate") else None
        )
        self.weight_decay = (
            hyperparams.weight_decay if hasattr(hyperparams, "weight_decay") else 0.0
        )
        self.patience = (
            hyperparams.patience if hasattr(hyperparams, "patience") else 100
        )

        self.loss = nn.MSELoss()
        # Each input has 5 particles. This list defines all the edges, since our graph is fully connected.
        # vertex at self.edges[0][i] has an edge connecting to self.edges[1][i]

    def training_step(self, batch: torch.Tensor):
        """
        Performs one training step.

        Args:
            `batch`: a list of tensors [loc, vel, edge_attr, charges, loc_end]
            `loc`: batch_size x n_nodes x 3
            `vel`: batch_size x n_nodes x 3
            `edge_attr`: batch_size x n_edges x 1
            `charges`: batch_size x n_nodes x 1
            `loc_end`: batch_size x n_nodes x 3
        """

        batch_size, n_nodes, _ = batch[0].size()
        batch = [d.view(-1, d.size(2)) for d in batch]  # converts to 2D matrices
        loc, vel, edge_attr, charges, loc_end = batch
        edges = get_edges(
            batch_size, n_nodes
        )  # returns a list of two tensors, each of size num_edges * batch_size (where num_edges is always 20, since G = K5)

        nodes = (
            torch.sqrt(torch.sum(vel**2, dim=1)).unsqueeze(1).detach()
        )  # norm of velocity vectors
        rows, cols = edges
        loc_dist = torch.sum((loc[rows] - loc[cols]) ** 2, 1).unsqueeze(
            1
        )  # relative distances among locations
        edge_attr = torch.cat(
            [edge_attr, loc_dist], 1
        ).detach()  # concatenate all edge properties

        # PIPELINE

        canonical_loc, canonical_vel = self.canonicalizer(
            nodes, loc, edges, vel, edge_attr, charges
        )  # canonicalize the input data

        pred_loc = self.prediction_network(
            nodes, canonical_loc, edges, canonical_vel, edge_attr, charges
        )  # predict the output

        outputs = self.canonicalizer.invert_canonicalization(
            pred_loc
        )  # invert the canonicalization

        # outputs and loc_end are both (5*batch_size)x3
        loss = self.loss(outputs, loc_end)

        metrics = {"train/loss": loss}
        self.log_dict(metrics, on_epoch=True)

        return loss

    def validation_step(self, batch: torch.Tensor):
        batch_size, n_nodes, _ = batch[0].size()
        batch = [d.view(-1, d.size(2)) for d in batch]  # converts to 2D matrices
        loc, vel, edge_attr, charges, loc_end = batch
        edges = get_edges(
            batch_size, n_nodes
        )  # returns a list of two tensors, each of size num_edges * batch_size (where num_edges is always 20, since G = K5)

        nodes = (
            torch.sqrt(torch.sum(vel**2, dim=1)).unsqueeze(1).detach()
        )  # norm of velocity vectors
        rows, cols = edges
        loc_dist = torch.sum((loc[rows] - loc[cols]) ** 2, 1).unsqueeze(
            1
        )  # relative distances among locations
        edge_attr = torch.cat(
            [edge_attr, loc_dist], 1
        ).detach()  # concatenate all edge properties

        # PIPELINE

        canonical_loc, canonical_vel = self.canonicalizer(
            nodes, loc, edges, vel, edge_attr, charges
        )  # canonicalize the input data

        pred_loc = self.prediction_network(
            nodes, canonical_loc, edges, canonical_vel, edge_attr, charges
        )  # predict the output

        outputs = self.canonicalizer.invert_canonicalization(
            pred_loc
        )  # invert the canonicalization

        # outputs and loc_end are both (5*batch_size)x3
        loss = self.loss(outputs, loc_end)

        metrics = {"valid/loss": loss}
        self.log_dict(metrics, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {
                    "params": self.prediction_network.parameters(),
                    "lr": self.learning_rate,
                },
                {"params": self.canonicalizer.parameters(), "lr": self.learning_rate},
            ]
        )
        return optimizer
