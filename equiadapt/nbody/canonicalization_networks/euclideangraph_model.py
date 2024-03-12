import torch
import pytorch_lightning as pl

from equiadapt.nbody.canonicalization_networks.euclideangraph_base_models import EGNN_vel, GNN, VNDeepSets, BaseEuclideangraphModel, Transformer
from canonical_network.utils import define_hyperparams

# Input dim is 6 because location and velocity vectors are concatenated.
NBODY_HYPERPARAMS = {
    "learning_rate": 1e-4, #1e-3
    "weight_decay": 1e-12,
    "patience": 1000,
    "hidden_dim": 32, #32
    "input_dim": 6,
    "in_node_nf": 1,
    "in_edge_nf": 2,
    "num_layers": 2, #4
    "out_dim": 1,
    "canon_num_layers": 4,
    "canon_hidden_dim": 16,
    "canon_layer_pooling": "mean",
    "canon_final_pooling": "mean",
    "canon_nonlinearity": "relu",
    "canon_feature": "p",
    "canon_translation": False,
    "canon_angular_feature": 0,
    "canon_dropout": 0.5,
    "freeze_canon": False,
    "layer_pooling": "sum",
    "final_pooling": "mean",
    "nonlinearity": "relu",
    "angular_feature": "pv",
    "dropout": 0.5, #0
    "nheads": 8,
    "ff_hidden": 32
}

class EuclideangraphCanonFunction(pl.LightningModule):
    """
    Returns rotation matrix and translation vectors for canonicalization
    following eqns (9) and (10) in https://arxiv.org/pdf/2211.06489.pdf.
    """
    def __init__(self, hyperparams):
        super().__init__()
        self.model_type = hyperparams.canon_model_type
        self.num_layers = hyperparams.canon_num_layers
        self.hidden_dim = hyperparams.canon_hidden_dim
        self.layer_pooling = hyperparams.canon_layer_pooling
        self.final_pooling = hyperparams.canon_final_pooling
        self.nonlinearity = hyperparams.canon_nonlinearity
        self.canon_feature = hyperparams.canon_feature
        self.angular_feature = hyperparams.canon_angular_feature
        self.dropout = hyperparams.canon_dropout
        self.batch_size = hyperparams.batch_size
        self.canon_translation = hyperparams.canon_translation

        model_hyperparams = {
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "layer_pooling": self.layer_pooling,
            "final_pooling": self.final_pooling,
            "out_dim": 4,
            "batch_size": self.batch_size,
            "nonlinearity": self.nonlinearity,
            "canon_feature": self.canon_feature,
            "canon_translation": self.canon_translation,
            "angular_feature": self.angular_feature,
            "dropout": self.dropout,
        }

        self.model = {
            "EGNN": lambda: EGNN_vel(define_hyperparams(model_hyperparams)),
            "vndeepsets": lambda: VNDeepSets(define_hyperparams(model_hyperparams)),
        }[self.model_type]()

    def forward(self, nodes, loc, edges, vel, edge_attr, charges):
        """
        Returns rotation matrix and translation vectors, which are denoted as O and t respectively in eqn. 10
        of https://arxiv.org/pdf/2211.06489.pdf.

        Args:
            `nodes`: Norms of velocity vectors. Shape: (n_nodes*batch_size) x 1
            `loc`: Starting locations of nodes. Shape: (n_nodes*batch_size) x 3
            `edges`: list of length 2, where each element is a 2000 dimensional tensor
            `vel`: Starting velocities of nodes. Shape: (n_nodes*batch_size) x 3
            `edge_attr`: Products of charges and squared relative distances between adjacent nodes (each have their own column). Shape: (n_edges*batch_size) x 2
            `charges`: Charges of nodes . Shape: (n_nodes * batch_size) x 1
        """
        # (n_nodes * batch_size) x 3 x 3, (n_nodes * batch_size) x 3
        rotation_vectors, translation_vectors = self.model(nodes, loc, edges, vel, edge_attr, charges)
        # Apply gram schmidt to make vectors orthogonal for rotation matrix
        rotation_matrix = self.modified_gram_schmidt(rotation_vectors) # (n_nodes * batch_size) x 3 x 3

        return rotation_matrix, translation_vectors

    def gram_schmidt(self, vectors):
        v1 = vectors[:, 0]
        v1 = v1 / torch.norm(v1, dim=1, keepdim=True)
        v2 = vectors[:, 1] - torch.sum(vectors[:, 1] * v1, dim=1, keepdim=True) * v1
        v2 = v2 / torch.norm(v2, dim=1, keepdim=True)
        v3 = (
            vectors[:, 2]
            - torch.sum(vectors[:, 2] * v1, dim=1, keepdim=True) * v1
            - torch.sum(vectors[:, 2] * v2, dim=1, keepdim=True) * v2
        )
        v3 = v3 / torch.norm(v3, dim=1, keepdim=True)
        return torch.stack([v1, v2, v3], dim=1)

    def modified_gram_schmidt(self, vectors):
        v1 = vectors[:, 0]
        v1 = v1 / torch.norm(v1, dim=1, keepdim=True)
        v2 = vectors[:, 1] - torch.sum(vectors[:, 1] * v1, dim=1, keepdim=True) * v1
        v2 = v2 / torch.norm(v2, dim=1, keepdim=True)
        v3 = vectors[:, 2] - torch.sum(vectors[:, 2] * v1, dim=1, keepdim=True) * v1
        v3 = v3 - torch.sum(v3 * v2, dim=1, keepdim=True) * v2
        v3 = v3 / torch.norm(v3, dim=1, keepdim=True)
        return torch.stack([v1, v2, v3], dim=1)


class EuclideangraphPredFunction(pl.LightningModule):
    """
    Defines a neural network that makes predictions after canonicalization.
    """
    def __init__(self, hyperparams):
        super().__init__()
        self.model_type = hyperparams.pred_model_type
        self.num_layers = hyperparams.num_layers
        self.hidden_dim = hyperparams.hidden_dim
        self.input_dim = hyperparams.input_dim
        self.in_node_nf = hyperparams.in_node_nf
        self.in_edge_nf = hyperparams.in_edge_nf

        model_hyperparams = {
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "input_dim": self.input_dim,
            "in_node_nf": self.in_node_nf,
            "in_edge_nf": self.in_edge_nf,
        }

        self.model = {
            "GNN": lambda: GNN(define_hyperparams(model_hyperparams)),
            "EGNN": lambda: EGNN_vel(define_hyperparams(model_hyperparams)),
            "vndeepsets": lambda: VNDeepSets(define_hyperparams(model_hyperparams)),
            "Transformer": lambda: Transformer(define_hyperparams(model_hyperparams))
        }[self.model_type]()

    def forward(self, nodes, loc, edges, vel, edge_attr, charges):
        return self.model(nodes, loc, edges, vel, edge_attr, charges)


class EuclideanGraphModel(BaseEuclideangraphModel):
    def __init__(self, hyperparams):
        super(EuclideanGraphModel, self).__init__(hyperparams)
        self.model = "euclideangraph_model"
        self.hyperparams = hyperparams
        self.canon_translation = hyperparams.canon_translation

        self.canon_function = EuclideangraphCanonFunction(hyperparams)
        self.pred_function = EuclideangraphPredFunction(hyperparams)

        if hyperparams.freeze_canon:
            self.canon_function.freeze()

    def forward(self, nodes, loc, edges, vel, edge_attr, charges):
        """
        Returns predicted coordinates.

        Args:
            `nodes`: Norms of velocity vectors. Shape: (n_nodes*batch_size) x 1
            `loc`: Starting locations of nodes. Shape: (n_nodes*batch_size) x coord_dim
            `edges`: list of length 2, where each element is a 2000 dimensional tensor
            `vel`: Starting velocities of nodes. Shape: (n_nodes*batch_size) x vel_dim
            `edge_attr`: Products of charges and squared relative distances between adjacent nodes (each have their own column). Shape: (n_edges*batch_size) x 2
            `charges`: Charges of nodes . Shape: (n_nodes * batch_size) x 1
        """
        # Rotation and translation vectors from eqn (10) in https://arxiv.org/pdf/2211.06489.pdf.
        # Shapes: (n_nodes * batch_size) x 3 x 3 and (n_nodes * batch_size) x 3
        rotation_matrix, translation_vectors = self.canon_function(nodes, loc, edges, vel, edge_attr, charges)
        rotation_matrix_inverse = rotation_matrix.transpose(1, 2) # Inverse of a rotation matrix is its transpose.

        # Canonicalizes coordinates by rotating node coordinates and translation vectors by inverse rotation.
        # Shape: (n_nodes * batch_size) x coord_dim.
        canonical_loc = (
            torch.bmm(loc[:, None, :], rotation_matrix_inverse).squeeze()
            - torch.bmm(translation_vectors[:, None, :], rotation_matrix_inverse).squeeze()
        )
        # Canonicalizes velocities.
        # Shape: (n_nodes * batch_size) x vel_dim.
        canonical_vel = torch.bmm(vel[:, None, :], rotation_matrix_inverse).squeeze()

        # Makes prediction on canonical inputs.
        # Shape: (n_nodes * batch_size) x coord_dim.
        position_prediction = self.pred_function(nodes, canonical_loc, edges, canonical_vel, edge_attr, charges)

        # Applies rotation to predictions, following equation (10) from https://arxiv.org/pdf/2211.06489.pdf
        # Shape: (n_nodes * batch_size) x coord_dim.
        position_prediction = (
            torch.bmm(position_prediction[:, None, :], rotation_matrix).squeeze() + translation_vectors
        )

        return position_prediction
