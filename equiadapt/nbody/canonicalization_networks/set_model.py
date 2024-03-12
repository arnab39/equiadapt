
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter as ts
import pytorch_lightning as pl
from einops import rearrange
import wandb

from canonical_network.models.set_base_models import BaseSetModel, DeepSets, SequentialMultiple
from canonical_network.utils import define_hyperparams, dict_to_object

SET_HYPERPARAMS = {
    "learning_rate": 1e-3,
    "num_embeddings": 10,
    "canon_num_layers": 3,
    "canon_hidden_dim": 32,
    "num_clusters": 3,
    "canon_model_type": "deepsets",
    "canon_layer_pooling": "sum",
    "num_layers": 6,
    "hidden_dim": 64,
    "out_dim": 1,
    "layer_pooling": "sum",
    "final_pooling": "",
    "temperature_anneal": 0.0
}


class SetCanonFunction(pl.LightningModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.num_embeddings = hyperparams.num_embeddings
        self.num_layers = hyperparams.canon_num_layers
        self.hidden_dim = hyperparams.canon_hidden_dim
        self.num_clusters = hyperparams.num_clusters
        self.model_type = hyperparams.canon_model_type
        self.canon_layer_pooling = hyperparams.canon_layer_pooling
        self.temprature_anneal = hyperparams.temperature_anneal

        model_hyperparams = {
            "num_embeddings": self.num_embeddings,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "out_dim": self.num_clusters,
            "layer_pooling": self.canon_layer_pooling,
            "final_pooling": "",
        }

        self.model = {"deepsets": lambda: DeepSets(define_hyperparams(model_hyperparams))}[self.model_type]()

    def forward(self, x, set_indices, batch_idx):
        output = self.model(x, set_indices, batch_idx)
        temperature = self.get_temperature(batch_idx)
        return F.softmax(output / temperature, dim=1)

    def get_temperature(self, batch_idx):
        temperature = np.exp(-self.temprature_anneal * batch_idx)
        return np.max([0.1, temperature])


class SetPredictionLayer(pl.LightningModule):
    def __init__(self, in_dim, out_dim, num_clusters, batch_size, pooling):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_clusters = num_clusters
        self.batch_size = batch_size
        self.pooling = pooling

        self.identity_linear = nn.Linear(self.in_dim, self.out_dim)
        self.outer_linear = nn.Linear(self.in_dim * self.num_clusters, self.out_dim * self.num_clusters)

    def forward(self, x, clusters, set_indices):
        identity = self.identity_linear(x)

        cluster_matrix = torch.zeros((x.shape[0], self.batch_size, self.num_clusters))
        cluster_matrix[torch.arange(x.shape[0]), set_indices, :] = clusters

        pooled = torch.einsum("nbk,nf->bkf", cluster_matrix, x)
        pooled = rearrange(pooled, "b k f -> b (k f)")
        outer = self.outer_linear(pooled)
        outer = rearrange(outer, "b (k f) -> b k f", k=self.num_clusters, f=self.out_dim)
        outer = torch.einsum("nbk, bkf->nf", cluster_matrix, outer)

        # residual connection
        output = F.relu(identity + outer) + x

        return output, clusters, set_indices


class SetPredictionFunction(pl.LightningModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.batch_size = hyperparams.batch_size
        self.num_clusters = hyperparams.num_clusters
        self.num_embeddings = hyperparams.num_embeddings
        self.num_layers = hyperparams.num_layers
        self.hidden_dim = hyperparams.hidden_dim
        self.out_dim = hyperparams.out_dim
        self.layer_pooling = hyperparams.layer_pooling
        self.final_pooling = hyperparams.final_pooling

        self.embedding_layer = nn.Embedding(self.num_embeddings, self.hidden_dim)
        self.set_layers = SequentialMultiple(
            *[
                SetPredictionLayer(
                    self.hidden_dim, self.hidden_dim, self.num_clusters, self.batch_size, self.layer_pooling
                )
                for i in range(self.num_layers - 1)
            ]
        )
        self.output_layer = SequentialMultiple(nn.Linear(self.hidden_dim, self.out_dim), nn.Sigmoid())

    def forward(self, x, clusters, set_indices):
        embeddings = self.embedding_layer(x)
        x, _, _ = self.set_layers(embeddings, clusters, set_indices)
        if self.final_pooling:
            x = ts.scatter(x, set_indices, reduce=self.final_pooling)
        output = self.output_layer(x)
        return output.squeeze()


class SetModel(BaseSetModel):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self.model = "set_model"
        self.canon_function = SetCanonFunction(hyperparams)
        self.prediction_function = SetPredictionFunction(hyperparams)
        self.learning_rate = hyperparams.learning_rate
        self.batch_size = hyperparams.batch_size
        self.num_embeddings = hyperparams.num_embeddings

    def forward(self, x, set_indices, batch_idx):
        clusters = self.canon_function(x, set_indices, batch_idx)
        output = self.prediction_function(x, clusters, set_indices)

        return output, clusters

    def get_predictions(self, x):
        return x[0]

    def validation_epoch_end(self, validation_step_outputs):
        scheduler = self.lr_schedulers()
        self.log("lr", scheduler.optimizer.param_groups[0]["lr"])

        predictions, clusters = validation_step_outputs[0]
        if self.current_epoch == 0:
            dummy_input = torch.zeros(self.num_embeddings, device=self.device, dtype=torch.long)
            dummy_indices = torch.zeros(1, device=self.device, dtype=torch.long)
            model_filename = (
                f"canonical_network/results/digits/onnx_models/set_model_{wandb.run.name}_{str(self.global_step)}.onnx"
            )
            torch.onnx.export(self, (dummy_input, dummy_indices, 0.0), model_filename, opset_version=12)
            wandb.save(model_filename)

        self.logger.experiment.log(
            {
                "valid/logits": wandb.Histogram(predictions.to("cpu")),
                "valid/clusters": wandb.Histogram(clusters.to("cpu")),
                "global_step": self.global_step,
            }
        )


def main():
    model = SetModel(dict_to_object(SET_HYPERPARAMS))

    print(model(torch.arange(5), torch.LongTensor([0, 0, 0, 1, 1])))


if __name__ == "__main__":
    main()
