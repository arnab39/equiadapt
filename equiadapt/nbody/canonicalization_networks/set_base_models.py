from sched import scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter as ts
import pytorch_lightning as pl
import torchmetrics.functional as tmf
import wandb
import torch_scatter as ts
# import torchsort


class SequentialMultiple(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


# Set model base class

class BaseSetModel(pl.LightningModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.num_embeddings = hyperparams.num_embeddings
        self.num_layers = hyperparams.num_layers
        self.hidden_dim = hyperparams.hidden_dim
        self.out_dim = hyperparams.out_dim
        self.layer_pooling = hyperparams.layer_pooling
        self.final_pooling = hyperparams.final_pooling
        self.learning_rate = hyperparams.learning_rate if hasattr(hyperparams, "learning_rate") else None

    def get_predictions(self, x):
        return x

    def training_step(self, batch, batch_idx):
        inputs, indices, targets = batch
        output = self(inputs, indices, batch_idx)
        predictions = self.get_predictions(output).squeeze()

        loss = F.binary_cross_entropy(predictions, targets.to(torch.float32))
        accuracy = tmf.accuracy(predictions, targets)

        metrics = {"train/loss": loss, "train/accuracy": accuracy}
        self.log_dict(metrics, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, indices, targets = batch
        output = self(inputs, indices, batch_idx)
        predictions = self.get_predictions(output)

        loss = F.binary_cross_entropy(predictions, targets.to(torch.float32))
        accuracy = tmf.accuracy(predictions, targets)
        f1_score = tmf.f1_score(predictions, targets)

        if self.global_step == 0:
            wandb.define_metric("valid/loss", summary="min")
            wandb.define_metric("valid/accuracy", summary="max")
            wandb.define_metric("valid/f1_score", summary="max")

        metrics = {"valid/loss": loss, "valid/accuracy": accuracy, "valid/f1_score": f1_score}
        self.log_dict(metrics, prog_bar=True)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-10)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, min_lr=1e-6, mode="max")
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid/f1_score"}

    def validation_epoch_end(self, validation_step_outputs):
        scheduler = self.lr_schedulers()
        self.log("lr", scheduler.optimizer.param_groups[0]["lr"])

        predictions = validation_step_outputs[0]
        if self.current_epoch == 0:
            model_filename = (
                f"canonical_network/results/digits/onnx_models/{self.model}_{wandb.run.name}_{str(self.global_step)}.onnx"
            )
            torch.onnx.export(self, (self.dummy_input, self.dummy_indices, 0.0), model_filename, opset_version=12)
            wandb.save(model_filename)

        self.logger.experiment.log(
            {
                "valid/logits": wandb.Histogram(predictions.to("cpu")),
                "global_step": self.global_step,
            }
        )

# DeepSets model

class SetLayer(pl.LightningModule):
    def __init__(self, in_dim, out_dim, pooling="sum"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pooling = pooling

        self.identity_linear = nn.Linear(in_dim, out_dim)
        self.pooling_linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, set_indices):
        identity = self.identity_linear(x)

        pooled_set = ts.scatter(x, set_indices, 0, reduce=self.pooling)
        pooling = self.pooling_linear(pooled_set)
        pooling = torch.index_select(pooling, 0, set_indices)

        output = F.relu(identity + pooling) + x

        return output, set_indices


class DeepSets(BaseSetModel):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self.model = "deepsets"
        self.embedding_layer = nn.Embedding(self.num_embeddings, self.hidden_dim)
        self.set_layers = SequentialMultiple(
            *[SetLayer(self.hidden_dim, self.hidden_dim, self.layer_pooling) for i in range(self.num_layers - 1)]
        )
        self.output_layer = nn.Linear(self.hidden_dim, self.out_dim) if not self.out_dim == 1 else SequentialMultiple(nn.Linear(self.hidden_dim, self.out_dim), nn.Sigmoid())

        self.dummy_input = torch.zeros(1, device=self.device, dtype=torch.long)
        self.dummy_indices = torch.zeros(1, device=self.device, dtype=torch.long)

    def forward(self, x, set_indices, _):
        embeddings = self.embedding_layer(x)
        x, _ = self.set_layers(embeddings, set_indices)
        if self.final_pooling:
            x = ts.scatter(x, set_indices, reduce=self.final_pooling)
        output = self.output_layer(x)
        return output

# Transformer

class Transformer(BaseSetModel):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self.model = "transformer"
        self.embedding_layer = nn.Embedding(self.num_embeddings, self.hidden_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(self.hidden_dim, 8, self.hidden_dim, dropout=0,  batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, self.num_layers)
        self.output_layer = nn.Linear(self.hidden_dim, self.out_dim) if not self.out_dim == 1 else SequentialMultiple(nn.Linear(self.hidden_dim, self.out_dim), nn.Sigmoid())

        self.dummy_input = torch.zeros((1,1), device=self.device, dtype=torch.long)
        self.dummy_indices = torch.zeros((1,1), device=self.device, dtype=torch.long)

    def forward(self, x, mask, _):
        embeddings = self.embedding_layer(x)
        x = self.transformer_encoder(embeddings, src_key_padding_mask=mask)
        # FIXME : Implement this
        # if self.final_pooling:
        #     x = ts.scatter(x, set_indices, reduce=self.final_pooling)
        output = self.output_layer(x).squeeze() * mask
        return output
