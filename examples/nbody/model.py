import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiStepLR
from examples.nbody.model_utils import get_canonicalization_network, get_prediction_network
import torch.nn as nn
from omegaconf import DictConfig

class NBodyPipeline(pl.LightningModule):
    def __init__(self, hyperparams: DictConfig):
        super().__init__()

        self.loss = nn.MSELoss()
        self.prediction_network = get_prediction_network(hyperparams)
        self.canonicalizer = get_canonicalization_network(hyperparams)
        self.hyperparams = hyperparams
        self.save_hyperparameters()

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
        batch = [d.view(-1, d.size(2)) for d in batch] # converts to 2D matrices
        loc, vel, edge_attr, charges, loc_end = batch
        edges = self.get_edges(batch_size, n_nodes) # returns a list of two tensors, each of size num_edges * batch_size (where num_edges is always 20, since G = K5)

        nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach() # norm of velocity vectors
        rows, cols = edges
        loc_dist = torch.sum((loc[rows] - loc[cols]) ** 2, 1).unsqueeze(1)  # relative distances among locations
        edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties

        # PIPELINE

        canonical_loc, canonical_vel = self.canonicalizer(nodes, loc, edges, vel, edge_attr, charges) # canonicalize the input data

        pred_loc = self.prediction_network(nodes, canonical_loc, edges, canonical_vel, edge_attr, charges) # predict the output

        outputs = self.canonicalizer.invert_canonicalization(pred_loc) # invert the canonicalization

        # outputs and loc_end are both (5*batch_size)x3
        loss = self.loss(outputs, loc_end)

        metrics = {"train/loss": loss}
        self.log_dict(metrics, on_epoch=True)

        return loss
    

    def validation_step(self, batch: torch.Tensor):
        x, y = batch
        
        batch_size, num_channels, height, width = x.shape
        
        # assert that the input is in the right shape
        assert (num_channels, height, width) == self.image_shape
            
        validation_metrics = {}
        
        # canonicalize the input data
        # For the vanilla model, the canonicalization is the identity transformation
        x_canonicalized = self.canonicalizer(x)
        
        # Forward pass through the prediction network as you'll normally do
        logits = self.prediction_network(x_canonicalized)

       # Get the predictions and calculate the accuracy 
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        
         # Log the identity metric if the prior weight is non-zero
        if self.hyperparams.experiment.training.loss.prior_weight:
            metric_identity = self.canonicalizer.get_identity_metric()
            validation_metrics.update({
                    "train/identity_metric": metric_identity
                })
            
        
        # Logging to TensorBoard by default
        validation_metrics.update({
            "val/acc": acc
        })
        
        self.log_dict(validation_metrics, prog_bar=True)

        return {'acc': acc}


    def test_step(self, batch: torch.Tensor):
        x, y = batch
        batch_size, num_channels, height, width = x.shape
        
        # assert that the input is in the right shape
        assert (num_channels, height, width) == self.image_shape

        test_metrics = self.inference_method.get_inference_metrics(x, y)
        
        # Log the test metrics
        self.log_dict(test_metrics, prog_bar=True)
        
        return test_metrics 
          

    def configure_optimizers(self):
        if 'resnet' in self.hyperparams.prediction.prediction_network_architecture and 'mnist' not in self.hyperparams.dataset.dataset_name:
            print('using SGD optimizer')
            optimizer = torch.optim.SGD(
                [
                    {'params': self.prediction_network.parameters(), 'lr': self.hyperparams.experiment.training.prediction_lr},
                    {'params': self.canonicalizer.parameters(), 'lr': self.hyperparams.experiment.training.canonicalization_lr},
                ], 
                momentum=0.9,
                weight_decay=5e-4,
            )
            
            if self.max_epochs > 100:
                milestones = [self.trainer.max_epochs // 6, self.trainer.max_epochs // 3, self.trainer.max_epochs // 2]
            else:
                milestones = [self.trainer.max_epochs // 3, self.trainer.max_epochs // 2] # for small training epochs
            
            scheduler_dict = {
                "scheduler": MultiStepLR(
                    optimizer,
                    milestones=milestones,
                    gamma=0.1,
                ),
                "interval": "epoch",
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        else:
            print(f'using Adam optimizer')
            optimizer = torch.optim.AdamW([
                    {'params': self.prediction_network.parameters(), 'lr': self.hyperparams.experiment.training.prediction_lr},
                    {'params': self.canonicalizer.parameters(), 'lr': self.hyperparams.experiment.training.canonicalization_lr},
                ])
            return optimizer