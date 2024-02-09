import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiStepLR

from omegaconf import DictConfig

from inference_utils import get_inference_method
from model_utils import get_canonicalization_network, get_canonicalizer, get_dataset_specific_info, get_prediction_network

# define the LightningModule
class ImageClassifierPipeline(pl.LightningModule):
    def __init__(self, hyperparams: DictConfig):
        super().__init__()
        
        self.loss, self.image_shape, self.num_classes = get_dataset_specific_info(hyperparams.dataset.dataset_name)

        self.prediction_network = get_prediction_network(
            architecture=hyperparams.prediction.prediction_network_architecture,
            dataset_name=hyperparams.dataset.dataset_name,
            use_pretrained=hyperparams.prediction.use_pretrained,
            freeze_encoder=hyperparams.prediction.freeze_pretrained_encoder,
            input_shape=self.image_shape,
            num_classes=self.num_classes
        )

        canonicalization_network = get_canonicalization_network(
            hyperparams.canonicalization_type, 
            hyperparams.canonicalization,
            self.image_shape,
        )
        
        self.canonicalizer = get_canonicalizer(
            hyperparams.canonicalization_type,
            canonicalization_network,
            hyperparams.canonicalization,
            self.image_shape
        )      
        
        self.hyperparams = hyperparams
        
        self.inference_method = get_inference_method(
            self.canonicalizer,
            self.prediction_network,
            self.num_classes,
            hyperparams.experiment.inference,
            self.image_shape
        )
        
        self.max_epochs = hyperparams.experiment.training.num_epochs
        
        self.save_hyperparameters()


    def training_step(self, batch: torch.Tensor):
        x, y = batch
        batch_size, num_channels, height, width = x.shape
        
        # assert that the input is in the right shape
        assert (num_channels, height, width) == self.image_shape

        training_metrics = {}
        loss = 0.0
        
        # canonicalize the input data
        # For the vanilla model, the canonicalization is the identity transformation
        x_canonicalized = self.canonicalizer(x)
        
        # add group contrast loss while using optmization based canonicalization method
        if self.hyperparams.canonicalization_type == 'opt_equivariant':
            group_contrast_loss = self.canonicalizer.get_group_contrast_loss()
            loss += group_contrast_loss * self.hyperparams.experiment.training.loss.group_contrast_weight
            training_metrics.update({"train/group_contrast_loss": group_contrast_loss})
        
        # Forward pass through the prediction network as you'll normally do
        logits = self.prediction_network(x_canonicalized)
        
        # calculate the task loss which is the cross-entropy loss for classification
        loss = self.loss(logits, y)
        
        # Add prior regularization loss if the prior weight is non-zero
        if self.hyperparams.experiment.training.loss.prior_weight:
            prior_loss = self.canonicalizer.get_prior_regularization_loss()
            loss += prior_loss * self.hyperparams.experiment.training.loss.prior_weight
            metric_identity = self.canonicalizer.get_identity_metric()
            training_metrics.update({
                    "train/prior_loss": prior_loss, 
                    "train/identity_metric": metric_identity
                })
            

        # Get the predictions and calculate the accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        
        training_metrics.update({
                "train/loss": loss,
                "train/acc": acc
            })
        
        # Log the training metrics
        self.log_dict(training_metrics, prog_bar=True)
        
        return {'loss': loss, 'acc': acc}
        

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