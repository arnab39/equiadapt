from torch import optim, nn
import numpy as np, math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import OneCycleLR, MultiStepLR
import torchvision
from canonical_network.utils import check_rotation_invariance, check_rotoreflection_invariance, save_images_class_wise
from examples.images.classification.model_utils import get_canonicalization_network, get_canonicalizer, get_dataset_specific_info, get_prediction_network

# define the LightningModule
class ImageClassifierPipeline(pl.LightningModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.save_hyperparameters()
        
        self.loss, self.image_shape, self.num_classes = get_dataset_specific_info(hyperparams.dataset.dataset_name)

        self.prediction_network = get_prediction_network(
            architecture=hyperparams.prediction.prediction_network_architecture,
            dataset_name=hyperparams.dataset.dataset_name,
            use_pretrained=hyperparams.prediction.use_pretrained,
            freeze_encoder=hyperparams.prediction.freeze_pretrained_encoder,
            input_shape=self.image_shape,
            num_classes=self.num_classes
        )

        self.canonicalization_network = get_canonicalization_network(
            hyperparams.canonicalization_type, hyperparams.canonicalization_kwargs
        )
        
        self.canonicalizer = get_canonicalizer(
            hyperparams.canonicalization_type,
            self.canonicalization_network,
        )      
        
        self.hyperparams = hyperparams


    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), self.im_shape[0], self.im_shape[1], self.im_shape[2])

        training_metrics = {}
        
        x_canonicalized = self.canonicalizer(x)
        
        if self.hyperparams.canonicalization_type == 'opt_equivariant':
            loss, group_contrast_loss = self.canonicalizer.add_group_contrast_loss(loss)
            training_metrics.update({"train/group_contrast_loss": group_contrast_loss})
        
        logits = self.prediction_network(x_canonicalized)
        loss = self.loss(logits, y)
        
        if self.hyperparams.experiment.loss.prior_weight:
            loss, prior_loss = self.canonicalizer.add_prior_regularizer(loss)
            metric_identity = self.canonicalizer.get_identity_metric()
            training_metrics.update({
                    "train/prior_loss": prior_loss, 
                    "train/identity_metric": metric_identity
                })
            

        
        preds = logits.argmax(dim=-1) if self.hyperparams.dataset.dataset_name != 'celeba' else logits > 0
        acc = (preds == y).float().mean()
        
        training_metrics.update({
                "train/loss": loss,
                "train/acc": acc
            })
        self.log_dict(training_metrics, prog_bar=True)
        
        return {
            'loss': loss, 
            'metric_identity': metric_identity, 
            'prior_loss': prior_loss, 
            'acc': acc}
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        if self.hyperparams.canonicalization_type == 'steerable':
            # regression training for continuous rotations
            self.pad = torchvision.transforms.Pad(math.ceil(x.shape[-2] * 0.3), padding_mode='edge')
            self.crop = torchvision.transforms.CenterCrop((x.shape[-2], x.shape[-1])) 

            # create a list of random rotations
            ## dataset_prior = torch.eye(2).to(self.hyperparams.device)
            dataset_prior = torch.rand((x.size(0), 1)).to(self.hyperparams.device)
            rand_rotations = dataset_prior * 2 * np.pi

            # complete training with rotated images
            # NOTE: if training is switched to normal images at any point, equivariance is lost
            x = self.pad(x)
            x = torch.stack([self.crop(torchvision.transforms.functional.rotate(x[i], rand_rotations[i].item() * 180 / np.pi)) for i in range(x.size(0))])
            
            logits, inverse_rotation, vectors = self.network(x) # vectors should match dataset prior
            
            if self.hyperparams.version == 'v0':
                # no prior-regularization
                loss = self.loss(logits, y)
                metric_identity = F.mse_loss(vectors, dataset_prior)
                prior_loss = metric_identity

            elif self.hyperparams.version == 'v1':
                # with prior-regularization
                dataset_prior_reshaped = dataset_prior.unsqueeze(0).repeat(inverse_rotation.size(0), 1, 1)
                if self.current_epoch < self.hyperparams.prior_train_epochs:
                    loss = F.mse_loss(inverse_rotation, dataset_prior.unsqueeze(0).repeat(inverse_rotation.size(0), 1, 1))
                else:
                    loss = self.loss(logits, y)
                metric_identity = F.mse_loss(inverse_rotation, dataset_prior.unsqueeze(0).repeat(inverse_rotation.size(0), 1, 1))
                prior_loss = metric_identity

            elif self.hyperparams.version == 'v2':
                # find the angle corresponding to predicted rotation matrix (-pi, pi)
                angles = torch.nan_to_num(torch.atan2(inverse_rotation[:, 0, 1], inverse_rotation[:, 0, 0]) / np.pi)
                
                # loss evaluation
                if self.current_epoch < self.hyperparams.prior_train_epochs:
                    loss = F.mse_loss((angles + 1)/2, dataset_prior)
                else:
                    if self.hyperparams.freeze_steerable_regression_model:
                        # freeze the steerable network once the training is done
                        # only train the pre-trained large model
                        for p in self.network.canonization_network.parameters():
                            p.requires_grad = False
                    loss = self.loss(logits, y)

                metric_identity = F.mse_loss((angles + 1)/2, dataset_prior)
                prior_loss = metric_identity

            elif self.hyperparams.version == 'v3':
                # loss evaluation for regression (angle prediction) canonization network
                loss = self.loss(logits, y) + F.mse_loss(vectors, dataset_prior)

                # metric evaluation
                metric_identity = F.mse_loss(vectors, dataset_prior)
                prior_loss = metric_identity

        elif self.hyperparams.canonicalization_type in ('equivariant', 'opt_equivariant'):
            logits, feature_fibres, _, vectors = self.network(x)

            if self.hyperparams.model == 'opt_equivariant':
                # shape of feature_fibres is (batch_size, group_size)
                if self.hyperparams.opt_type == 'cosine':
                    vectors = vectors.reshape(feature_fibres.shape[-1], -1, vectors.shape[-1]).permute((1, 0, 2)) # (batch_size, group_size, reference_vector_size)
                    distances = vectors @ vectors.permute((0, 2, 1))
                    mask = 1.0 - torch.eye(feature_fibres.shape[-1]).to(self.hyperparams.device) # (group_size, group_size)
                    group_contrast_loss = torch.abs(distances * mask).sum()
                    loss += 1e-4 * group_contrast_loss

                elif self.hyperparams.opt_type == 'hinge':
                    hinge_threshold = torch.tensor(self.hyperparams.hinge_threshold).to(self.hyperparams.device)
                    vectors = vectors.reshape(feature_fibres.shape[-1], -1, vectors.shape[-1]).permute((1, 0, 2)) # (batch_size, group_size, reference_vector_size)
                    distances = torch.cdist(vectors, vectors) # (batch_size, group_size, group_size)
                    mask = 1.0 - torch.eye(feature_fibres.shape[-1]).to(self.hyperparams.device) # (group_size, group_size)
                    hinged_similarity = torch.minimum(hinge_threshold, distances)
                    group_contrast_loss = ((hinge_threshold - hinged_similarity) * mask).mean()
                    loss += group_contrast_loss

            if self.hyperparams.group_type == 'roto-reflection':
                # take mean of the first half and the second half of soft activations
                feature_fibres = (feature_fibres[:, :self.hyperparams.num_rotations]
                                        + feature_fibres[:, self.hyperparams.num_rotations:])

            if self.hyperparams.version == 'v0':
                # no prior; no pretraining for mapping to prior
                if self.hyperparams.dataset == 'celeba':
                    loss += self.loss(logits, y.float())
                else:
                    loss += self.loss(logits, y)

            elif self.hyperparams.version == 'v-prior':
                # only prior; no task loss
                dataset_prior = torch.zeros((feature_fibres.size(0),), dtype=torch.long).to(self.hyperparams.device)
                loss = nn.CrossEntropyLoss()(feature_fibres, dataset_prior)

            # each of the following versions has a dataset prior
            # thus the loss in combined training needs a weighting factor
            elif self.hyperparams.version == 'v1':
                # combined training with dataset prior; no pretraining for mapping to prior
                # dataset prior is just a class prior
                dataset_prior = torch.zeros((feature_fibres.size(0),), dtype=torch.long).to(self.hyperparams.device)
                prior_loss = nn.CrossEntropyLoss()(feature_fibres, dataset_prior) # torch.mean(-(feature_fibres[:, 0][:, None] - feature_fibres[:, 1:])) #  # torch.mean(-feature_fibres[:, 0] + feature_fibres[:, 1:].mean(dim=-1)) # torch.mean(-feature_fibres[:, 0] + torch.logsumexp(feature_fibres[:, 1:], dim=-1))
                if self.hyperparams.dataset == 'celeba':
                    loss += self.loss(logits, y.float()) 
                    loss += self.hyperparams.prior_weight * prior_loss
                else:
                    loss += self.loss(logits, y) + self.hyperparams.prior_weight * self.loss(feature_fibres, dataset_prior)

            elif self.hyperparams.version == 'v2':
                # combined training with dataset prior; no pretraining for mapping to prior
                # dataset prior is just a class-wise probability prior
                dataset_prior = torch.zeros((self.hyperparams.num_rotations, ), dtype=torch.float64).to(self.hyperparams.device)
                base = 1
                scale = 0.2

                dataset_prior[0] = base
                for r in range(self.hyperparams.num_rotations // 4):
                    dataset_prior[r + 1] = base * scale**(r+1)
                    dataset_prior[self.hyperparams.num_rotations - 1 - r] = base * scale**(r+1)

                dataset_prior = dataset_prior/sum(dataset_prior)
                dataset_prior = dataset_prior.repeat(feature_fibres.size(0), 1)

                prior_loss = nn.CrossEntropyLoss()(feature_fibres, dataset_prior)
                if self.hyperparams.dataset == 'celeba':
                    loss += self.loss(logits, y.float()) 
                    loss += self.hyperparams.prior_weight * prior_loss
                else:
                    loss += self.loss(logits, y) 
                    loss += self.hyperparams.prior_weight * prior_loss

            elif self.hyperparams.version == 'v3':
                # combined training with dataset prior after pretraining; pretraining for mapping to prior
                # dataset prior is just a class prior
                dataset_prior = torch.zeros((feature_fibres.size(0),), dtype=torch.long).to(self.hyperparams.device)
                prior_loss = nn.CrossEntropyLoss()(feature_fibres, dataset_prior)
                if self.current_epoch < self.hyperparams.prior_train_epochs:
                    loss += prior_loss
                else:
                    if self.hyperparams.dataset == 'celeba':
                        loss += self.loss(logits, y.float()) 
                        loss += self.hyperparams.prior_weight * prior_loss
                    else:
                        loss += self.loss(logits, y) 
                        loss += self.hyperparams.prior_weight * prior_loss


            elif self.hyperparams.version == 'v4':
                # combined training with dataset prior after pretraining; pretraining for mapping to prior
                # dataset prior is just a class-wise probability prior
                dataset_prior = torch.zeros((self.hyperparams.num_rotations, ), dtype=torch.float64).to(self.hyperparams.device)
                base = 1
                scale = 0.2

                dataset_prior[0] = base
                for r in range(self.hyperparams.num_rotations // 4):
                    dataset_prior[r + 1] = base * scale**(r+1)
                    dataset_prior[self.hyperparams.num_rotations - 1 - r] = base * scale**(r+1)

                dataset_prior = dataset_prior/sum(dataset_prior)
                dataset_prior = dataset_prior.repeat(feature_fibres.size(0), 1)

                prior_loss = nn.CrossEntropyLoss()(feature_fibres, dataset_prior)
                if self.current_epoch < self.hyperparams.prior_train_epochs:
                    loss += prior_loss
                else:
                    if self.hyperparams.dataset == 'celeba':
                        loss += self.loss(logits, y.float()) 
                        loss += self.hyperparams.prior_weight * prior_loss
                    else:
                        loss += self.loss(logits, y) 
                        loss += self.hyperparams.prior_weight * prior_loss
            
            metric_identity = (feature_fibres.argmax(dim=-1) == 0).float().mean()
        
        else:
            logits = self.network(x)
            if self.hyperparams.dataset == 'celeba':
                loss += self.loss(logits, y.float())
            else:
                loss += self.loss(logits, y)

        if self.hyperparams.dataset == 'ImageNet' and self.hyperparams.version == 'v-prior':
            metrics = {"train/loss": loss, "train/identity_metric": metric_identity}
            if group_contrast_loss:
                metrics["train/group_contrast_loss"] = group_contrast_loss
            self.log_dict(metrics, prog_bar=True)
            return {'loss': loss, 'metric_identity': metric_identity}

        if self.hyperparams.dataset == 'celeba':
            preds = logits > 0
        else:
            preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()

        # Logging to TensorBoard by default
        if self.hyperparams.dataset == 'celeba':
            metrics = {"train/loss": loss, "train/identity_metric": metric_identity, "train/acc": acc, "train/prior_loss": prior_loss, "train/logit_loss": self.loss(logits, y.float())}
        else:
            metrics = {"train/loss": loss, "train/collapse": group_contrast_loss, "train/identity": metric_identity, "train/acc": acc, "train/prior": prior_loss, "train/logit": self.loss(logits, y)}
        self.log_dict(metrics, prog_bar=True)
        return {'loss': loss, 'metric_identity': metric_identity, 'prior_loss': prior_loss, 'acc': acc}

    # def training_epoch_end(self, training_step_outputs):
    #     avg_metric_identity = torch.stack([x['metric_identity'] for x in training_step_outputs]).mean()
    #     avg_acc = torch.stack([x['acc'] for x in training_step_outputs]).mean()
    #     avg_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
    #     avg_prior_loss = torch.stack([x['prior_loss'] for x in training_step_outputs]).mean()

    #     print(f'\nTraining Epoch {self.current_epoch} completed - avg_loss: {avg_loss:.5f}, avg_acc: {avg_acc:.5f}, avg_metric_identity: {avg_metric_identity:.5f}, avg_prior_loss: {avg_prior_loss:.5f}\n')
            

    def on_validation_start(self):
        self.all_distr = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), self.im_shape[0], self.im_shape[1], self.im_shape[2])
        metric_identity = 0
        distr = []

        if self.hyperparams.model == 'steerable':
            dataset_prior = torch.zeros((x.size(0), 1)).to(self.hyperparams.device)
            logits, inverse_rotation, vectors = self.network(x)
            metric_identity = F.mse_loss(vectors, dataset_prior)

        elif self.hyperparams.model in ('equivariant', 'opt_equivariant'):
            logits, feature_fibres, _, _ = self.network(x)
            if self.hyperparams.group_type == 'roto-reflection':
                # take mean of the first half and the second half of soft activations
                feature_fibres = (feature_fibres[:, :self.hyperparams.num_rotations]
                                        + feature_fibres[:, self.hyperparams.num_rotations:])
            metric_identity = (feature_fibres.argmax(dim=-1) == 0).float().mean()
            distr = feature_fibres.argmax(dim=-1)
        else:
            logits = self.network(x)

        if self.hyperparams.dataset == 'celeba':
            preds = logits > 0
        else:
            preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        

        # Logging to TensorBoard by default
        metrics = {"val/acc": acc, "val/identity_metric": metric_identity}
        # if len(distr) > 0:
        #     self.all_distr.append(distr.cpu().detach().numpy())
        self.log_dict(metrics)

        # NOTE: always save canonization (escnn) networks in eval mode()
        # otherwise there are unavoidable key mismatch errors
        if self.hyperparams.model in ('equivariant', 'opt_equivariant'):
            import os
            os.makedirs(os.path.join(self.hyperparams.checkpoint_path, self.hyperparams.checkpoint_name), exist_ok=True)
            # torch.save(self.network.canonization_network.state_dict(), os.path.join(self.hyperparams.checkpoint_path, self.hyperparams.checkpoint_name, f'cano_model_{self.current_epoch}.pt'))
            torch.save(self.network.scalar_network.state_dict(), os.path.join(self.hyperparams.checkpoint_path, self.hyperparams.checkpoint_name, f'cano_model_{self.current_epoch}.pt'))
        return metrics

    # def validation_epoch_end(self, validation_step_outputs):
    #     avg_metric_identity = torch.stack([x['val/identity_metric'] for x in validation_step_outputs]).mean()
    #     avg_acc = torch.stack([x['val/acc'] for x in validation_step_outputs]).mean()
    #     print(f'\nValidation Epoch {self.current_epoch} completed - avg_acc: {avg_acc:.5f}, avg_metric_identity: {avg_metric_identity:.5f}\n')
    
    # def on_validation_end(self):
    #     # accumulate the distr
    #     if self.hyperparams.model in ('equivariant', 'opt_equivariant'):
    #         # plot and save histogram
    #         bins_list = np.concatenate([x * 90 for x in self.all_distr])
    #         plt.hist(bins_list, bins=self.hyperparams.num_rotations, color='skyblue')
    #         plt.title("Histogram of output angles")
    #         plt.xlabel("Angles")
    #         plt.ylabel("Frequency")
    #         plt.savefig('histogram.png')


    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), self.im_shape[0], self.im_shape[1], self.im_shape[2])
        if self.hyperparams.model in ('equivariant', 'canonized_pca'):
            if self.hyperparams.check_invariance:
                if self.hyperparams.group_type == 'roto-reflection':
                    self.num_batches_invariant += check_rotoreflection_invariance(self.network, x, self.hyperparams.num_rotations)
                elif self.hyperparams.group_type == 'rotation':
                    self.num_batches_invariant += check_rotation_invariance(self.network, x, self.hyperparams.num_rotations)
                else:
                    raise ValueError('group_type not implemented for now.')
            if batch_idx == 0:
                self.image_buffer = x
                self.canonized_image_buffer = self.network.get_canonized_images(x)[0]
                self.y_buffer = y
            elif self.image_buffer.shape[0] < 2000:
                self.image_buffer = torch.cat((self.image_buffer, x), dim=0)
                canonized_images = self.network.get_canonized_images(x)[0]
                self.canonized_image_buffer = torch.cat((self.canonized_image_buffer, canonized_images), dim=0)
                self.y_buffer = torch.cat((self.y_buffer, y), dim=0)
                if self.image_buffer.shape[0] > 2000 and self.hyperparams.save_canonized_images:
                    save_images_class_wise(
                        self.image_buffer, self.y_buffer,
                        './canonical_network/visualization/' + self.hyperparams.dataset + '/' + self.hyperparams.model +
                        '/kernel_' +
                        str(self.hyperparams.canonization_kernel_size) + '_num_layers_' +
                        str(self.hyperparams.canonization_num_layers) +
                        '_' + self.hyperparams.group_type + '_' + str(self.hyperparams.num_rotations),
                        'original_images', self.num_classes
                    )
                    save_images_class_wise(
                        self.canonized_image_buffer, self.y_buffer,
                        './canonical_network/visualization/' + self.hyperparams.dataset + '/' + self.hyperparams.model +
                        '/kernel_' +
                        str(self.hyperparams.canonization_kernel_size) + '_num_layers_' +
                        str(self.hyperparams.canonization_num_layers) +
                        '_' + self.hyperparams.group_type + '_' + str(self.hyperparams.num_rotations),
                        'canonized_images', self.num_classes
                    )
                    print('saving canonized images')
        

        if self.hyperparams.eval_mode == 'so2_continuous':
            pass

        elif self.hyperparams.eval_mode == 'so2_classification':
            so2_inference = SO2Inference(self.hyperparams.eval_num_rotations, self.network, self.hyperparams.model, self.im_shape)
            # logit_dict, y_dict = so2_inference.infer_classification(x,y)
            logit_dict, y_dict, id_dict = so2_inference.infer_classification(x,y)
            acc_per_rotation = torch.zeros(self.hyperparams.eval_num_rotations)
            # loss_per_rotation = torch.zeros(self.hyperparams.eval_num_rotations)

            for rot_index, (key, logits) in enumerate(logit_dict.items()):
                if self.hyperparams.dataset == 'celeba':
                    preds = logits > 0
                else:
                    preds = logits.argmax(dim=-1)
                acc_per_rotation[rot_index] = (preds == y_dict[key]).float().mean()

            metrics = {"test/rot_acc": torch.mean(acc_per_rotation)}
            for i in range(self.hyperparams.eval_num_rotations):
                metrics[f'test/acc_rot_{i}'] = acc_per_rotation[i] if acc_per_rotation[i] > 0 else 0.0
            
            
            # calculate acc and acc_per_class for the original and rotated images separately
            # collect logits and y for the original images only by index 0
            logits = logit_dict[0]
            y = y_dict[0]

            for degree, deg_acc in id_dict.items():
                metrics[f"test/id_metric_{int(degree)}"] = deg_acc

            if self.hyperparams.dataset == 'celeba':
                preds = logits > 0
            else:
                preds = logits.argmax(dim=-1) 
            acc = (preds == y).float().mean()
            acc_per_class = torch.zeros(self.num_classes)
            for i in range(self.num_classes):
                if self.hyperparams.dataset == 'celeba':
                    acc_per_class[i] = (preds[:, i] == y[:, i]).float().mean()
                else:
                    acc_per_class[i] = (preds[y == i] == y[y == i]).float().mean()
            for i in range(self.num_classes):
                metrics[f'test/acc_class_{i}'] = acc_per_class[i] if acc_per_class[i] > 0 else 0.0
            
            metrics["test/acc"] = acc
            self.log_dict(metrics)
            print(metrics)
            return metrics

        elif self.hyperparams.eval_mode == 'o2_classification':
            o2_inference = O2Inference(self.hyperparams.eval_num_rotations, self.network, self.hyperparams.model, self.im_shape)
            logit_dict, y_dict = o2_inference.infer_classification(x,y)
            acc_per_rotation = torch.zeros(self.hyperparams.eval_num_rotations * 2)
            # loss_per_rotation = torch.zeros(self.hyperparams.eval_num_rotations * 2)

            for rot_ref_index, (key, logits) in enumerate(logit_dict.items()):
                if self.hyperparams.dataset == 'celeba':
                    preds = logits > 0
                else:
                    preds = logits.argmax(dim=-1)
                acc_per_rotation[rot_ref_index] = (preds == y_dict[key]).float().mean()

            metrics = {"test/rot_ref_acc": torch.mean(acc_per_rotation)}
            for i in range(self.hyperparams.eval_num_rotations * 2):
                metrics[f'test/acc_rot_ref_{i}'] = acc_per_rotation[i] if acc_per_rotation[i] > 0 else 0.0
            
            
            # calculate acc and acc_per_class for the original and rotated images separately
            # collect logits and y for the original images only by index 0
            logits = logit_dict[0]
            y = y_dict[0]

            if self.hyperparams.dataset == 'celeba':
                preds = logits > 0
            else:
                preds = logits.argmax(dim=-1)
            acc = (preds == y).float().mean()
            acc_per_class = torch.zeros(self.num_classes)
            for i in range(self.num_classes):
                if self.hyperparams.dataset == 'celeba':
                    acc_per_class[i] = (preds[:, i] == y[:, i]).float().mean()
                else:
                    acc_per_class[i] = (preds[y == i] == y[y == i]).float().mean()
            for i in range(self.num_classes):
                metrics[f'test/acc_class_{i}'] = acc_per_class[i] if acc_per_class[i] > 0 else 0.0

            metrics["test/acc"] = acc
            self.log_dict(metrics)
            return metrics

        else:
            logits = self.network(x)
            # loss = self.loss(logits, y)
            if self.hyperparams.dataset == 'celeba':
                preds = logits > 0
            else:
                preds = logits.argmax(dim=-1)
            acc = (preds == y).float().mean()
            acc_per_class = torch.zeros(self.num_classes)
            for i in range(self.num_classes):
                if self.hyperparams.dataset == 'celeba':
                    acc_per_class[i] = (preds[:, i] == y[:, i]).float().mean()
                else:
                    acc_per_class[i] = (preds[y == i] == y[y == i]).float().mean()
            metrics = {"test/acc": acc}
            for i in range(self.num_classes):
                metrics[f'test/acc_class_{i}'] = acc_per_class[i] if acc_per_class[i] > 0 else 0.0
            self.log_dict(metrics)
            return metrics

    def on_test_end(self):
        if self.hyperparams.model == 'equivariant' and self.hyperparams.check_invariance:
            print(f'Fraction of batches which are {self.hyperparams.group_type} '
                  f'invariant: {self.num_batches_invariant / self.trainer.num_test_batches[0]}')

    def forward(self, x):
        breakpoint() # ideally these functions are not called during training
        x = x.reshape(x.size(0), self.im_shape[0], self.im_shape[1], self.im_shape[2])
        logits = self.network(x)
        preds = logits.argmax(dim=-1)
        return preds

    def predict_step(self, batch, batch_idx, **kwargs):
        breakpoint() # ideally these functions are not called during training
        x, y = batch
        x = x.reshape(x.size(0), self.im_shape[0], self.im_shape[1], self.im_shape[2])
        logits = self.network(x)
        preds = logits.argmax(dim=-1)
        return preds

    def configure_optimizers(self):
        if 'resnet' in self.hyperparams.base_encoder and 'mnist' not in self.hyperparams.dataset:
            print('using SGD optimizer')
            if self.hyperparams.model == 'steerable':
                optimizer = torch.optim.SGD(
                    [
                        {'params': self.encoder.parameters()},
                        {'params': self.network.canonization_network.parameters(), 'lr': 5e-5},
                    ], 
                    lr=self.hyperparams.lr, 
                    momentum=0.9,
                    weight_decay=5e-4,
                )
            else:
                optimizer = torch.optim.SGD(
                    self.parameters(),
                    lr=self.hyperparams.lr, 
                    momentum=0.9,
                    weight_decay=5e-4,
                )

            steps_per_epoch = 50000 // self.hyperparams.batch_size + 50000 % self.hyperparams.batch_size
            scheduler_dict = {
                "scheduler": MultiStepLR(
                    optimizer,
                    # milestones=[self.trainer.max_epochs // 6, self.trainer.max_epochs // 3, self.trainer.max_epochs // 2],
                    milestones=[self.trainer.max_epochs // 3, self.trainer.max_epochs // 2], # for small training epochs
                    gamma=0.1,
                ),
                "interval": "epoch",
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        else:
            print(f'using Adam optimizer with lr={self.hyperparams.lr}')
            optimizer = optim.AdamW(self.parameters(), lr=self.hyperparams.lr)
            return optimizer