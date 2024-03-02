from torch import optim, nn
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import OneCycleLR, MultiStepLR
from canonical_network.models.image_networks import VanillaNetwork, EquivariantCanonizationNetwork, \
    BasicConvEncoder, Identity, PCACanonizationNetwork, RotationEquivariantConvEncoder, OptimizationCanonizationNetwork
from canonical_network.models.resnet import resnet44
import torchvision
from canonical_network.utils import check_rotation_invariance, check_rotoreflection_invariance, save_images_class_wise

# define the LightningModule
class LitClassifier(pl.LightningModule):
    def __init__(self, hyperparams):
        super().__init__()
        #self.save_hyperparameters()
        pretrained = True if hyperparams.pretrained else False
        self.loss = nn.CrossEntropyLoss()
        if hyperparams.dataset == 'rotated_mnist':
            self.im_shape = (1, 28, 28)
            num_classes = 10
            if hyperparams.base_encoder == 'rotation_eqv_cnn':
                out_channels = hyperparams.num_channels
            else:
                out_channels = 32
        elif hyperparams.dataset in ('cifar10', 'cifar100'):
            self.im_shape = (3, 32, 32)
            num_classes = 10 if hyperparams.dataset == 'cifar10' else 100
            out_channels = 64
        else:
            raise ValueError('dataset not implemented for now.')
        if hyperparams.base_encoder == 'cnn':
            self.encoder = BasicConvEncoder(self.im_shape, out_channels)
        elif hyperparams.base_encoder == 'resnet18':
            self.encoder = torchvision.models.resnet18(pretrained=pretrained)
            self.set_start_layer(hyperparams)
            self.encoder.fc = nn.Identity()
        elif hyperparams.base_encoder == 'resnet44':
            self.encoder = resnet44()
            self.set_start_layer(hyperparams)
            self.encoder.fc = nn.Identity()
        elif hyperparams.base_encoder == 'resnet50':
            self.encoder = torchvision.models.resnet50(pretrained=pretrained)
            self.set_start_layer(hyperparams)
            self.encoder.fc = nn.Identity()
        elif hyperparams.base_encoder == 'resnet101':
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            self.set_start_layer(hyperparams)
            self.encoder.fc = Identity()
        elif hyperparams.base_encoder == 'wide_resnet':
            self.encoder = torchvision.models.wide_resnet50_2(pretrained=False)
            self.encoder.fc = Identity()
        elif hyperparams.base_encoder == 'vit':
            #TODO: implement vit
            self.encoder = torchvision.models.vit_base_patch16_224(pretrained=False)
            self.encoder.fc = Identity()
        elif hyperparams.base_encoder == 'rotation_eqv_cnn':
            self.encoder = RotationEquivariantConvEncoder(
                self.im_shape, out_channels,
                num_rotations=hyperparams.num_rotations,
                device=hyperparams.device
            )
        else:
            raise ValueError('base_encoder not implemented for now.')


        if hyperparams.model == 'vanilla':
            self.network = VanillaNetwork(self.encoder, self.im_shape, num_classes,
                                          hyperparams.batch_size, hyperparams.device)
        elif hyperparams.model == 'equivariant':
            self.network = EquivariantCanonizationNetwork(
                self.encoder, self.im_shape, num_classes,
                hyperparams.canonization_out_channels,
                hyperparams.canonization_num_layers,
                hyperparams.canonization_kernel_size,
                hyperparams.canonization_beta,
                hyperparams.group_type,
                hyperparams.num_rotations,
                hyperparams.device,
                hyperparams.batch_size
            )
        elif hyperparams.model == 'canonized_pca':
            self.network = PCACanonizationNetwork(
                self.encoder, self.im_shape, num_classes, hyperparams.batch_size
            )
        elif hyperparams.model == 'equivariant_optimization':
            self.network = OptimizationCanonizationNetwork(
                self.encoder, self.im_shape, num_classes,
                hyperparams
            )
        else:
            raise ValueError('model not implemented for now.')
        self.hyperparams = hyperparams
        self.num_classes = num_classes
        self.image_buffer = []
        self.canonized_image_buffer = []
        self.num_batches_invariant = 0.0

    def set_start_layer(self, hyperparams):
        if hyperparams.dataset in ('cifar10', 'cifar100'):
            self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.encoder.maxpool = nn.Identity()
        elif hyperparams.dataset == 'rotated_mnist':
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.encoder.maxpool = nn.Identity()


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        if self.hyperparams.data_mode == 'image':
            x, y = batch
        else:
            x, points, y = batch
        x = x.reshape(x.size(0), self.im_shape[0], self.im_shape[1], self.im_shape[2])

        logits = self.network(x) if self.hyperparams.data_mode == 'image' else self.network(x, points)
        loss = self.loss(logits, y)

        acc = (logits.argmax(dim=-1) == y).float().mean()
        # Logging to TensorBoard by default
        metrics = {"train/loss": loss, "train/acc": acc}
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.hyperparams.data_mode == 'image':
            x, y = batch
        else:
            x, points, y = batch
        x = x.reshape(x.size(0), self.im_shape[0], self.im_shape[1], self.im_shape[2])
        logits = self.network(x) if self.hyperparams.data_mode == 'image' else self.network(x, points)
        loss = self.loss(logits, y)
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        # Logging to TensorBoard by default
        metrics = {"val/loss": loss, "val/acc": acc}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        if self.hyperparams.data_mode == 'image':
            x, y = batch
        else:
            x, points, y = batch
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
                self.canonized_image_buffer, _ = self.network.get_canonized_images(x)
                self.y_buffer = y
            elif self.image_buffer.shape[0] < 2000:
                self.image_buffer = torch.cat((self.image_buffer, x), dim=0)
                canonized_images, _ = self.network.get_canonized_images(x)
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
        logits = self.network(x) if self.hyperparams.data_mode == 'image' else self.network(x, points)
        loss = self.loss(logits, y)
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        acc_per_class = torch.zeros(self.num_classes)
        for i in range(self.num_classes):
            acc_per_class[i] = (preds[y == i] == y[y == i]).float().mean()
        metrics = {"test/loss": loss, "test/acc": acc}
        for i in range(self.num_classes):
            metrics[f'test/acc_class_{i}'] = acc_per_class[i]
        self.log_dict(metrics)
        return metrics

    def on_test_end(self):
        if self.hyperparams.model == 'equivariant' and self.hyperparams.check_invariance:
            print(f'Fraction of batches which are {self.hyperparams.group_type} '
                  f'invariant: {self.num_batches_invariant / self.trainer.num_test_batches[0]}')

    def forward(self, x):
        x = x.reshape(x.size(0), self.im_shape[0], self.im_shape[1], self.im_shape[2])
        logits = self.network(x)
        preds = logits.argmax(dim=-1)
        return preds

    def predict_step(self, batch, batch_idx, **kwargs):
        x, y = batch
        x = x.reshape(x.size(0), self.im_shape[0], self.im_shape[1], self.im_shape[2])
        logits = self.network(x)
        preds = logits.argmax(dim=-1)
        return preds

    def configure_optimizers(self):
        if 'resnet' in self.hyperparams.base_encoder and 'mnist' not in self.hyperparams.dataset:
            print('using SGD optimizer')
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hyperparams.lr,
                momentum=0.9,
                weight_decay=5e-4,
            )
            #steps_per_epoch = 50000 // self.hyperparams.batch_size + 50000 % self.hyperparams.batch_size
            # scheduler_dict = {
            #     "scheduler": OneCycleLR(
            #         optimizer,
            #         0.1,
            #         epochs=self.trainer.max_epochs,
            #         steps_per_epoch=steps_per_epoch,
            #     ),
            #     "interval": "step",
            # }
            scheduler_dict = {
                "scheduler": MultiStepLR(
                    optimizer,
                    milestones=[self.trainer.max_epochs // 6, self.trainer.max_epochs // 3, self.trainer.max_epochs // 2],
                    gamma=0.1,
                ),
                "interval": "epoch",
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        else:
            print('using Adam optimizer')
            optimizer = optim.AdamW(self.parameters(), lr=self.hyperparams.lr)
            return optimizer
